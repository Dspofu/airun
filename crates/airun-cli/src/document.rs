//! Text out of attached documents, for the web UI's attachments and for the
//! `file` / `input_file` / `document` parts of the chat APIs. Plain text in
//! any common encoding and PDFs. PDF pages without a text layer (scans) come
//! out as page images, for a server with a vision encoder to read.

/// What came out of a document.
#[derive(Debug)]
pub struct Extracted {
    pub text: String,
    /// Page count, for PDFs.
    pub pages: Option<usize>,
    /// JPEG images of the PDF pages that have no text (scans), in page order.
    pub images: Vec<Vec<u8>>,
}

/// Scanned pages sent as images at most; each costs about a thousand tokens.
pub const MAX_PAGE_IMAGES: usize = 20;
/// Long side of a page image: about 110 dpi on A4, enough for body text.
const PAGE_IMAGE_SIDE: u32 = 1280;

/// Extracts the text of `bytes`. `name` only goes into the error messages.
pub fn extract(bytes: &[u8], name: &str) -> Result<Extracted, String> {
    let name = if name.is_empty() { "the file" } else { name };
    if bytes.is_empty() {
        return Err(format!("{name} is empty: attach a file with content"));
    }
    if is_pdf(bytes) {
        return pdf(bytes, name);
    }
    match decode_text(bytes) {
        Some(text) => Ok(Extracted { text: tidy(&text), pages: None, images: Vec::new() }),
        None => Err(format!(
            "{name} is neither text nor PDF, so its content cannot be read: save it as .txt, .md or .pdf \
             (Word and spreadsheets: File > Save As / Export) and attach it again")),
    }
}

fn is_pdf(bytes: &[u8]) -> bool {
    // The header may follow a few junk bytes; readers accept it in the first 1 KiB.
    bytes[..bytes.len().min(1024)].windows(5).any(|w| w == b"%PDF-")
}

fn pdf(bytes: &[u8], name: &str) -> Result<Extracted, String> {
    // The parsers panic on some malformed files; a bad attachment must not
    // take a server thread down with it.
    let texts = match std::panic::catch_unwind(|| pdf_extract::extract_text_from_mem_by_pages(bytes)) {
        Ok(Ok(p)) => Some(p),
        Ok(Err(e)) => {
            let e = e.to_string().to_ascii_lowercase();
            if e.contains("encrypt") || e.contains("password") {
                return Err(format!(
                    "{name} is password-protected: open it, save or print an unprotected copy, and attach that"));
            }
            None
        }
        Err(_) => None,
    };
    let scans = std::panic::catch_unwind(|| page_images(bytes)).unwrap_or_else(|_| PageImages::default());
    let n = texts.as_ref().map_or(scans.pages, Vec::len).max(scans.pages);
    if texts.is_none() && scans.pages == 0 {
        return Err(format!(
            "could not read {name} as a PDF: open it and save or print it to a new PDF \
             (Microsoft Print to PDF), or copy its text into the message"));
    }
    let mut text = String::new();
    let mut images = Vec::new();
    let mut skipped = Vec::new();
    for i in 0..n {
        let page = texts.as_ref().and_then(|t| t.get(i)).map(|p| tidy(p)).unwrap_or_default();
        let sep = if text.is_empty() { "" } else { "\n\n" };
        if !page.is_empty() {
            if n > 1 {
                text.push_str(&format!("{sep}[page {}]\n", i + 1));
            }
            text.push_str(&page);
        } else if let Some(Some(img)) = scans.by_page.get(i) {
            if images.len() < MAX_PAGE_IMAGES {
                images.push(img.clone());
                text.push_str(&format!("{sep}[page {}: scanned, sent as image {}]", i + 1, images.len()));
            } else {
                skipped.push(i + 1);
            }
        }
    }
    if !skipped.is_empty() {
        text.push_str(&format!(
            "\n\n[pages {}-{} are scans too, not sent: over {MAX_PAGE_IMAGES} images]",
            skipped[0], skipped[skipped.len() - 1]));
    }
    if text.trim().is_empty() {
        return Err(match scans.unsupported.as_deref() {
            Some(f) => format!(
                "{name} has no text layer and its scanned pages use {f}, which airun does not decode: \
                 run it through OCR, or print it to a new PDF (Microsoft Print to PDF) and attach that"),
            None => format!(
                "{name} has no text and no page images airun can read (text drawn as shapes?): \
                 run it through OCR, or copy its text into the message"),
        });
    }
    Ok(Extracted { text, pages: Some(n), images })
}

// --- page images of scanned PDFs ----------------------------------------------

#[derive(Default)]
struct PageImages {
    pages: usize,
    /// The page's largest image, as JPEG, when it decoded.
    by_page: Vec<Option<Vec<u8>>>,
    /// A filter seen on a page image that is not decoded (JBIG2, JPEG 2000).
    unsupported: Option<String>,
}

fn page_images(bytes: &[u8]) -> PageImages {
    let Ok(doc) = lopdf::Document::load_mem(bytes) else { return PageImages::default() };
    let pages = doc.get_pages();
    let mut out = PageImages { pages: pages.len(), ..Default::default() };
    for (_, &page_id) in &pages {
        let mut found: Vec<&lopdf::Stream> = Vec::new();
        if let Ok((own, inherited)) = doc.get_page_resources(page_id) {
            let mut dicts: Vec<&lopdf::Dictionary> = own.into_iter().collect();
            dicts.extend(inherited.iter().filter_map(|&id| doc.get_dictionary(id).ok()));
            for res in dicts {
                collect_images(&doc, res, 0, &mut found);
            }
        }
        // A scan is the page's one large image; logos and stamps are smaller.
        found.sort_by_key(|s| std::cmp::Reverse(dim(&s.dict, b"Width") * dim(&s.dict, b"Height")));
        let mut img = None;
        for s in found.iter().take(3) {
            match decode_image(&doc, s) {
                Ok(i) => { img = Some(i); break; }
                Err(Some(f)) => { out.unsupported.get_or_insert(f); }
                Err(None) => {}
            }
        }
        out.by_page.push(img.and_then(|i| to_jpeg(i)));
    }
    out
}

fn dim(d: &lopdf::Dictionary, key: &[u8]) -> i64 {
    d.get(key).ok().and_then(|o| o.as_i64().ok()).unwrap_or(0)
}

/// Image XObjects of a resource dictionary, and of the forms it draws (scans
/// are sometimes wrapped in one).
fn collect_images<'a>(doc: &'a lopdf::Document, res: &'a lopdf::Dictionary, depth: u32,
                      out: &mut Vec<&'a lopdf::Stream>) {
    let Ok(xo) = res.get(b"XObject").and_then(|o| doc.dereference(o)).and_then(|(_, o)| o.as_dict()) else { return };
    for (_, v) in xo.iter() {
        let Ok((_, obj)) = doc.dereference(v) else { continue };
        let Ok(s) = obj.as_stream() else { continue };
        match s.dict.get(b"Subtype").and_then(|o| o.as_name()) {
            Ok(b"Image") if dim(&s.dict, b"Width") * dim(&s.dict, b"Height") >= 200 * 200 => out.push(s),
            Ok(b"Form") if depth < 3 => {
                if let Ok(r) = s.dict.get(b"Resources").and_then(|o| doc.dereference(o)).and_then(|(_, o)| o.as_dict()) {
                    collect_images(doc, r, depth + 1, out);
                }
            }
            _ => {}
        }
    }
}

/// The stream's filter names, in decoding order.
fn filters(s: &lopdf::Stream) -> Vec<Vec<u8>> {
    match s.dict.get(b"Filter") {
        Ok(lopdf::Object::Name(n)) => vec![n.clone()],
        Ok(lopdf::Object::Array(a)) => a.iter().filter_map(|o| o.as_name().ok().map(<[u8]>::to_vec)).collect(),
        _ => Vec::new(),
    }
}

/// Decodes an image XObject. Err(Some(filter)) names a filter airun does not
/// decode; Err(None) is anything else that did not work.
fn decode_image(doc: &lopdf::Document, s: &lopdf::Stream) -> Result<image::DynamicImage, Option<String>> {
    let f = filters(s);
    let last = f.last().map(Vec::as_slice);
    match last {
        Some(b"DCTDecode") | Some(b"DCT") if f.len() == 1 => {
            image::load_from_memory_with_format(&s.content, image::ImageFormat::Jpeg).map_err(|_| None)
        }
        Some(b"CCITTFaxDecode") | Some(b"CCF") if f.len() == 1 => ccitt(doc, s).ok_or(None),
        Some(b"JBIG2Decode") => Err(Some("JBIG2 compression".into())),
        Some(b"JPXDecode") => Err(Some("JPEG 2000 compression".into())),
        _ => {
            let data = s.decompressed_content().map_err(|_| None)?;
            raw(doc, &s.dict, &data).ok_or(None)
        }
    }
}

fn decode_params(doc: &lopdf::Document, s: &lopdf::Stream) -> Option<lopdf::Dictionary> {
    let p = s.dict.get(b"DecodeParms").or_else(|_| s.dict.get(b"DP")).ok()?;
    let (_, p) = doc.dereference(p).ok()?;
    match p {
        lopdf::Object::Dictionary(d) => Some(d.clone()),
        lopdf::Object::Array(a) => a.last().and_then(|o| doc.dereference(o).ok())
            .and_then(|(_, o)| o.as_dict().ok().cloned()),
        _ => None,
    }
}

/// The image's /Decode array says 1 is dark (inverted samples).
fn inverted(d: &lopdf::Dictionary) -> bool {
    d.get(b"Decode").and_then(|o| o.as_array()).ok()
        .and_then(|a| a.first()).and_then(|o| o.as_float().ok().or_else(|| o.as_i64().ok().map(|v| v as f32)))
        .is_some_and(|v| v > 0.5)
}

/// CCITT G3/G4 fax scans (most black-and-white office scanners).
fn ccitt(doc: &lopdf::Document, s: &lopdf::Stream) -> Option<image::DynamicImage> {
    let p = decode_params(doc, s).unwrap_or_default();
    let k = dim(&p, b"K");
    let w = match dim(&p, b"Columns") { 0 => 1728, c => c } as u32;
    let h = match dim(&p, b"Rows") { 0 => dim(&s.dict, b"Height"), r => r } as u32;
    let black_is_1 = p.get(b"BlackIs1").and_then(|o| o.as_bool()).unwrap_or(false);
    let mut px: Vec<u8> = Vec::with_capacity((w * h.max(1)) as usize);
    let mut line = |t: &[u32]| {
        px.extend(fax::decoder::pels(t, w).map(|c| if c == fax::Color::Black { 0u8 } else { 255 }));
    };
    let ok = if k < 0 {
        fax::decoder::decode_g4(s.content.iter().copied(), w, (h > 0).then_some(h), &mut line)
    } else {
        fax::decoder::decode_g3(s.content.iter().copied(), &mut line)
    };
    if ok.is_none() && px.is_empty() {
        return None;
    }
    let rows = (px.len() / w as usize) as u32;
    px.truncate((rows * w) as usize);
    // Fax "black" is ink; BlackIs1 or a /Decode [1 0] flips what the samples mean.
    let mask = s.dict.get(b"ImageMask").and_then(|o| o.as_bool()).unwrap_or(false);
    if black_is_1 != (inverted(&s.dict) && !mask) {
        px.iter_mut().for_each(|v| *v = 255 - *v);
    }
    image::GrayImage::from_raw(w, rows, px).map(image::DynamicImage::ImageLuma8)
}

/// Uncompressed (or Flate/LZW) samples in the common color spaces.
fn raw(doc: &lopdf::Document, d: &lopdf::Dictionary, data: &[u8]) -> Option<image::DynamicImage> {
    let (w, h) = (dim(d, b"Width") as usize, dim(d, b"Height") as usize);
    let mask = d.get(b"ImageMask").and_then(|o| o.as_bool()).unwrap_or(false);
    let bpc = if mask { 1 } else { match dim(d, b"BitsPerComponent") { 0 => 8, b => b as usize } };
    if w == 0 || h == 0 || !matches!(bpc, 1 | 2 | 4 | 8 | 16) {
        return None;
    }
    // Color space: (components, palette for /Indexed, base components of the palette, ink-like).
    enum Cs { Gray, Rgb, Cmyk, Indexed(Vec<u8>, usize), Ink }
    fn space(doc: &lopdf::Document, o: &lopdf::Object) -> Option<Cs> {
        let (_, o) = doc.dereference(o).ok()?;
        match o {
            lopdf::Object::Name(n) => match n.as_slice() {
                b"DeviceGray" | b"CalGray" | b"G" => Some(Cs::Gray),
                b"DeviceRGB" | b"CalRGB" | b"RGB" => Some(Cs::Rgb),
                b"DeviceCMYK" | b"CMYK" => Some(Cs::Cmyk),
                _ => None,
            },
            lopdf::Object::Array(a) => match a.first()?.as_name().ok()? {
                b"ICCBased" => {
                    let (_, st) = doc.dereference(a.get(1)?).ok()?;
                    match dim(&st.as_stream().ok()?.dict, b"N") { 1 => Some(Cs::Gray), 4 => Some(Cs::Cmyk), _ => Some(Cs::Rgb) }
                }
                b"CalGray" => Some(Cs::Gray),
                b"CalRGB" | b"Lab" => Some(Cs::Rgb),
                b"Separation" | b"DeviceN" => Some(Cs::Ink),
                b"Indexed" | b"I" => {
                    let base = space(doc, a.get(1)?)?;
                    let n = match base { Cs::Gray | Cs::Ink => 1, Cs::Cmyk => 4, _ => 3 };
                    let (_, lut) = doc.dereference(a.get(3)?).ok()?;
                    let lut = match lut {
                        lopdf::Object::String(b, _) => b.clone(),
                        lopdf::Object::Stream(s) => s.decompressed_content().ok()?,
                        _ => return None,
                    };
                    Some(Cs::Indexed(lut, n))
                }
                _ => None,
            },
            _ => None,
        }
    }
    let cs = if mask { Cs::Gray } else { space(doc, d.get(b"ColorSpace").ok()?)? };
    let comps = match &cs { Cs::Gray | Cs::Ink | Cs::Indexed(..) => 1, Cs::Rgb => 3, Cs::Cmyk => 4 };
    let row_bytes = (w * comps * bpc).div_ceil(8);
    if data.len() < row_bytes * h {
        return None;
    }
    let max = ((1u32 << bpc.min(8)) - 1) as u32;
    let sample = |row: &[u8], i: usize| -> u32 {
        match bpc {
            8 => row[i] as u32,
            16 => row[i * 2] as u32,
            _ => {
                let bit = i * bpc;
                ((row[bit / 8] >> (8 - bpc - bit % 8)) as u32) & max
            }
        }
    };
    let invert = inverted(d);
    let mut rgb = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        let row = &data[y * row_bytes..(y + 1) * row_bytes];
        for x in 0..w {
            let v = |c: usize| -> u8 {
                let s = sample(row, x * comps + c);
                let s = if invert && !matches!(cs, Cs::Indexed(..)) { max - s } else { s };
                (s * 255 / max) as u8
            };
            match &cs {
                Cs::Gray => { let g = if mask { 255 - v(0) } else { v(0) }; rgb.extend([g, g, g]) }
                Cs::Ink => { let g = 255 - v(0); rgb.extend([g, g, g]) }
                Cs::Rgb => rgb.extend([v(0), v(1), v(2)]),
                Cs::Cmyk => {
                    let k = 255 - v(3) as u32;
                    rgb.extend([0, 1, 2].map(|c| ((255 - v(c) as u32) * k / 255) as u8));
                }
                Cs::Indexed(lut, n) => {
                    let i = sample(row, x) as usize * n;
                    let e = |c: usize| lut.get(i + c).copied().unwrap_or(0);
                    match n { 1 => rgb.extend([e(0); 3]), 4 => {
                        let k = 255 - e(3) as u32;
                        rgb.extend([0, 1, 2].map(|c| ((255 - e(c) as u32) * k / 255) as u8));
                    } _ => rgb.extend([e(0), e(1), e(2)]) }
                }
            }
        }
    }
    image::RgbImage::from_raw(w as u32, h as u32, rgb).map(image::DynamicImage::ImageRgb8)
}

/// Scaled to PAGE_IMAGE_SIDE on the long side and stored as JPEG.
fn to_jpeg(img: image::DynamicImage) -> Option<Vec<u8>> {
    let img = if img.width().max(img.height()) > PAGE_IMAGE_SIDE {
        img.resize(PAGE_IMAGE_SIDE, PAGE_IMAGE_SIDE, image::imageops::FilterType::Triangle)
    } else {
        img
    };
    let mut out = std::io::Cursor::new(Vec::new());
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 85).encode_image(&img.to_rgb8()).ok()?;
    Some(out.into_inner())
}

/// UTF-8 (with or without BOM), UTF-16 with a BOM, or a single-byte legacy
/// encoding (Windows-1252) when the bytes are not UTF-8 but look like text.
fn decode_text(bytes: &[u8]) -> Option<String> {
    if let Some((enc, bom)) = encoding_rs::Encoding::for_bom(bytes) {
        return Some(enc.decode_without_bom_handling(&bytes[bom..]).0.into_owned());
    }
    if let Ok(s) = std::str::from_utf8(bytes) {
        return (!looks_binary(s.as_bytes())).then(|| s.to_string());
    }
    if looks_binary(bytes) {
        return None;
    }
    Some(encoding_rs::WINDOWS_1252.decode(bytes).0.into_owned())
}

/// NUL bytes, or many control characters, mean a binary format.
fn looks_binary(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(8192)];
    if head.contains(&0) {
        return true;
    }
    let ctrl = head.iter().filter(|&&b| b < 0x20 && !matches!(b, b'\n' | b'\r' | b'\t' | 0x0c)).count();
    ctrl * 20 > head.len()
}

/// Unix newlines, no trailing spaces, at most one blank line in a row.
fn tidy(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut blank = 0;
    for line in s.replace("\r\n", "\n").replace('\r', "\n").split('\n') {
        let line = line.trim_end();
        if line.is_empty() {
            blank += 1;
            if blank > 1 {
                continue;
            }
        } else {
            blank = 0;
        }
        out.push_str(line);
        out.push('\n');
    }
    out.trim_matches('\n').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_encodings() {
        assert_eq!(extract("olá\r\n\r\n\r\nmundo  ".as_bytes(), "a.txt").unwrap().text, "olá\n\nmundo");
        let mut bom = vec![0xEF, 0xBB, 0xBF];
        bom.extend_from_slice("ação".as_bytes());
        assert_eq!(extract(&bom, "b.txt").unwrap().text, "ação");
        // UTF-16 LE with BOM.
        let mut u16 = vec![0xFF, 0xFE];
        for c in "não".encode_utf16() { u16.extend_from_slice(&c.to_le_bytes()); }
        assert_eq!(extract(&u16, "c.txt").unwrap().text, "não");
        // Windows-1252: "café" with é = 0xE9.
        assert_eq!(extract(b"caf\xe9", "d.txt").unwrap().text, "café");
    }

    #[test]
    fn binary_and_empty_rejected() {
        assert!(extract(b"PK\x03\x04\x00\x00binary", "x.docx").unwrap_err().contains("Save As"));
        assert!(extract(b"", "e.txt").unwrap_err().contains("empty"));
    }

    #[test]
    fn broken_pdf_reports_what_to_do() {
        let e = extract(b"%PDF-1.4\n garbage", "f.pdf").unwrap_err();
        assert!(e.contains("f.pdf") && (e.contains("new PDF") || e.contains("OCR")), "{e}");
    }

    #[test]
    fn minimal_pdf() {
        let pdf = minimal_pdf_bytes("Hello airun");
        let out = extract(&pdf, "g.pdf").unwrap();
        assert_eq!(out.pages, Some(1));
        assert!(out.text.contains("Hello airun"), "{:?}", out.text);
    }

    #[test]
    fn scanned_page_becomes_image() {
        // A 400x300 gray page: white with a black band, as raw samples.
        let (w, h) = (400usize, 300usize);
        let px: Vec<u8> = (0..w * h).map(|i| if (100..200).contains(&(i / w)) { 0 } else { 255 }).collect();
        let img = [format!("<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceGray \
                            /BitsPerComponent 8 /Length {} >>\nstream\n", px.len()).into_bytes(), px,
                   b"\nendstream".to_vec()].concat();
        let pdf = pdf_bytes(&[
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /XObject << /Im0 4 0 R >> >> \
              /Contents 5 0 R >>".to_vec(),
            img,
            b"<< /Length 30 >>\nstream\nq 612 0 0 792 0 0 cm /Im0 Do Q\nendstream".to_vec(),
        ]);
        let out = extract(&pdf, "scan.pdf").unwrap();
        assert_eq!(out.images.len(), 1);
        assert!(out.text.contains("scanned, sent as image 1"), "{}", out.text);
        let back = image::load_from_memory(&out.images[0]).unwrap().to_luma8();
        assert_eq!((back.width(), back.height()), (400, 300));
        assert!(back.get_pixel(200, 150)[0] < 40 && back.get_pixel(200, 20)[0] > 215);
    }

    /// A one-page PDF with a Helvetica text line, with a correct xref table.
    pub(crate) fn minimal_pdf_bytes(line: &str) -> Vec<u8> {
        let content = format!("BT /F1 24 Tf 72 720 Td ({line}) Tj ET");
        pdf_bytes(&[
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> \
              /Contents 5 0 R >>".to_vec(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>".to_vec(),
            format!("<< /Length {} >>\nstream\n{content}\nendstream", content.len()).into_bytes(),
        ])
    }

    /// Objects 1..n with a correct xref table; object 1 is the catalog.
    fn pdf_bytes(objs: &[Vec<u8>]) -> Vec<u8> {
        let mut out = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (i, o) in objs.iter().enumerate() {
            offsets.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
            out.extend_from_slice(o);
            out.extend_from_slice(b"\nendobj\n");
        }
        let xref = out.len();
        out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).as_bytes());
        for off in offsets {
            out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objs.len() + 1).as_bytes());
        out
    }
}
