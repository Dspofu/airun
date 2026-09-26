//! GGUF metadata values.

use std::fmt;

/// Wire type tags for GGUF metadata values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl ValueType {
    pub fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return None,
        })
    }
}

/// A GGUF metadata value.
///
/// Arrays are stored column-wise (one `Vec` per element type) rather than as
/// `Vec<Value>`. Tokenizer vocabularies run to hundreds of thousands of entries,
/// and the enum-per-element layout costs several times more memory for no gain.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Array),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Array {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
    String(Vec<String>),
    /// Nested arrays are legal in the format but never emitted in practice.
    Nested(Vec<Value>),
}

impl Array {
    pub fn len(&self) -> usize {
        match self {
            Array::U8(v) => v.len(),
            Array::I8(v) => v.len(),
            Array::U16(v) => v.len(),
            Array::I16(v) => v.len(),
            Array::U32(v) => v.len(),
            Array::I32(v) => v.len(),
            Array::U64(v) => v.len(),
            Array::I64(v) => v.len(),
            Array::F32(v) => v.len(),
            Array::F64(v) => v.len(),
            Array::Bool(v) => v.len(),
            Array::String(v) => v.len(),
            Array::Nested(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn element_type_name(&self) -> &'static str {
        match self {
            Array::U8(_) => "u8",
            Array::I8(_) => "i8",
            Array::U16(_) => "u16",
            Array::I16(_) => "i16",
            Array::U32(_) => "u32",
            Array::I32(_) => "i32",
            Array::U64(_) => "u64",
            Array::I64(_) => "i64",
            Array::F32(_) => "f32",
            Array::F64(_) => "f64",
            Array::Bool(_) => "bool",
            Array::String(_) => "string",
            Array::Nested(_) => "array",
        }
    }

    pub fn as_strings(&self) -> Option<&[String]> {
        match self {
            Array::String(v) => Some(v),
            _ => None,
        }
    }

    /// Integer arrays, widened to `i64`. Covers token-type and merge-rank fields
    /// which different converters emit at different widths.
    pub fn as_i64_vec(&self) -> Option<Vec<i64>> {
        Some(match self {
            Array::U8(v) => v.iter().map(|&x| x as i64).collect(),
            Array::I8(v) => v.iter().map(|&x| x as i64).collect(),
            Array::U16(v) => v.iter().map(|&x| x as i64).collect(),
            Array::I16(v) => v.iter().map(|&x| x as i64).collect(),
            Array::U32(v) => v.iter().map(|&x| x as i64).collect(),
            Array::I32(v) => v.iter().map(|&x| x as i64).collect(),
            Array::U64(v) => v.iter().map(|&x| x as i64).collect(),
            Array::I64(v) => v.clone(),
            _ => return None,
        })
    }

    pub fn as_f32_vec(&self) -> Option<Vec<f32>> {
        Some(match self {
            Array::F32(v) => v.clone(),
            Array::F64(v) => v.iter().map(|&x| x as f32).collect(),
            _ => return None,
        })
    }
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::U8(_) => "u8",
            Value::I8(_) => "i8",
            Value::U16(_) => "u16",
            Value::I16(_) => "i16",
            Value::U32(_) => "u32",
            Value::I32(_) => "i32",
            Value::U64(_) => "u64",
            Value::I64(_) => "i64",
            Value::F32(_) => "f32",
            Value::F64(_) => "f64",
            Value::Bool(_) => "bool",
            Value::String(_) => "string",
            Value::Array(_) => "array",
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Any integer or bool, widened to `i64`.
    ///
    /// Converters are inconsistent about width for the same logical key
    /// (`block_count` shows up as both u32 and u64 in the wild), so readers
    /// should always go through this rather than matching one variant.
    pub fn as_i64(&self) -> Option<i64> {
        Some(match *self {
            Value::U8(v) => v as i64,
            Value::I8(v) => v as i64,
            Value::U16(v) => v as i64,
            Value::I16(v) => v as i64,
            Value::U32(v) => v as i64,
            Value::I32(v) => v as i64,
            Value::U64(v) => v as i64,
            Value::I64(v) => v,
            Value::Bool(v) => v as i64,
            _ => return None,
        })
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self.as_i64() {
            Some(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_u64().and_then(|v| usize::try_from(v).ok())
    }

    pub fn as_f64(&self) -> Option<f64> {
        Some(match *self {
            Value::F32(v) => v as f64,
            Value::F64(v) => v,
            _ => return self.as_i64().map(|v| v as f64),
        })
    }

    pub fn as_f32(&self) -> Option<f32> {
        self.as_f64().map(|v| v as f32)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(v) => Some(v),
            _ => self.as_i64().map(|v| v != 0),
        }
    }

    pub fn as_array(&self) -> Option<&Array> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    /// One-line rendering for `inspect`. Long strings and arrays are elided so a
    /// 150k-entry vocabulary does not flood the terminal.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::U8(v) => write!(f, "{v}"),
            Value::I8(v) => write!(f, "{v}"),
            Value::U16(v) => write!(f, "{v}"),
            Value::I16(v) => write!(f, "{v}"),
            Value::U32(v) => write!(f, "{v}"),
            Value::I32(v) => write!(f, "{v}"),
            Value::U64(v) => write!(f, "{v}"),
            Value::I64(v) => write!(f, "{v}"),
            Value::F32(v) => write!(f, "{v}"),
            Value::F64(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::String(s) => {
                let flat = s.replace('\n', "\\n");
                if flat.chars().count() > 96 {
                    let head: String = flat.chars().take(93).collect();
                    write!(f, "\"{head}...\" ({} chars)", s.chars().count())
                } else {
                    write!(f, "\"{flat}\"")
                }
            }
            Value::Array(a) => {
                write!(f, "[{}; {}]", a.element_type_name(), a.len())?;
                match a {
                    Array::String(v) if !v.is_empty() => {
                        let preview: Vec<String> = v
                            .iter()
                            .take(4)
                            .map(|s| format!("{:?}", s.replace('\n', "\\n")))
                            .collect();
                        write!(f, " {}", preview.join(", "))?;
                        if v.len() > 4 {
                            write!(f, ", ...")?;
                        }
                    }
                    Array::I32(v) if !v.is_empty() && v.len() <= 8 => {
                        write!(f, " {v:?}")?;
                    }
                    Array::F32(v) if !v.is_empty() && v.len() <= 8 => {
                        write!(f, " {v:?}")?;
                    }
                    _ => {}
                }
                Ok(())
            }
        }
    }
}
