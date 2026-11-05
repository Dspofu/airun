//! Stream replay, as llama-server's `/v1/stream*` routes: a streamed
//! generation whose request carries an `X-Conversation-Id` header outlives
//! the HTTP connection. Its SSE bytes go into a per-conversation buffer, the
//! generation runs to its end even if the client goes away (a closed tab, a
//! dropped network), and a client that comes back reads the text from where
//! it stopped:
//!
//! - `GET /v1/stream/{id}?from=N` replays the buffer from byte N, then follows
//!   the live generation until it ends;
//! - `POST /v1/streams/lookup {"conversation_ids": [...]}` reports the
//!   sessions of ids the caller already knows (never lists others);
//! - `DELETE /v1/stream/{id}` is the user's Stop: it cancels the generation
//!   (here the GPU stops too, where llama-server lets it finish) and drops the
//!   buffer.
//!
//! [`tee`] is a middleware, so every SSE route (chat, completions, responses,
//! messages) gets it without its handler knowing.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::Request;
use axum::http::header::CONTENT_TYPE;
use axum::middleware::Next;
use axum::response::Response;
use tokio::sync::{Notify, mpsc};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

/// A finished session stays readable this long (llama-server's value).
const TTL_SECONDS: i64 = 300;
/// Bytes kept per conversation; past it the oldest are dropped, and a reader
/// asking for them is told the offset is lost.
const MAX_BYTES: usize = 4 * 1024 * 1024;
const GC_INTERVAL: Duration = Duration::from_secs(60);
/// How often a waiting reader checks that its client is still there.
const WAKE: Duration = Duration::from_millis(200);

pub const HEADER: &str = "x-conversation-id";

fn now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
}

pub struct Session {
    pub id: String,
    pub started_at: i64,
    inner: Mutex<Inner>,
    notify: Notify,
    cancelled: AtomicBool,
}

#[derive(Default)]
struct Inner {
    buf: Vec<u8>,
    /// Bytes dropped from the front to stay under MAX_BYTES.
    dropped: usize,
    done: bool,
    completed_at: i64,
}

/// What a read at some offset found.
enum Read {
    Data(Bytes),
    /// Nothing new yet; the generation is still running.
    Wait,
    End,
    /// The offset fell into the dropped prefix.
    Lost,
}

impl Session {
    fn new(id: String) -> Self {
        Self { id, started_at: now(), inner: Mutex::new(Inner::default()), notify: Notify::new(),
               cancelled: AtomicBool::new(false) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn append(&self, data: &[u8]) {
        {
            let mut s = self.lock();
            if s.done {
                return;
            }
            if data.len() >= MAX_BYTES {
                let skip = data.len() - MAX_BYTES;
                s.dropped += s.buf.len() + skip;
                s.buf.clear();
                s.buf.extend_from_slice(&data[skip..]);
            } else {
                let over = (s.buf.len() + data.len()).saturating_sub(MAX_BYTES);
                if over > 0 {
                    s.buf.drain(..over);
                    s.dropped += over;
                }
                s.buf.extend_from_slice(data);
            }
        }
        self.notify.notify_waiters();
    }

    fn finalize(&self) {
        {
            let mut s = self.lock();
            if s.done {
                return;
            }
            s.done = true;
            s.completed_at = now();
        }
        self.notify.notify_waiters();
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.finalize();
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    fn read(&self, offset: usize) -> Read {
        let s = self.lock();
        if offset < s.dropped {
            return Read::Lost;
        }
        let local = offset - s.dropped;
        if local < s.buf.len() {
            return Read::Data(Bytes::copy_from_slice(&s.buf[local..]));
        }
        if s.done || self.is_cancelled() { Read::End } else { Read::Wait }
    }

    pub fn is_done(&self) -> bool {
        self.lock().done
    }

    pub fn total_bytes(&self) -> usize {
        let s = self.lock();
        s.dropped + s.buf.len()
    }

    pub fn dropped_bytes(&self) -> usize {
        self.lock().dropped
    }

    pub fn completed_at(&self) -> i64 {
        self.lock().completed_at
    }
}

static SESSIONS: LazyLock<Mutex<HashMap<String, Arc<Session>>>> = LazyLock::new(Default::default);

fn sessions() -> std::sync::MutexGuard<'static, HashMap<String, Arc<Session>>> {
    SESSIONS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A fresh session for `id`; one already there (an earlier turn of the same
/// conversation still generating) is cancelled, as in llama-server.
fn create_or_replace(id: &str) -> Arc<Session> {
    let fresh = Arc::new(Session::new(id.to_string()));
    let previous = sessions().insert(id.to_string(), Arc::clone(&fresh));
    if let Some(p) = previous {
        p.cancel();
    }
    fresh
}

pub fn get(id: &str) -> Option<Arc<Session>> {
    sessions().get(id).cloned()
}

/// The sessions of the requested ids: the exact key and any `<id>::<model>`
/// variant, as llama-server matches them.
pub fn lookup(ids: &[String]) -> Vec<Arc<Session>> {
    let all: Vec<Arc<Session>> = sessions().values().cloned().collect();
    let mut out = Vec::new();
    for id in ids {
        let prefix = format!("{id}::");
        out.extend(all.iter().filter(|s| s.id == *id || s.id.starts_with(&prefix)).cloned());
    }
    out
}

/// The user's Stop: cancels the generation and drops the buffer. Idempotent.
pub fn cancel(id: &str) {
    let s = sessions().remove(id);
    if let Some(s) = s {
        s.cancel();
    }
}

/// Drops sessions finished more than TTL_SECONDS ago, every GC_INTERVAL.
pub fn spawn_gc() {
    tokio::spawn(async {
        let mut tick = tokio::time::interval(GC_INTERVAL);
        loop {
            tick.tick().await;
            let cutoff = now() - TTL_SECONDS;
            sessions().retain(|_, s| {
                let c = s.completed_at();
                c == 0 || c > cutoff
            });
        }
    });
}

/// The session's bytes from `from` on, then the live ones until it ends; the
/// reading stops as soon as the client goes away (the generation does not).
pub fn reader(session: Arc<Session>, from: usize) -> Body {
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, Infallible>>();
    tokio::spawn(async move {
        let mut offset = from;
        loop {
            // Registered before the read, so an append in between still wakes it.
            let notified = session.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            match session.read(offset) {
                Read::Data(b) => {
                    offset += b.len();
                    if tx.send(Ok(b)).is_err() {
                        return;
                    }
                }
                Read::End | Read::Lost => return,
                Read::Wait => {
                    tokio::select! {
                        _ = &mut notified => {}
                        _ = tokio::time::sleep(WAKE) => {
                            if tx.is_closed() {
                                return;
                            }
                        }
                    }
                }
            }
        }
    });
    Body::from_stream(UnboundedReceiverStream::new(rx))
}

/// The conversation id a request names, if any.
pub fn conversation_id(headers: &axum::http::HeaderMap) -> Option<String> {
    headers.get(HEADER).and_then(|v| v.to_str().ok()).map(str::trim).filter(|s| !s.is_empty()).map(String::from)
}

/// Middleware: a successful SSE response to a request with a conversation id
/// is drained into that conversation's session by a task of its own, which
/// keeps the generation alive without the client, and the client reads the
/// session like any later reader. A request that fails before streaming
/// leaves no session behind.
pub async fn tee(req: Request, next: Next) -> Response {
    let conv = conversation_id(req.headers());
    let res = next.run(req).await;
    let Some(conv) = conv else { return res };
    let sse = res.headers().get(CONTENT_TYPE).and_then(|v| v.to_str().ok())
        .is_some_and(|c| c.starts_with("text/event-stream"));
    if !sse || !res.status().is_success() {
        return res;
    }
    let session = create_or_replace(&conv);
    let (parts, body) = res.into_parts();
    let producer = Arc::clone(&session);
    tokio::spawn(async move {
        let mut stream = body.into_data_stream();
        loop {
            if producer.is_cancelled() {
                break; // dropping the body stops the generation at its next token
            }
            tokio::select! {
                chunk = stream.next() => match chunk {
                    Some(Ok(b)) => producer.append(&b),
                    _ => break,
                },
                _ = tokio::time::sleep(WAKE) => {}
            }
        }
        producer.finalize();
    });
    Response::from_parts(parts, reader(session, 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_buffer_drops_its_oldest_bytes_past_the_cap() {
        let s = Session::new("c".into());
        s.append(&vec![b'a'; MAX_BYTES - 10]);
        s.append(&[b'b'; 30]);
        assert_eq!(s.total_bytes(), MAX_BYTES + 20);
        assert_eq!(s.dropped_bytes(), 20);
        assert!(matches!(s.read(0), Read::Lost));
        match s.read(MAX_BYTES + 10) {
            Read::Data(b) => assert_eq!(&b[..], &[b'b'; 10]),
            _ => panic!("expected the last 10 bytes"),
        }
        assert!(matches!(s.read(MAX_BYTES + 20), Read::Wait));
        s.finalize();
        assert!(matches!(s.read(MAX_BYTES + 20), Read::End));
    }

    #[test]
    fn lookup_matches_the_id_and_its_model_variants_only() {
        create_or_replace("lk-a");
        create_or_replace("lk-a::qwen");
        create_or_replace("lk-ab");
        let mut ids: Vec<String> = lookup(&["lk-a".into()]).iter().map(|s| s.id.clone()).collect();
        ids.sort();
        assert_eq!(ids, ["lk-a", "lk-a::qwen"]);
        for id in ["lk-a", "lk-a::qwen", "lk-ab"] {
            cancel(id);
        }
        assert!(lookup(&["lk-a".into()]).is_empty());
    }

    #[test]
    fn a_new_turn_cancels_the_previous_one() {
        let first = create_or_replace("turns");
        let second = create_or_replace("turns");
        assert!(first.is_cancelled() && first.is_done());
        assert!(!second.is_cancelled());
        cancel("turns");
        assert!(second.is_cancelled());
    }
}
