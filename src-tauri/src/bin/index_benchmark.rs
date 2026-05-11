use rayon::prelude::*;
use rusqlite::{params, Connection};
use std::{
    collections::{HashMap, VecDeque},
    env, fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const BLACKLISTED: &[&str] = &[
    "thumbnails",
    "thumbnail",
    "thumbs",
    ".thumbnails",
    "ipod photo cache",
    "apple tv photo cache",
    "photo cache",
    "previews",
    "preview",
    "heic-conversions",
    "derivatives",
    "renders",
    "proxies",
    "database",
    "contents",
    "backup",
    "private",
    "cpl",
    "cloudsharing",
    "journals",
    "__macosx",
    ".spotlight-v100",
    ".temporaryitems",
    ".trashes",
];
fn available_threads() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}
fn is_blacklisted_path(path: &Path) -> bool {
    path.components().any(|c| {
        c.as_os_str()
            .to_str()
            .map(|s| BLACKLISTED.iter().any(|b| s.eq_ignore_ascii_case(b)) || s.starts_with('.'))
            .unwrap_or(false)
    })
}
fn media_type_for_ext(ext: Option<&str>) -> Option<&'static str> {
    match ext
        .unwrap_or_default()
        .trim_start_matches('.')
        .to_ascii_lowercase()
        .as_str()
    {
        "jpg" | "jpeg" | "jpe" | "jfif" | "png" | "gif" | "webp" | "bmp" | "dib" | "tif"
        | "tiff" | "heic" | "heif" | "avif" | "svg" | "ico" | "raw" | "arw" | "cr2" | "cr3"
        | "nef" | "nrw" | "orf" | "rw2" | "raf" | "dng" | "pef" | "srw" => Some("image"),
        "mp4" | "mov" | "m4v" | "avi" | "mkv" | "webm" | "wmv" | "mpg" | "mpeg" | "3gp" | "3g2"
        | "mts" | "m2ts" | "ts" => Some("video"),
        _ => None,
    }
}
fn is_facetile_image_path(path: &Path) -> bool {
    if media_type_for_ext(path.extension().and_then(|e| e.to_str())) != Some("image") {
        return false;
    }
    const PREFIX: &str = "facetile";
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|name| {
            name.len() >= PREFIX.len() && name[..PREFIX.len()].eq_ignore_ascii_case(PREFIX)
        })
}
fn unix_time(t: SystemTime) -> Option<i64> {
    t.duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs() as i64)
}

fn discover_files(root: PathBuf, threads: usize, limit: usize) -> (Vec<PathBuf>, usize) {
    let queue = Arc::new(Mutex::new(VecDeque::from([root])));
    let files = Arc::new(Mutex::new(Vec::with_capacity(limit.min(100_000))));
    let active = Arc::new(AtomicUsize::new(0));
    let found = Arc::new(AtomicUsize::new(0));
    let errors = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::with_capacity(threads);
    for _ in 0..threads {
        let (queue, files, active, found, errors) = (
            queue.clone(),
            files.clone(),
            active.clone(),
            found.clone(),
            errors.clone(),
        );
        handles.push(thread::spawn(move || loop {
            if found.load(Ordering::Relaxed) >= limit {
                break;
            }
            let Some(dir) = queue.lock().unwrap().pop_front() else {
                if active.load(Ordering::SeqCst) == 0 {
                    break;
                }
                thread::sleep(Duration::from_millis(2));
                continue;
            };
            active.fetch_add(1, Ordering::SeqCst);
            let Ok(rd) = fs::read_dir(&dir) else {
                errors.fetch_add(1, Ordering::Relaxed);
                active.fetch_sub(1, Ordering::SeqCst);
                continue;
            };
            let mut discovered_dirs = Vec::new();
            let mut discovered_files = Vec::new();
            for e in rd.flatten() {
                let path = e.path();
                if is_blacklisted_path(&path) {
                    continue;
                }
                let Ok(ft) = e.file_type() else { continue };
                if ft.is_dir() {
                    discovered_dirs.push(path);
                    if discovered_dirs.len() >= 64 {
                        queue.lock().unwrap().extend(discovered_dirs.drain(..));
                    }
                } else if ft.is_file()
                    && media_type_for_ext(path.extension().and_then(|e| e.to_str())).is_some()
                    && !is_facetile_image_path(&path)
                {
                    let n = found.fetch_add(1, Ordering::Relaxed);
                    if n >= limit {
                        break;
                    }
                    discovered_files.push(path);
                    if discovered_files.len() >= 256 {
                        files.lock().unwrap().extend(discovered_files.drain(..));
                    }
                }
            }
            if !discovered_dirs.is_empty() {
                queue.lock().unwrap().extend(discovered_dirs);
            }
            if !discovered_files.is_empty() {
                files.lock().unwrap().extend(discovered_files);
            }
            active.fetch_sub(1, Ordering::SeqCst);
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    let out = Arc::try_unwrap(files).unwrap().into_inner().unwrap();
    (out, errors.load(Ordering::Relaxed))
}

#[derive(Clone)]
struct Row {
    path: String,
    name: String,
    ext: Option<String>,
    media_type: String,
    size: i64,
    created: Option<i64>,
    modified: Option<i64>,
}
#[derive(Clone, Copy)]
struct Fingerprint {
    size: i64,
    modified: Option<i64>,
}
enum ProcessResult {
    Changed,
    Unchanged,
}
fn clean_path_string(path: &Path) -> String {
    path.to_string_lossy()
        .trim_start_matches("\\\\?\\")
        .to_string()
}
fn process_file(p: &Path) -> Option<Row> {
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase());
    let mt = media_type_for_ext(ext.as_deref())?;
    if is_facetile_image_path(p) {
        return None;
    }
    let md = fs::metadata(p).ok()?;
    if !md.is_file() {
        return None;
    }
    Some(Row {
        path: clean_path_string(p),
        name: p.file_name()?.to_string_lossy().to_string(),
        ext,
        media_type: mt.into(),
        size: md.len() as i64,
        created: md.created().ok().and_then(unix_time),
        modified: md.modified().ok().and_then(unix_time),
    })
}
fn process_file_with_cache(
    p: &Path,
    existing: &HashMap<String, Fingerprint>,
) -> Option<ProcessResult> {
    media_type_for_ext(p.extension().and_then(|e| e.to_str()))?;
    if is_facetile_image_path(p) {
        return None;
    }
    let md = fs::metadata(p).ok()?;
    if !md.is_file() {
        return None;
    }
    let path = clean_path_string(p);
    let modified = md.modified().ok().and_then(unix_time);
    let size = md.len() as i64;
    if existing
        .get(&path)
        .is_some_and(|known| known.size == size && known.modified == modified)
    {
        return Some(ProcessResult::Unchanged);
    }
    Some(ProcessResult::Changed)
}

fn main() {
    let root = PathBuf::from(
        env::args()
            .nth(1)
            .unwrap_or_else(|| r"D:\TEMP\Input Photos".into()),
    );
    let limit: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let discovery_threads: usize = env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| (available_threads() * 4).clamp(8, 128));
    let indexing_threads: usize = env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| (available_threads() / 2).max(1).clamp(1, 4));
    println!(
        "benchmark root={} limit={} discovery_threads={} indexing_threads={}",
        root.display(),
        limit,
        discovery_threads,
        indexing_threads
    );
    let t0 = Instant::now();
    let (files, errors) = discover_files(root, discovery_threads, limit);
    let t_discovery = t0.elapsed();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(indexing_threads)
        .build()
        .unwrap();
    let t1 = Instant::now();
    let rows: Vec<Row> =
        pool.install(|| files.par_iter().filter_map(|p| process_file(p)).collect());
    let t_process = t1.elapsed();
    let t2 = Instant::now();
    let mut conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE media_items(path TEXT PRIMARY KEY,file_name TEXT,extension TEXT,media_type TEXT,size_bytes INTEGER,created_at INTEGER,modified_at INTEGER);").unwrap();
    let tx = conn.transaction().unwrap();
    for r in &rows {
        tx.execute(
            "INSERT OR REPLACE INTO media_items VALUES(?1,?2,?3,?4,?5,?6,?7)",
            params![
                r.path,
                r.name,
                r.ext,
                r.media_type,
                r.size,
                r.created,
                r.modified
            ],
        )
        .unwrap();
    }
    tx.commit().unwrap();
    let t_db = t2.elapsed();
    let total = t0.elapsed();
    let existing: HashMap<String, Fingerprint> = rows
        .iter()
        .map(|r| {
            (
                r.path.clone(),
                Fingerprint {
                    size: r.size,
                    modified: r.modified,
                },
            )
        })
        .collect();
    let t3 = Instant::now();
    let warm_results: Vec<ProcessResult> = pool.install(|| {
        files
            .par_iter()
            .filter_map(|p| process_file_with_cache(p, &existing))
            .collect()
    });
    let t_warm_process = t3.elapsed();
    let warm_changed = warm_results
        .iter()
        .filter(|r| matches!(r, ProcessResult::Changed))
        .count();
    let warm_unchanged = warm_results
        .iter()
        .filter(|r| matches!(r, ProcessResult::Unchanged))
        .count();
    println!(
        "discovered={} processed={} discovery_errors={}",
        files.len(),
        rows.len(),
        errors
    );
    println!(
        "discovery_ms={} process_ms={} db_ms={} total_ms={}",
        t_discovery.as_millis(),
        t_process.as_millis(),
        t_db.as_millis(),
        total.as_millis()
    );
    println!(
        "files_per_sec={:.1}",
        rows.len() as f64 / total.as_secs_f64()
    );
    println!(
        "warm_process_ms={} warm_changed={} warm_unchanged={} warm_files_per_sec={:.1}",
        t_warm_process.as_millis(),
        warm_changed,
        warm_unchanged,
        warm_results.len() as f64 / t_warm_process.as_secs_f64()
    );
}
