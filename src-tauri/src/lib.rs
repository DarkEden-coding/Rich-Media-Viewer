use rayon::prelude::*;
use rusqlite::{params, params_from_iter, Connection, OptionalExtension, ToSql};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    cmp::Ordering as CmpOrdering,
    collections::{HashMap, HashSet, VecDeque},
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc, Mutex, OnceLock,
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tauri::Emitter;
#[cfg(not(debug_assertions))]
use tauri::Manager;
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_window_state::StateFlags;

const DB_FILE: &str = "rich-media-viewer.sqlite3";
const HEIC_CONVERSIONS_DIR: &str = "heic-conversions";
/// Cosine threshold for InsightFace `buffalo_l` L2-normalized embeddings (inner product).
const INSIGHTFACE_FACE_MATCH_THRESHOLD: f64 = 0.42;
const FACE_EMBEDDING_MODEL: &str = "insightface-buffalo_l";
const DISCOVERY_THREAD_MULTIPLIER: usize = 4;
const INDEXING_THREAD_DIVISOR: usize = 2;
const EMBEDDING_BATCH_SIZE: usize = 256;
static FACE_SIDECARS: OnceLock<Mutex<Vec<FaceSidecarSlot>>> = OnceLock::new();
static CLEANUP_PLANS: OnceLock<Mutex<HashMap<String, CleanupPlan>>> = OnceLock::new();

fn backend_log(message: &str) {
    eprintln!("[rich-media-viewer backend] {message}");
}
fn available_threads() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}
fn discovery_thread_count() -> usize {
    std::env::var("RMV_DISCOVERY_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| available_threads().saturating_mul(DISCOVERY_THREAD_MULTIPLIER))
        .clamp(8, 128)
}
fn indexing_thread_count() -> usize {
    std::env::var("RMV_INDEXING_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| (available_threads() / INDEXING_THREAD_DIVISOR).max(1))
        .clamp(1, 4)
}
fn embedding_thread_count(provider: &str) -> usize {
    if provider.eq_ignore_ascii_case("fastembed") {
        std::env::var("RMV_EMBEDDING_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4usize)
            .clamp(1, 16)
    } else {
        indexing_thread_count()
    }
}
fn embedding_batch_size(provider: &str) -> usize {
    if provider.eq_ignore_ascii_case("fastembed") {
        std::env::var("RMV_FASTEMBED_BATCH_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16usize)
            .clamp(1, 256)
    } else {
        16
    }
}
fn face_indexing_thread_count() -> usize {
    std::env::var("RMV_FACE_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| indexing_thread_count().max(2))
        .clamp(1, 8)
}
fn cleanup_hash_thread_count() -> usize {
    std::env::var("RMV_CLEANUP_HASH_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| available_threads().saturating_mul(2).max(8))
        .clamp(1, 32)
}
fn cleanup_visual_thread_count() -> usize {
    std::env::var("RMV_CLEANUP_VISUAL_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| (available_threads() / 2).max(4))
        .clamp(1, 16)
}
const BLACKLISTED_FOLDER_NAMES: &[&str] = &[
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
fn is_excluded_path(path: &Path) -> bool {
    path.components().any(|c| {
        c.as_os_str()
            .to_str()
            .map(|s| {
                s.starts_with('.')
                    || BLACKLISTED_FOLDER_NAMES
                        .iter()
                        .any(|b| s.eq_ignore_ascii_case(b))
            })
            .unwrap_or(false)
    })
}
/// Google Photos-style face preview tiles (`facetile*.jpg`, etc.): omit from indexing and UI lists.
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
fn sql_path_not_blacklisted_clause() -> String {
    let mut clause: String = BLACKLISTED_FOLDER_NAMES
        .iter()
        .map(|name| format!(" AND lower(path) NOT LIKE '%{}%'", name.replace("'", "''")))
        .collect();
    clause.push_str(" AND file_name NOT LIKE '.%'");
    clause.push_str(" AND (media_type != 'image' OR lower(file_name) NOT LIKE 'facetile%')");
    clause
}

#[derive(Debug, Serialize)]
struct AppInfo {
    data_dir: String,
    database_path: String,
    index_exists: bool,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
struct MediaItem {
    id: i64,
    path: String,
    display_path: Option<String>,
    file_name: String,
    extension: Option<String>,
    media_type: String,
    size_bytes: Option<i64>,
    created_at: Option<i64>,
    modified_at: Option<i64>,
    imported_at: i64,
    missing: bool,
    camera_make: Option<String>,
    camera_model: Option<String>,
    lens_model: Option<String>,
    captured_at: Option<i64>,
    latitude: Option<f64>,
    longitude: Option<f64>,
    metadata_json: Option<String>,
}
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
struct AppSettings {
    library_folders: Vec<String>,
}
#[derive(Debug, Deserialize, Default)]
struct SearchFilter {
    query: Option<String>,
    media_type: Option<String>,
    missing: Option<bool>,
    from_ts: Option<i64>,
    to_ts: Option<i64>,
    camera: Option<String>,
    lat: Option<f64>,
    lng: Option<f64>,
    radius_km: Option<f64>,
    person_id: Option<i64>,
    person_name: Option<String>,
    has_gps: Option<bool>,
    has_camera: Option<bool>,
    sort_order: Option<String>,
    limit: Option<i64>,
    offset: Option<i64>,
}
#[derive(Debug, Serialize, Clone, Default)]
struct ScanSummary {
    scanned_files: usize,
    imported_or_updated: usize,
    skipped_files: usize,
    missing_marked: usize,
    errors: Vec<String>,
}
#[derive(Debug, Serialize, Clone)]
struct ScanProgress {
    phase: String,
    current_path: Option<String>,
    scanned_files: usize,
    imported_or_updated: usize,
    skipped_files: usize,
    missing_marked: usize,
    errors: usize,
    discovered_files: usize,
    total_files: Option<usize>,
    /// Images processed in the current / last face-embedding pass (parallel workers).
    faces_done: usize,
    /// Total images queued for face embedding in the current pass (`None` when not in that phase).
    faces_total: Option<usize>,
    done: bool,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
struct CleanupEntry {
    path: String,
    size_bytes: Option<i64>,
    reason: String,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
struct CleanupCandidate {
    path: String,
    display_path: Option<String>,
    file_name: String,
    size_bytes: Option<i64>,
    created_at: Option<i64>,
    modified_at: Option<i64>,
    captured_at: Option<i64>,
    width: Option<i64>,
    height: Option<i64>,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
struct CleanupDuplicateGroup {
    id: String,
    kind: String,
    reason: String,
    score: Option<i64>,
    default_keep_path: String,
    candidates: Vec<CleanupCandidate>,
}
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
struct CleanupTotals {
    ignored_files: usize,
    empty_folders: usize,
    duplicate_groups: usize,
    duplicate_files: usize,
    selected_files: usize,
    selected_folders: usize,
    selected_bytes: i64,
    errors: usize,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
struct CleanupPlan {
    plan_id: String,
    ignored_files: Vec<CleanupEntry>,
    empty_folders: Vec<CleanupEntry>,
    duplicate_groups: Vec<CleanupDuplicateGroup>,
    totals: CleanupTotals,
    errors: Vec<String>,
}
#[derive(Debug, Serialize, Clone)]
struct CleanupProgress {
    phase: String,
    current_path: Option<String>,
    processed: usize,
    total: Option<usize>,
    errors: usize,
    done: bool,
}
#[derive(Debug, Deserialize)]
struct ApplyCleanupSelections {
    remove_paths: Vec<String>,
    empty_folders: Vec<String>,
}
#[derive(Debug, Serialize)]
struct ApplyCleanupResult {
    files_deleted: usize,
    folders_deleted: usize,
    bytes_deleted: i64,
    rows_marked_missing: usize,
    errors: Vec<String>,
}
#[derive(Debug, Serialize)]
struct Person {
    id: i64,
    name: String,
    created_at: i64,
    face_count: i64,
}
#[derive(Debug, Serialize)]
struct Face {
    id: i64,
    media_item_id: i64,
    person_id: Option<i64>,
    person_name: Option<String>,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    confidence: Option<f64>,
    created_at: i64,
}
struct NamedFaceEmbedding {
    person_id: i64,
    embedding: Vec<f32>,
}
#[derive(Debug, Serialize)]
struct SidecarResult {
    ok: bool,
    stdout: String,
    stderr: String,
}
struct FaceSidecarProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}
struct FaceSidecarSlot {
    process: Option<FaceSidecarProcess>,
    in_use: bool,
}
#[derive(Debug, Serialize)]
struct SemanticHit {
    item: MediaItem,
    score: f64,
}
#[derive(Debug, Serialize)]
struct GeoPoint {
    latitude: f64,
    longitude: f64,
}

fn app_data_dir(_app: &tauri::AppHandle) -> Result<PathBuf, String> {
    #[cfg(debug_assertions)]
    {
        std::env::current_dir()
            .map(|p| p.join("dev-data"))
            .map_err(|e| format!("failed to resolve current dir: {e}"))
    }
    #[cfg(not(debug_assertions))]
    {
        _app.path()
            .app_data_dir()
            .map_err(|e| format!("failed to resolve app data dir: {e}"))
    }
}
fn db_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    Ok(app_data_dir(app)?.join(DB_FILE))
}
fn heic_conversions_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    Ok(app_data_dir(app)?.join(HEIC_CONVERSIONS_DIR))
}
fn open_db(app: &tauri::AppHandle) -> Result<Connection, String> {
    let dir = app_data_dir(app)?;
    fs::create_dir_all(&dir).map_err(|e| format!("failed to create app data dir: {e}"))?;
    let conn =
        Connection::open(dir.join(DB_FILE)).map_err(|e| format!("failed to open database: {e}"))?;
    init_db(&conn)?;
    Ok(conn)
}
fn init_db(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(r#"
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=60000;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS media_items(id INTEGER PRIMARY KEY AUTOINCREMENT,path TEXT NOT NULL UNIQUE,file_name TEXT NOT NULL,extension TEXT,media_type TEXT NOT NULL,size_bytes INTEGER,created_at INTEGER,modified_at INTEGER,imported_at INTEGER NOT NULL,missing INTEGER NOT NULL DEFAULT 0,camera_make TEXT,camera_model TEXT,latitude REAL,longitude REAL);
CREATE INDEX IF NOT EXISTS idx_media_items_path ON media_items(path); CREATE INDEX IF NOT EXISTS idx_media_items_type ON media_items(media_type);
CREATE TABLE IF NOT EXISTS people(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL,created_at INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS idx_people_name ON people(name COLLATE NOCASE);
CREATE TABLE IF NOT EXISTS faces(id INTEGER PRIMARY KEY AUTOINCREMENT,media_item_id INTEGER NOT NULL,person_id INTEGER,x REAL NOT NULL,y REAL NOT NULL,width REAL NOT NULL,height REAL NOT NULL,confidence REAL,created_at INTEGER NOT NULL,FOREIGN KEY(media_item_id) REFERENCES media_items(id) ON DELETE CASCADE,FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE SET NULL);
CREATE INDEX IF NOT EXISTS idx_faces_media_item_id ON faces(media_item_id);
CREATE INDEX IF NOT EXISTS idx_faces_person_media ON faces(person_id, media_item_id);
CREATE TABLE IF NOT EXISTS embeddings(id INTEGER PRIMARY KEY AUTOINCREMENT,media_item_id INTEGER,face_id INTEGER,model TEXT NOT NULL,vector BLOB NOT NULL,created_at INTEGER NOT NULL,FOREIGN KEY(media_item_id) REFERENCES media_items(id) ON DELETE CASCADE,FOREIGN KEY(face_id) REFERENCES faces(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL,updated_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS library_folders(id INTEGER PRIMARY KEY AUTOINCREMENT,path TEXT NOT NULL UNIQUE,created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS cleanup_file_cache(path TEXT PRIMARY KEY,size_bytes INTEGER,modified_at INTEGER,sha256 TEXT,width INTEGER,height INTEGER,ahash TEXT,dhash TEXT,phash TEXT,updated_at INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS idx_cleanup_file_cache_fingerprint ON cleanup_file_cache(size_bytes,modified_at);
"#).map_err(|e|format!("failed to initialize database: {e}"))?;
    for col in [
        "lens_model TEXT",
        "captured_at INTEGER",
        "metadata_json TEXT",
    ] {
        let _ = conn.execute(&format!("ALTER TABLE media_items ADD COLUMN {col}"), []);
    }
    let _ = conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_media_items_captured ON media_items(captured_at)",
        [],
    );
    let _ = conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_media_items_sort_date ON media_items(COALESCE(captured_at,modified_at,created_at), id)",
        [],
    );
    let _ = conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_embeddings_model_media ON embeddings(model, media_item_id)",
        [],
    );
    Ok(())
}
fn unix_time(t: SystemTime) -> Option<i64> {
    t.duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs() as i64)
}
fn now_unix() -> i64 {
    unix_time(SystemTime::now()).unwrap_or(0)
}
fn clean_path_str(s: &str) -> String {
    s.strip_prefix("\\\\?\\").unwrap_or(s).to_string()
}
fn clean_path_string(path: &Path) -> String {
    clean_path_str(&path.to_string_lossy())
}
#[derive(Debug, Clone, Copy)]
struct MediaFingerprint {
    size_bytes: Option<i64>,
    modified_at: Option<i64>,
}
#[derive(Debug)]
enum MediaIndexResult {
    Upsert(MediaItem),
    Unchanged,
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
fn is_heic_path(path: &str) -> bool {
    matches!(
        Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.trim_start_matches('.').to_ascii_lowercase())
            .as_deref(),
        Some("heic" | "heif")
    )
}
fn parse_exif(
    path: &Path,
) -> (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<i64>,
    Option<f64>,
    Option<f64>,
    Option<String>,
) {
    let Some(file) = fs::File::open(path).ok() else {
        return (None, None, None, None, None, None, None);
    };
    let mut br = std::io::BufReader::new(file);
    let Some(exif) = exif::Reader::new().read_from_container(&mut br).ok() else {
        return (None, None, None, None, None, None, None);
    };
    use exif::{In, Tag, Value};
    let get = |tag| {
        exif.get_field(tag, In::PRIMARY)
            .map(|f| f.display_value().with_unit(&exif).to_string())
    };
    let make = get(Tag::Make);
    let model = get(Tag::Model);
    let lens = get(Tag::LensModel);
    let dt = get(Tag::DateTimeOriginal)
        .and_then(|s| {
            chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S")
                .or_else(|_| chrono::NaiveDateTime::parse_from_str(&s, "%Y:%m:%d %H:%M:%S"))
                .ok()
        })
        .map(|d| d.and_utc().timestamp());
    let gps = |tag| {
        exif.get_field(tag, In::PRIMARY).and_then(|f| {
            if let Value::Rational(v) = &f.value {
                if v.len() >= 3 {
                    Some(v[0].to_f64() + v[1].to_f64() / 60.0 + v[2].to_f64() / 3600.0)
                } else {
                    None
                }
            } else {
                None
            }
        })
    };
    let mut lat = gps(Tag::GPSLatitude);
    let mut lng = gps(Tag::GPSLongitude);
    if get(Tag::GPSLatitudeRef).unwrap_or_default().contains('S') {
        lat = lat.map(|v| -v)
    };
    if get(Tag::GPSLongitudeRef).unwrap_or_default().contains('W') {
        lng = lng.map(|v| -v)
    };
    let json = serde_json::to_string(
        &serde_json::json!({"make":make,"model":model,"lens_model":lens,"captured_at":dt,"latitude":lat,"longitude":lng}),
    )
    .ok();
    (make, model, lens, dt, lat, lng, json)
}
fn media_from_path_with_existing(
    path: &Path,
    existing: Option<&HashMap<String, MediaFingerprint>>,
) -> Result<Option<MediaIndexResult>, String> {
    if is_excluded_path(path) {
        return Ok(None);
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase());
    let Some(mt) = media_type_for_ext(ext.as_deref()) else {
        return Ok(None);
    };
    if is_facetile_image_path(path) {
        return Ok(None);
    }
    let md = fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !md.is_file() {
        return Ok(None);
    };
    let modified_at = md.modified().ok().and_then(unix_time);
    let size_bytes = Some(md.len() as i64);
    let path_string = clean_path_string(path);
    if existing
        .and_then(|known| known.get(&path_string))
        .is_some_and(|known| known.size_bytes == size_bytes && known.modified_at == modified_at)
    {
        return Ok(Some(MediaIndexResult::Unchanged));
    }
    let (mk, mo, lens, cap, lat, lng, mjson) = if mt == "image" {
        parse_exif(path)
    } else {
        (None, None, None, None, None, None, None)
    };
    Ok(Some(MediaIndexResult::Upsert(MediaItem {
        id: 0,
        path: path_string,
        display_path: None,
        file_name: path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string(),
        extension: ext,
        media_type: mt.to_string(),
        size_bytes,
        created_at: md.created().ok().and_then(unix_time),
        modified_at,
        imported_at: now_unix(),
        missing: false,
        camera_make: mk,
        camera_model: mo,
        lens_model: lens,
        captured_at: cap,
        latitude: lat,
        longitude: lng,
        metadata_json: mjson,
    })))
}
fn upsert_media(conn: &Connection, item: &MediaItem) -> Result<(), String> {
    conn.execute(r#"INSERT INTO media_items(path,file_name,extension,media_type,size_bytes,created_at,modified_at,imported_at,missing,camera_make,camera_model,lens_model,captured_at,latitude,longitude,metadata_json) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,0,?9,?10,?11,?12,?13,?14,?15) ON CONFLICT(path) DO UPDATE SET file_name=excluded.file_name,extension=excluded.extension,media_type=excluded.media_type,size_bytes=excluded.size_bytes,created_at=excluded.created_at,modified_at=excluded.modified_at,missing=0,camera_make=excluded.camera_make,camera_model=excluded.camera_model,lens_model=excluded.lens_model,captured_at=excluded.captured_at,latitude=excluded.latitude,longitude=excluded.longitude,metadata_json=excluded.metadata_json"#,params![item.path,item.file_name,item.extension,item.media_type,item.size_bytes,item.created_at,item.modified_at,item.imported_at,item.camera_make,item.camera_model,item.lens_model,item.captured_at,item.latitude,item.longitude,item.metadata_json]).map_err(|e|format!("failed to upsert media item: {e}"))?;
    Ok(())
}
fn load_existing_media_fingerprints(
    conn: &Connection,
) -> Result<HashMap<String, MediaFingerprint>, String> {
    let mut stmt = conn
        .prepare("SELECT path,size_bytes,modified_at FROM media_items WHERE missing=0")
        .map_err(|e| format!("failed to prepare existing media lookup: {e}"))?;
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                MediaFingerprint {
                    size_bytes: row.get(1)?,
                    modified_at: row.get(2)?,
                },
            ))
        })
        .map_err(|e| format!("failed to query existing media lookup: {e}"))?;
    let mut out = HashMap::new();
    for row in rows {
        let (path, fingerprint) = row.map_err(|e| e.to_string())?;
        out.insert(path, fingerprint);
    }
    Ok(out)
}
const MEDIA_SELECT:&str="SELECT id,path,file_name,extension,media_type,size_bytes,created_at,modified_at,imported_at,missing,camera_make,camera_model,lens_model,captured_at,latitude,longitude,metadata_json FROM media_items";
fn row_to_media(row: &rusqlite::Row<'_>) -> rusqlite::Result<MediaItem> {
    Ok(MediaItem {
        id: row.get(0)?,
        path: row.get(1)?,
        display_path: None,
        file_name: row.get(2)?,
        extension: row.get(3)?,
        media_type: row.get(4)?,
        size_bytes: row.get(5)?,
        created_at: row.get(6)?,
        modified_at: row.get(7)?,
        imported_at: row.get(8)?,
        missing: row.get::<_, i64>(9)? != 0,
        camera_make: row.get(10)?,
        camera_model: row.get(11)?,
        lens_model: row.get(12)?,
        captured_at: row.get(13)?,
        latitude: row.get(14)?,
        longitude: row.get(15)?,
        metadata_json: row.get(16)?,
    })
}

fn converted_heic_paths(
    app: &tauri::AppHandle,
    paths: &[String],
) -> Result<HashMap<String, String>, String> {
    let heic_paths: Vec<String> = paths.iter().filter(|p| is_heic_path(p)).cloned().collect();
    if heic_paths.is_empty() {
        return Ok(HashMap::new());
    }
    let cache_dir = heic_conversions_dir(app)?;
    fs::create_dir_all(&cache_dir).map_err(|e| {
        format!(
            "failed to create HEIC conversion cache {}: {e}",
            cache_dir.display()
        )
    })?;
    let payload = serde_json::json!({"paths": heic_paths, "cache_dir": cache_dir});
    let res = run_sidecar_json_payload(vec!["convert-heic".into()], &payload)?;
    if !res.ok {
        return Err(if res.stderr.trim().is_empty() {
            "HEIC conversion failed".to_string()
        } else {
            res.stderr.trim().to_string()
        });
    }
    let root: serde_json::Value = serde_json::from_str(&res.stdout)
        .map_err(|e| format!("invalid HEIC conversion JSON: {e}"))?;
    let mut out = HashMap::new();
    if let Some(items) = root.pointer("/data/conversions").and_then(|v| v.as_array()) {
        for item in items {
            let Some(source) = item.get("source").and_then(|v| v.as_str()) else {
                continue;
            };
            let Some(converted) = item.get("converted").and_then(|v| v.as_str()) else {
                continue;
            };
            out.insert(clean_path_str(source), clean_path_str(converted));
        }
    }
    Ok(out)
}

fn enrich_display_paths(app: &tauri::AppHandle, items: &mut [MediaItem]) -> Result<(), String> {
    let paths: Vec<String> = items
        .iter()
        .filter(|item| !item.missing)
        .map(|item| item.path.clone())
        .collect();
    let conversions = converted_heic_paths(app, &paths)?;
    for item in items {
        item.display_path = conversions.get(&item.path).cloned();
    }
    Ok(())
}

fn sidecar_media_rows(
    app: &tauri::AppHandle,
    rows: Vec<(i64, String)>,
) -> Result<Vec<(i64, String)>, String> {
    let paths: Vec<String> = rows.iter().map(|(_, path)| path.clone()).collect();
    let conversions = converted_heic_paths(app, &paths)?;
    Ok(rows
        .into_iter()
        .map(|(id, path)| {
            let sidecar_path = conversions.get(&path).cloned().unwrap_or(path);
            (id, sidecar_path)
        })
        .collect())
}
fn haversine_km(lat1: f64, lng1: f64, lat2: f64, lng2: f64) -> f64 {
    let earth_radius_km = 6371.0088_f64;
    let dlat = (lat2 - lat1).to_radians();
    let dlng = (lng2 - lng1).to_radians();
    let lat1 = lat1.to_radians();
    let lat2 = lat2.to_radians();
    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlng / 2.0).sin().powi(2);
    2.0 * earth_radius_km * a.sqrt().asin()
}
fn radius_bounds(lat: f64, lng: f64, radius_km: f64) -> (f64, f64, Option<(f64, f64)>) {
    let lat_delta = radius_km / 111.32;
    let min_lat = (lat - lat_delta).max(-90.0);
    let max_lat = (lat + lat_delta).min(90.0);
    let lng_bounds = if min_lat <= -90.0 || max_lat >= 90.0 {
        None
    } else {
        let cos_lat = lat.to_radians().cos().abs().max(0.01);
        let lng_delta = (radius_km / (111.32 * cos_lat)).min(180.0);
        Some(((lng - lng_delta).max(-180.0), (lng + lng_delta).min(180.0)))
    };
    (min_lat, max_lat, lng_bounds)
}
#[tauri::command]
fn initialize_app(app: tauri::AppHandle) -> Result<AppInfo, String> {
    let db = db_path(&app)?;
    let index_exists = db.exists();
    let c = open_db(&app)?;
    drop(c);
    Ok(AppInfo {
        data_dir: app_data_dir(&app)?.to_string_lossy().to_string(),
        database_path: db.to_string_lossy().to_string(),
        index_exists,
    })
}
#[tauri::command]
fn delete_current_index(app: tauri::AppHandle) -> Result<AppInfo, String> {
    let path = db_path(&app)?;
    if path.exists() {
        fs::remove_file(&path)
            .map_err(|e| format!("failed to delete database index {}: {e}", path.display()))?;
    }
    let conversions = heic_conversions_dir(&app)?;
    if conversions.exists() {
        fs::remove_dir_all(&conversions).map_err(|e| {
            format!(
                "failed to delete HEIC conversion cache {}: {e}",
                conversions.display()
            )
        })?;
    }
    Ok(AppInfo {
        data_dir: app_data_dir(&app)?.to_string_lossy().to_string(),
        database_path: path.to_string_lossy().to_string(),
        index_exists: false,
    })
}
#[tauri::command]
fn get_settings(app: tauri::AppHandle) -> Result<AppSettings, String> {
    Ok(AppSettings {
        library_folders: list_library_folders(app)?,
    })
}
#[tauri::command]
fn update_settings(app: tauri::AppHandle, settings: AppSettings) -> Result<AppSettings, String> {
    let c = open_db(&app)?;
    c.execute("DELETE FROM library_folders", [])
        .map_err(|e| e.to_string())?;
    for p in &settings.library_folders {
        c.execute(
            "INSERT OR IGNORE INTO library_folders(path,created_at) VALUES(?1,?2)",
            params![p, now_unix()],
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(settings)
}
#[tauri::command]
fn list_library_folders(app: tauri::AppHandle) -> Result<Vec<String>, String> {
    let c = open_db(&app)?;
    let mut s = c
        .prepare("SELECT path FROM library_folders ORDER BY path")
        .map_err(|e| e.to_string())?;
    let res = s
        .query_map([], |r| r.get::<_, String>(0))
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    Ok(res.into_iter().map(|p| clean_path_str(&p)).collect())
}
#[tauri::command]
fn add_library_folder(app: tauri::AppHandle, path: String) -> Result<Vec<String>, String> {
    let c = open_db(&app)?;
    let root = Path::new(&path);
    if !root.is_dir() {
        return Err("not a directory".into());
    }
    let saved = clean_path_string(&root.canonicalize().unwrap_or_else(|_| root.to_path_buf()));
    c.execute(
        "INSERT OR IGNORE INTO library_folders(path,created_at) VALUES(?1,?2)",
        params![saved, now_unix()],
    )
    .map_err(|e| e.to_string())?;
    drop(c);
    list_library_folders(app)
}
#[tauri::command]
fn remove_library_folder(app: tauri::AppHandle, path: String) -> Result<Vec<String>, String> {
    let c = open_db(&app)?;
    c.execute("DELETE FROM library_folders WHERE path=?1", params![path])
        .map_err(|e| e.to_string())?;
    drop(c);
    list_library_folders(app)
}
#[tauri::command]
async fn choose_media_folder(
    app: tauri::AppHandle,
    path: Option<String>,
) -> Result<Option<String>, String> {
    if let Some(p) = path {
        return Ok(Path::new(&p).is_dir().then_some(clean_path_str(&p)));
    };
    Ok(app
        .dialog()
        .file()
        .blocking_pick_folder()
        .map(|p| clean_path_str(&p.to_string())))
}
fn discover_files(
    root: PathBuf,
    threads: usize,
    max_files: Option<usize>,
) -> (Vec<PathBuf>, Vec<String>) {
    backend_log(&format!(
        "starting discovery root={} threads={threads} max_files={max_files:?}",
        root.display()
    ));
    let queue = Arc::new(Mutex::new(VecDeque::from([root.clone()])));
    let files = Arc::new(Mutex::new(Vec::new()));
    let errors = Arc::new(Mutex::new(Vec::new()));
    let active = Arc::new(AtomicUsize::new(0));
    let found = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::with_capacity(threads);
    for _ in 0..threads {
        let queue = Arc::clone(&queue);
        let files = Arc::clone(&files);
        let errors = Arc::clone(&errors);
        let active = Arc::clone(&active);
        let found = Arc::clone(&found);
        let root = root.clone();
        handles.push(thread::spawn(move || loop {
            if max_files.is_some_and(|limit| found.load(Ordering::Relaxed) >= limit) {
                break;
            }
            let dir = queue.lock().unwrap().pop_front();
            let Some(dir) = dir else {
                if active.load(Ordering::SeqCst) == 0 {
                    break;
                }
                thread::sleep(Duration::from_millis(2));
                continue;
            };
            active.fetch_add(1, Ordering::SeqCst);
            let read_dir = match fs::read_dir(&dir) {
                Ok(rd) => rd,
                Err(e) => {
                    let msg = format!(
                        "walk error reading {} under {}: {e}",
                        dir.display(),
                        root.display()
                    );
                    backend_log(&msg);
                    errors.lock().unwrap().push(msg);
                    active.fetch_sub(1, Ordering::SeqCst);
                    continue;
                }
            };
            let mut discovered_dirs = Vec::new();
            let mut discovered_files = Vec::new();
            for entry in read_dir {
                let entry = match entry {
                    Ok(e) => e,
                    Err(e) => {
                        let msg = format!("walk error under {}: {e}", root.display());
                        backend_log(&msg);
                        errors.lock().unwrap().push(msg);
                        continue;
                    }
                };
                let path = entry.path();
                let Ok(ft) = entry.file_type() else { continue };
                if is_excluded_path(&path) {
                    continue;
                }
                if ft.is_dir() {
                    discovered_dirs.push(path);
                    if discovered_dirs.len() >= 64 {
                        queue.lock().unwrap().extend(discovered_dirs.drain(..));
                    }
                } else if ft.is_file() {
                    let ext = path.extension().and_then(|e| e.to_str());
                    if media_type_for_ext(ext).is_none() {
                        continue;
                    }
                    if is_facetile_image_path(&path) {
                        continue;
                    }
                    let n = found.fetch_add(1, Ordering::Relaxed);
                    if max_files.is_some_and(|limit| n >= limit) {
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
        if h.join().is_err() {
            let msg = format!(
                "discovery worker thread panicked for root {}",
                root.display()
            );
            backend_log(&msg);
            errors.lock().unwrap().push(msg);
        }
    }
    let files = match Arc::try_unwrap(files) {
        Ok(m) => m.into_inner().unwrap_or_default(),
        Err(a) => a.lock().map(|v| v.clone()).unwrap_or_default(),
    };
    let errors = match Arc::try_unwrap(errors) {
        Ok(m) => m.into_inner().unwrap_or_default(),
        Err(a) => a
            .lock()
            .map(|v| v.clone())
            .unwrap_or_else(|_| vec!["discovery error mutex poisoned".into()]),
    };
    backend_log(&format!(
        "finished discovery root={} files={} errors={}",
        root.display(),
        files.len(),
        errors.len()
    ));
    (files, errors)
}
fn emit_scan_progress(
    app: &tauri::AppHandle,
    sum: &ScanSummary,
    phase: &str,
    current_path: Option<String>,
    discovered_files: usize,
    total_files: Option<usize>,
    faces_done: usize,
    faces_total: Option<usize>,
    done: bool,
) {
    let _ = app.emit(
        "scan-progress",
        ScanProgress {
            phase: phase.to_string(),
            current_path,
            scanned_files: sum.scanned_files,
            imported_or_updated: sum.imported_or_updated,
            skipped_files: sum.skipped_files,
            missing_marked: sum.missing_marked,
            errors: sum.errors.len(),
            discovered_files,
            total_files,
            faces_done,
            faces_total,
            done,
        },
    );
}
fn list_image_media_ids(conn: &Connection) -> Result<Vec<i64>, String> {
    let sql = format!(
        "SELECT id FROM media_items WHERE missing=0 AND media_type='image'{} ORDER BY id",
        sql_path_not_blacklisted_clause()
    );
    let mut s = conn.prepare(&sql).map_err(|e| e.to_string())?;
    let rows: Vec<i64> = s
        .query_map([], |r| r.get(0))
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    Ok(rows)
}
const FACE_INDEX_BATCH: usize = 6;
fn run_face_embedding_index_phase(
    app: &tauri::AppHandle,
    sum: &mut ScanSummary,
    discovered_files: usize,
    total_files: Option<usize>,
    media_ids: Option<Vec<i64>>,
) -> Result<(), String> {
    let conn = open_db(app)?;
    let media_ids = if let Some(ids) = media_ids {
        media_paths_for_ids(&conn, Some(ids), true)?
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    } else {
        list_image_media_ids(&conn)?
    };
    let total_face = media_ids.len();
    let face_threads = face_indexing_thread_count();
    if total_face == 0 {
        emit_scan_progress(
            app,
            sum,
            "Face embeddings (no indexed images)",
            None,
            discovered_files,
            total_files,
            0,
            Some(0),
            false,
        );
        return Ok(());
    }
    let progress_base = sum.clone();
    emit_scan_progress(
        app,
        &progress_base,
        &format!("Face embeddings — {face_threads} parallel workers"),
        None,
        discovered_files,
        total_files,
        0,
        Some(total_face),
        false,
    );
    let batches: Vec<Vec<i64>> = media_ids
        .chunks(FACE_INDEX_BATCH)
        .map(|c| c.to_vec())
        .collect();
    let done = AtomicUsize::new(0);
    let errors: Mutex<Vec<String>> = Mutex::new(vec![]);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(face_threads)
        .build()
        .map_err(|e| format!("face thread pool: {e}"))?;
    let app_c = app.clone();
    pool.install(|| {
        batches.par_iter().for_each(|chunk| {
            match process_face_paths(
                app_c.clone(),
                Some(chunk.clone()),
                false,
                true,
                face_threads,
            ) {
                Ok(r) if r.ok => {
                    let now = done.fetch_add(chunk.len(), Ordering::Relaxed) + chunk.len();
                    emit_scan_progress(
                        &app_c,
                        &progress_base,
                        &format!("Face embeddings ({face_threads} workers)"),
                        None,
                        discovered_files,
                        total_files,
                        now,
                        Some(total_face),
                        false,
                    );
                }
                Ok(r) => {
                    let msg = format!(
                        "face batch ids {:?}: sidecar failed: {}",
                        chunk,
                        r.stderr.trim()
                    );
                    backend_log(&msg);
                    if let Ok(mut g) = errors.lock() {
                        g.push(msg);
                    }
                }
                Err(e) => {
                    let msg = format!("face batch ids {:?}: {e}", chunk);
                    backend_log(&msg);
                    if let Ok(mut g) = errors.lock() {
                        g.push(msg);
                    }
                }
            }
        });
    });
    let mut batch_errors = errors
        .into_inner()
        .map_err(|_| "face errors mutex poisoned".to_string())?;
    sum.errors.append(&mut batch_errors);
    emit_scan_progress(
        app,
        &sum.clone(),
        "Face embeddings complete",
        None,
        discovered_files,
        total_files,
        total_face,
        Some(total_face),
        false,
    );
    Ok(())
}
fn scan_library_impl(app: tauri::AppHandle, paths: Vec<String>) -> Result<ScanSummary, String> {
    const PROGRESS_FILES_PER_INDEXING_THREAD: usize = 20;

    let mut conn = open_db(&app)?;
    let existing_fingerprints = Arc::new(load_existing_media_fingerprints(&conn)?);
    let mut sum = ScanSummary {
        scanned_files: 0,
        imported_or_updated: 0,
        skipped_files: 0,
        missing_marked: 0,
        errors: vec![],
    };
    let mut seen = HashSet::new();
    let mut discovered_files = 0usize;
    let mut total_files = None;
    emit_scan_progress(
        &app,
        &sum,
        "Starting scan",
        None,
        discovered_files,
        total_files,
        0,
        None,
        false,
    );
    let discovery_threads = discovery_thread_count();
    let indexing_threads = indexing_thread_count();
    let progress_emit_interval =
        (PROGRESS_FILES_PER_INDEXING_THREAD * indexing_threads.max(1)).max(1);
    backend_log(&format!(
        "scan started roots={} cpu_threads={} discovery_threads={} indexing_threads={}",
        paths.len(),
        available_threads(),
        discovery_threads,
        indexing_threads
    ));
    for root in paths {
        emit_scan_progress(
            &app,
            &sum,
            &format!("Discovering files ({discovery_threads} discovery threads; {indexing_threads} indexing threads)"),
            Some(root.clone()),
            discovered_files,
            total_files,
            0,
            None,
            false,
        );
        backend_log(&format!("scan root requested: {root}"));
        let root_path = Path::new(&root);
        if !root_path.exists() {
            let msg = format!("scan root does not exist: {root}");
            backend_log(&msg);
            sum.errors.push(msg);
            emit_scan_progress(
                &app,
                &sum,
                "Folder missing",
                Some(root),
                discovered_files,
                total_files,
                0,
                None,
                false,
            );
            continue;
        }
        if !root_path.is_dir() {
            let msg = format!("scan root is not a directory: {root}");
            backend_log(&msg);
            sum.errors.push(msg);
            emit_scan_progress(
                &app,
                &sum,
                "Not a directory",
                Some(root),
                discovered_files,
                total_files,
                0,
                None,
                false,
            );
            continue;
        }
        let (files, mut errors) = discover_files(root_path.to_path_buf(), discovery_threads, None);
        sum.errors.append(&mut errors);
        let discovered_count = files.len();
        discovered_files += discovered_count;
        let files: Vec<PathBuf> = files
            .into_iter()
            .filter(|p| seen.insert(p.clone()))
            .collect();
        total_files = Some(total_files.unwrap_or(0) + files.len());
        backend_log(&format!(
            "root {root}: discovered_files={discovered_count} unique_files={} total_errors={}",
            files.len(),
            sum.errors.len()
        ));
        emit_scan_progress(
            &app,
            &sum,
            &format!("Indexing metadata ({indexing_threads} indexing threads; discovery used {discovery_threads})"),
            files.first().map(|p| clean_path_string(p)),
            discovered_files,
            total_files,
            0,
            None,
            false,
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(indexing_threads)
            .build()
            .map_err(|e| {
                let msg = format!("failed to build indexing thread pool: {e}");
                backend_log(&msg);
                msg
            })?;
        let (tx, rx) = mpsc::channel();
        let mut pending_progress_path = files.first().cloned();
        let existing_fingerprints_for_indexing = Arc::clone(&existing_fingerprints);
        let indexing_handle = thread::spawn(move || {
            pool.install(|| {
                files.par_iter().for_each_with(tx, |tx, p| {
                    let _ = tx.send((
                        p.clone(),
                        media_from_path_with_existing(
                            p,
                            Some(existing_fingerprints_for_indexing.as_ref()),
                        ),
                    ));
                })
            })
        });
        let mut results = 0usize;
        let tx_db = conn
            .transaction()
            .map_err(|e| format!("failed to begin index transaction: {e}"))?;
        for (scan_path, res) in rx {
            results += 1;
            sum.scanned_files += 1;
            pending_progress_path = Some(scan_path.clone());
            match res.and_then(|o| match o {
                Some(MediaIndexResult::Upsert(i)) => upsert_media(&tx_db, &i).map(|_| true),
                Some(MediaIndexResult::Unchanged) | None => Ok(false),
            }) {
                Ok(true) => sum.imported_or_updated += 1,
                Ok(false) => sum.skipped_files += 1,
                Err(e) => {
                    backend_log(&format!("index error for {}: {e}", scan_path.display()));
                    sum.errors.push(e)
                }
            }
            if results % progress_emit_interval == 0 {
                emit_scan_progress(
                    &app,
                    &sum,
                    "Indexing metadata",
                    pending_progress_path.as_ref().map(|p| clean_path_string(p)),
                    discovered_files,
                    total_files,
                    0,
                    None,
                    false,
                );
            }
        }
        if results > 0 && results % progress_emit_interval != 0 {
            emit_scan_progress(
                &app,
                &sum,
                "Indexing metadata",
                pending_progress_path.as_ref().map(|p| clean_path_string(p)),
                discovered_files,
                total_files,
                0,
                None,
                false,
            );
        }
        tx_db
            .commit()
            .map_err(|e| format!("failed to commit index transaction: {e}"))?;
        if indexing_handle.join().is_err() {
            let msg = format!("indexing worker thread panicked for root {root}");
            backend_log(&msg);
            sum.errors.push(msg);
        }
        backend_log(&format!(
            "root {root}: metadata indexing complete results={results}"
        ));
    }
    emit_scan_progress(
        &app,
        &sum,
        "Checking missing files",
        None,
        discovered_files,
        total_files,
        0,
        None,
        false,
    );
    backend_log("checking missing files");
    sum.missing_marked = mark_missing_internal(&conn)?;
    backend_log(&format!(
        "scan complete scanned={} imported_or_updated={} skipped={} missing_marked={} errors={}",
        sum.scanned_files,
        sum.imported_or_updated,
        sum.skipped_files,
        sum.missing_marked,
        sum.errors.len()
    ));
    emit_scan_progress(
        &app,
        &sum,
        "Scan complete",
        None,
        discovered_files,
        total_files,
        0,
        None,
        true,
    );
    Ok(sum)
}
#[tauri::command]
async fn scan_library(app: tauri::AppHandle, paths: Vec<String>) -> Result<ScanSummary, String> {
    tauri::async_runtime::spawn_blocking(move || scan_library_impl(app, paths))
        .await
        .map_err(|e| {
            let msg = format!("scan task failed/thread panicked: {e}");
            backend_log(&msg);
            msg
        })?
}
#[tauri::command]
async fn update_face_embeddings(app: tauri::AppHandle) -> Result<ScanSummary, String> {
    match tauri::async_runtime::spawn_blocking(move || {
        let mut sum = ScanSummary {
            scanned_files: 0,
            imported_or_updated: 0,
            skipped_files: 0,
            missing_marked: 0,
            errors: vec![],
        };
        let discovered_files = 0usize;
        let total_files: Option<usize> = None;
        emit_scan_progress(
            &app,
            &sum,
            "Updating face embeddings",
            None,
            discovered_files,
            total_files,
            0,
            None,
            false,
        );
        run_face_embedding_index_phase(&app, &mut sum, discovered_files, total_files, None)?;
        emit_scan_progress(
            &app,
            &sum,
            "Face embedding update finished",
            None,
            discovered_files,
            total_files,
            0,
            None,
            true,
        );
        Ok(sum)
    })
    .await
    {
        Ok(Ok(s)) => Ok(s),
        Ok(Err(e)) => Err(e),
        Err(e) => Err(format!("face update task failed: {e}")),
    }
}
fn mark_missing_internal(conn: &Connection) -> Result<usize, String> {
    let mut s = conn
        .prepare("SELECT id,path FROM media_items WHERE missing=0")
        .map_err(|e| e.to_string())?;
    let rows = s
        .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))
        .map_err(|e| e.to_string())?;
    let mut n = 0;
    for r in rows {
        let (id, p) = r.map_err(|e| e.to_string())?;
        if !Path::new(&p).exists() {
            conn.execute("UPDATE media_items SET missing=1 WHERE id=?1", params![id])
                .map_err(|e| e.to_string())?;
            n += 1
        }
    }
    Ok(n)
}
#[tauri::command]
fn mark_missing(app: tauri::AppHandle) -> Result<usize, String> {
    mark_missing_internal(&open_db(&app)?)
}

fn cleanup_plans() -> &'static Mutex<HashMap<String, CleanupPlan>> {
    CLEANUP_PLANS.get_or_init(|| Mutex::new(HashMap::new()))
}
fn emit_cleanup_progress(
    app: &tauri::AppHandle,
    phase: &str,
    current_path: Option<String>,
    processed: usize,
    total: Option<usize>,
    errors: usize,
    done: bool,
) {
    let _ = app.emit(
        "cleanup-progress",
        CleanupProgress {
            phase: phase.to_string(),
            current_path,
            processed,
            total,
            errors,
            done,
        },
    );
}
fn cleanup_plan_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("cleanup-{millis}")
}
fn file_sha256(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 128];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}
fn cleanup_date(c: &CleanupCandidate) -> i64 {
    c.captured_at
        .or(c.modified_at)
        .or(c.created_at)
        .unwrap_or(i64::MAX)
}
fn cleanup_pixels(c: &CleanupCandidate) -> i64 {
    c.width.unwrap_or(0).saturating_mul(c.height.unwrap_or(0))
}
fn choose_cleanup_keeper(candidates: &[CleanupCandidate]) -> Option<String> {
    candidates
        .iter()
        .min_by(|a, b| {
            cleanup_date(a)
                .cmp(&cleanup_date(b))
                .then_with(|| cleanup_pixels(b).cmp(&cleanup_pixels(a)))
                .then_with(|| b.size_bytes.unwrap_or(0).cmp(&a.size_bytes.unwrap_or(0)))
                .then_with(|| a.path.cmp(&b.path))
        })
        .map(|c| c.path.clone())
}
fn cleanup_candidate_from_media(item: &MediaItem) -> CleanupCandidate {
    CleanupCandidate {
        path: item.path.clone(),
        display_path: item.display_path.clone(),
        file_name: item.file_name.clone(),
        size_bytes: item.size_bytes,
        created_at: item.created_at,
        modified_at: item.modified_at,
        captured_at: item.captured_at,
        width: None,
        height: None,
    }
}
fn collect_ignored_cleanup_files(
    app: &tauri::AppHandle,
    root: &Path,
    out: &mut Vec<CleanupEntry>,
    errors: &mut Vec<String>,
    visited: &mut usize,
) {
    *visited += 1;
    if *visited % 500 == 0 {
        emit_cleanup_progress(
            app,
            "Finding ignored files",
            Some(clean_path_string(root)),
            *visited,
            None,
            errors.len(),
            false,
        );
    }
    let rd = match fs::read_dir(root) {
        Ok(rd) => rd,
        Err(e) => {
            errors.push(format!("failed to read {}: {e}", root.display()));
            return;
        }
    };
    for entry in rd {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        let Ok(ft) = entry.file_type() else { continue };
        if is_excluded_path(&path) {
            if ft.is_file() {
                out.push(CleanupEntry {
                    path: clean_path_string(&path),
                    size_bytes: fs::metadata(&path).ok().map(|m| m.len() as i64),
                    reason: "Ignored by indexing rules".into(),
                });
            } else if ft.is_dir() {
                for child in walkdir::WalkDir::new(&path)
                    .into_iter()
                    .filter_map(Result::ok)
                {
                    if child.file_type().is_file() {
                        let p = child.path();
                        out.push(CleanupEntry {
                            path: clean_path_string(p),
                            size_bytes: child.metadata().ok().map(|m| m.len() as i64),
                            reason: "Inside folder ignored by indexing rules".into(),
                        });
                    }
                }
            }
            continue;
        }
        if ft.is_dir() {
            collect_ignored_cleanup_files(app, &path, out, errors, visited);
        } else if ft.is_file() && is_facetile_image_path(&path) {
            out.push(CleanupEntry {
                path: clean_path_string(&path),
                size_bytes: fs::metadata(&path).ok().map(|m| m.len() as i64),
                reason: "Facetile image ignored by indexing rules".into(),
            });
        }
    }
}
fn collect_empty_folders(
    app: &tauri::AppHandle,
    root: &Path,
    out: &mut Vec<CleanupEntry>,
    errors: &mut Vec<String>,
) {
    let mut dirs: Vec<PathBuf> = walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_dir())
        .map(|e| e.path().to_path_buf())
        .collect();
    dirs.sort_by_key(|p| std::cmp::Reverse(p.components().count()));
    let total = dirs.len();
    for (i, dir) in dirs.into_iter().enumerate() {
        if i % 1000 == 0 {
            emit_cleanup_progress(
                app,
                "Finding empty folders",
                Some(clean_path_string(&dir)),
                i + 1,
                Some(total),
                errors.len(),
                false,
            );
        }
        if dir == root {
            continue;
        }
        match fs::read_dir(&dir) {
            Ok(mut rd) => {
                if rd.next().is_none() {
                    out.push(CleanupEntry {
                        path: clean_path_string(&dir),
                        size_bytes: None,
                        reason: "Empty folder".into(),
                    });
                }
            }
            Err(e) => errors.push(format!("failed to inspect {}: {e}", dir.display())),
        }
    }
}
fn cleanup_image_items(
    conn: &Connection,
    app: &tauri::AppHandle,
) -> Result<Vec<MediaItem>, String> {
    let sql = format!(
        "{MEDIA_SELECT} WHERE missing=0 AND media_type='image'{} ORDER BY id",
        sql_path_not_blacklisted_clause()
    );
    let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], row_to_media)
        .map_err(|e| e.to_string())?;
    let mut items = Vec::new();
    for row in rows {
        let item = row.map_err(|e| e.to_string())?;
        if Path::new(&item.path).is_file() {
            items.push(item);
        }
    }
    enrich_display_paths(app, &mut items)?;
    Ok(items)
}
fn path_is_under_roots(path: &str, roots: &[String]) -> bool {
    let path = Path::new(path);
    roots.iter().any(|root| path.starts_with(Path::new(root)))
}
fn hex_hamming(a: &str, b: &str) -> Option<i64> {
    let x = u128::from_str_radix(a, 16).ok()?;
    let y = u128::from_str_radix(b, 16).ok()?;
    Some((x ^ y).count_ones() as i64)
}

#[derive(Debug, Deserialize)]
struct VisualFingerprintItem {
    path: String,
    width: i64,
    height: i64,
    ahash: String,
    dhash: String,
    phash: String,
}
#[derive(Debug, Clone)]
struct CleanupFileCache {
    size_bytes: Option<i64>,
    modified_at: Option<i64>,
    sha256: Option<String>,
    width: Option<i64>,
    height: Option<i64>,
    ahash: Option<String>,
    dhash: Option<String>,
    phash: Option<String>,
}
#[derive(Debug, Clone)]
struct CleanupCacheUpdate {
    path: String,
    size_bytes: Option<i64>,
    modified_at: Option<i64>,
    sha256: Option<String>,
    width: Option<i64>,
    height: Option<i64>,
    ahash: Option<String>,
    dhash: Option<String>,
    phash: Option<String>,
}
#[derive(Debug, Serialize)]
struct CleanupCacheInfo {
    total_entries: i64,
    exact_hash_entries: i64,
    visual_fingerprint_entries: i64,
    complete_entries: i64,
    selected_images: usize,
    selected_exact_cached: usize,
    selected_visual_cached: usize,
    selected_complete_cached: usize,
    selected_uncached_exact: usize,
    selected_uncached_visual: usize,
    stale_selected_entries: usize,
    cache_bytes_estimate: i64,
    oldest_updated_at: Option<i64>,
    newest_updated_at: Option<i64>,
}
fn load_cleanup_file_cache(conn: &Connection) -> Result<HashMap<String, CleanupFileCache>, String> {
    let mut stmt = conn
        .prepare("SELECT path,size_bytes,modified_at,sha256,width,height,ahash,dhash,phash FROM cleanup_file_cache")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                CleanupFileCache {
                    size_bytes: row.get(1)?,
                    modified_at: row.get(2)?,
                    sha256: row.get(3)?,
                    width: row.get(4)?,
                    height: row.get(5)?,
                    ahash: row.get(6)?,
                    dhash: row.get(7)?,
                    phash: row.get(8)?,
                },
            ))
        })
        .map_err(|e| e.to_string())?;
    let mut out = HashMap::new();
    for row in rows {
        let (path, cache) = row.map_err(|e| e.to_string())?;
        out.insert(path, cache);
    }
    Ok(out)
}
fn cache_matches_media(cache: &CleanupFileCache, item: &MediaItem) -> bool {
    cache.size_bytes == item.size_bytes && cache.modified_at == item.modified_at
}
fn write_cleanup_cache_updates(
    conn: &mut Connection,
    updates: &[CleanupCacheUpdate],
) -> Result<(), String> {
    if updates.is_empty() {
        return Ok(());
    }
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    for update in updates {
        if update.sha256.is_some() && update.ahash.is_none() {
            tx.execute(
                r#"INSERT INTO cleanup_file_cache(path,size_bytes,modified_at,sha256,width,height,ahash,dhash,phash,updated_at)
                   VALUES(?1,?2,?3,?4,NULL,NULL,NULL,NULL,NULL,?5)
                   ON CONFLICT(path) DO UPDATE SET
                     size_bytes=excluded.size_bytes,
                     modified_at=excluded.modified_at,
                     sha256=excluded.sha256,
                     width=cleanup_file_cache.width,
                     height=cleanup_file_cache.height,
                     ahash=cleanup_file_cache.ahash,
                     dhash=cleanup_file_cache.dhash,
                     phash=cleanup_file_cache.phash,
                     updated_at=excluded.updated_at"#,
                params![
                    update.path,
                    update.size_bytes,
                    update.modified_at,
                    update.sha256,
                    now_unix()
                ],
            )
            .map_err(|e| e.to_string())?;
        } else if update.sha256.is_none() && update.ahash.is_some() {
            tx.execute(
                r#"INSERT INTO cleanup_file_cache(path,size_bytes,modified_at,sha256,width,height,ahash,dhash,phash,updated_at)
                   VALUES(?1,?2,?3,NULL,?4,?5,?6,?7,?8,?9)
                   ON CONFLICT(path) DO UPDATE SET
                     size_bytes=excluded.size_bytes,
                     modified_at=excluded.modified_at,
                     sha256=cleanup_file_cache.sha256,
                     width=excluded.width,
                     height=excluded.height,
                     ahash=excluded.ahash,
                     dhash=excluded.dhash,
                     phash=excluded.phash,
                     updated_at=excluded.updated_at"#,
                params![
                    update.path,
                    update.size_bytes,
                    update.modified_at,
                    update.width,
                    update.height,
                    update.ahash,
                    update.dhash,
                    update.phash,
                    now_unix()
                ],
            )
            .map_err(|e| e.to_string())?;
        }
    }
    tx.commit().map_err(|e| e.to_string())?;
    Ok(())
}

fn build_visual_duplicate_groups(
    app: &tauri::AppHandle,
    items: &[MediaItem],
    exact_paths: &HashSet<String>,
    cleanup_cache: &HashMap<String, CleanupFileCache>,
    errors: &mut Vec<String>,
) -> Result<(Vec<CleanupDuplicateGroup>, Vec<CleanupCacheUpdate>), String> {
    const VISUAL_FINGERPRINT_BATCH: usize = 256;
    let eligible: Vec<MediaItem> = items
        .iter()
        .filter(|i| !exact_paths.contains(&i.path))
        .cloned()
        .collect();
    let rows: Vec<(i64, String)> = eligible.iter().map(|i| (i.id, i.path.clone())).collect();
    let sidecar_rows = sidecar_media_rows(app, rows)?;
    let sidecar_to_original: HashMap<String, String> = eligible
        .iter()
        .zip(sidecar_rows.iter())
        .map(|(item, (_, p))| (p.clone(), item.path.clone()))
        .collect();
    let paths: Vec<String> = sidecar_rows.into_iter().map(|(_, p)| p).collect();
    if paths.is_empty() {
        return Ok((vec![], vec![]));
    }
    let mut fps = Vec::new();
    let mut cache_updates = Vec::new();
    let mut paths_to_fingerprint = Vec::new();
    for path in &paths {
        let Some(original_path) = sidecar_to_original.get(path) else {
            continue;
        };
        let Some(item) = eligible.iter().find(|item| item.path == *original_path) else {
            continue;
        };
        if let Some(cache) = cleanup_cache.get(original_path).filter(|cache| {
            cache_matches_media(cache, item)
                && cache.width.is_some()
                && cache.height.is_some()
                && cache.ahash.is_some()
                && cache.dhash.is_some()
                && cache.phash.is_some()
        }) {
            fps.push(VisualFingerprintItem {
                path: path.clone(),
                width: cache.width.unwrap_or(0),
                height: cache.height.unwrap_or(0),
                ahash: cache.ahash.clone().unwrap_or_default(),
                dhash: cache.dhash.clone().unwrap_or_default(),
                phash: cache.phash.clone().unwrap_or_default(),
            });
        } else {
            paths_to_fingerprint.push(path.clone());
        }
    }
    if !paths_to_fingerprint.is_empty() {
        emit_cleanup_progress(
            app,
            "Using cached visual fingerprints",
            None,
            fps.len(),
            Some(paths.len()),
            errors.len(),
            false,
        );
    }
    let chunks: Vec<Vec<String>> = paths_to_fingerprint
        .chunks(VISUAL_FINGERPRINT_BATCH)
        .map(|chunk| chunk.to_vec())
        .collect();
    let visual_threads = cleanup_visual_thread_count().min(chunks.len().max(1));
    if !chunks.is_empty() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(visual_threads)
            .build()
            .map_err(|e| format!("cleanup visual thread pool: {e}"))?;
        let (tx, rx) = mpsc::channel();
        let batch_count = chunks.len();
        let chunks_for_worker = chunks.clone();
        let visual_handle = thread::spawn(move || {
            pool.install(|| {
                chunks_for_worker.par_iter().enumerate().for_each_with(
                    tx,
                    |tx, (batch_index, chunk)| {
                        let payload = serde_json::json!({"paths":chunk});
                        let res =
                            run_sidecar_json_payload(vec!["visual-fingerprints".into()], &payload);
                        let _ = tx.send((batch_index, chunk.first().cloned(), res));
                    },
                );
            });
        });
        let mut batch_results = Vec::with_capacity(batch_count);
        for (done, result) in rx.into_iter().enumerate() {
            let (batch_index, first_path, res) = result;
            emit_cleanup_progress(
                app,
                &format!("Fingerprinting visual duplicates ({visual_threads} workers)"),
                first_path,
                fps.len() + ((done + 1) * VISUAL_FINGERPRINT_BATCH).min(paths_to_fingerprint.len()),
                Some(paths.len()),
                errors.len(),
                false,
            );
            batch_results.push((batch_index, res));
        }
        if visual_handle.join().is_err() {
            errors.push("visual fingerprint worker thread panicked".into());
        }
        batch_results.sort_by_key(|(batch_index, _)| *batch_index);
        let mut fps = Vec::new();
        for (_, res) in batch_results {
            let res = res?;
            if !res.ok {
                errors.push(res.stderr);
                continue;
            }
            let root: serde_json::Value = serde_json::from_str(&res.stdout)
                .map_err(|e| format!("invalid sidecar JSON: {e}"))?;
            if let Some(sidecar_errors) = root.pointer("/data/errors").and_then(|v| v.as_array()) {
                for err in sidecar_errors {
                    errors.push(format!("visual fingerprint failed: {err}"));
                }
            }
            let batch_fps: Vec<VisualFingerprintItem> = serde_json::from_value(
                root.pointer("/data/fingerprints")
                    .cloned()
                    .unwrap_or_else(|| serde_json::Value::Array(vec![])),
            )
            .map_err(|e| e.to_string())?;
            for fp in batch_fps {
                if let Some(original_path) = sidecar_to_original.get(&fp.path) {
                    if let Some(item) = eligible.iter().find(|item| item.path == *original_path) {
                        cache_updates.push(CleanupCacheUpdate {
                            path: item.path.clone(),
                            size_bytes: item.size_bytes,
                            modified_at: item.modified_at,
                            sha256: None,
                            width: Some(fp.width),
                            height: Some(fp.height),
                            ahash: Some(fp.ahash.clone()),
                            dhash: Some(fp.dhash.clone()),
                            phash: Some(fp.phash.clone()),
                        });
                    }
                }
                fps.push(fp);
            }
        }
    }
    let by_path: HashMap<String, MediaItem> = eligible
        .into_iter()
        .map(|item| (item.path.clone(), item))
        .collect();
    let mut keyed: Vec<(String, VisualFingerprintItem)> = fps
        .into_iter()
        .filter_map(|fp| sidecar_to_original.get(&fp.path).cloned().map(|p| (p, fp)))
        .collect();
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    let mut buckets: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, (_, fp)) in keyed.iter().enumerate() {
        buckets.entry(fp.ahash.clone()).or_default().push(idx);
    }
    let mut used = HashSet::new();
    let mut groups = vec![];
    for indices in buckets.values() {
        if indices.len() < 2 {
            continue;
        }
        for &i in indices {
            if used.contains(&keyed[i].0) {
                continue;
            }
            let mut paths = vec![keyed[i].0.clone()];
            let mut best_distance = 0;
            for &j in indices {
                if j <= i {
                    continue;
                }
                if used.contains(&keyed[j].0) {
                    continue;
                }
                let pd = hex_hamming(&keyed[i].1.phash, &keyed[j].1.phash).unwrap_or(65);
                let dd = hex_hamming(&keyed[i].1.dhash, &keyed[j].1.dhash).unwrap_or(65);
                let ad = hex_hamming(&keyed[i].1.ahash, &keyed[j].1.ahash).unwrap_or(65);
                if pd <= 6 && dd <= 8 && ad <= 10 {
                    best_distance = best_distance.max(pd + dd + ad);
                    paths.push(keyed[j].0.clone());
                }
            }
            if paths.len() < 2 {
                continue;
            }
            for p in &paths {
                used.insert(p.clone());
            }
            let mut candidates: Vec<CleanupCandidate> = paths
                .iter()
                .filter_map(|p| by_path.get(p))
                .map(cleanup_candidate_from_media)
                .collect();
            for candidate in &mut candidates {
                if let Some((_, fp)) = keyed.iter().find(|(p, _)| p == &candidate.path) {
                    candidate.width = Some(fp.width);
                    candidate.height = Some(fp.height);
                }
            }
            let Some(keeper) = choose_cleanup_keeper(&candidates) else {
                continue;
            };
            let id = format!("visual-{}", groups.len() + 1);
            groups.push(CleanupDuplicateGroup {
                id,
                kind: "visual-image".into(),
                reason: "Strict perceptual hash match".into(),
                score: Some(best_distance),
                default_keep_path: keeper,
                candidates,
            });
        }
    }
    Ok((groups, cache_updates))
}

fn generate_cleanup_plan_impl(
    app: tauri::AppHandle,
    paths: Option<Vec<String>>,
) -> Result<CleanupPlan, String> {
    let roots = if let Some(paths) = paths.filter(|p| !p.is_empty()) {
        paths
    } else {
        list_library_folders(app.clone())?
    };
    let mut errors = vec![];
    let mut ignored_files = vec![];
    let mut empty_folders = vec![];
    emit_cleanup_progress(
        &app,
        "Finding ignored files",
        None,
        0,
        Some(roots.len()),
        0,
        false,
    );
    for (i, root) in roots.iter().enumerate() {
        let path = Path::new(root);
        if path.is_dir() {
            let mut visited = 0usize;
            collect_ignored_cleanup_files(
                &app,
                path,
                &mut ignored_files,
                &mut errors,
                &mut visited,
            );
            emit_cleanup_progress(
                &app,
                "Finding empty folders",
                Some(root.clone()),
                0,
                None,
                errors.len(),
                false,
            );
            collect_empty_folders(&app, path, &mut empty_folders, &mut errors);
        } else {
            errors.push(format!("cleanup root is not a directory: {root}"));
        }
        emit_cleanup_progress(
            &app,
            "Finding ignored files",
            Some(root.clone()),
            i + 1,
            Some(roots.len()),
            errors.len(),
            false,
        );
    }
    let mut conn = open_db(&app)?;
    let cleanup_cache = load_cleanup_file_cache(&conn)?;
    let image_items: Vec<MediaItem> = cleanup_image_items(&conn, &app)?
        .into_iter()
        .filter(|item| path_is_under_roots(&item.path, &roots))
        .collect();
    emit_cleanup_progress(
        &app,
        &format!(
            "Hashing exact image duplicates ({} workers)",
            cleanup_hash_thread_count()
        ),
        None,
        0,
        Some(image_items.len()),
        errors.len(),
        false,
    );
    let mut by_hash: HashMap<String, Vec<CleanupCandidate>> = HashMap::new();
    let mut cache_updates = Vec::new();
    let mut cached_hashes = 0usize;
    for item in &image_items {
        if let Some(cache) = cleanup_cache
            .get(&item.path)
            .filter(|cache| cache_matches_media(cache, item) && cache.sha256.is_some())
        {
            by_hash
                .entry(cache.sha256.clone().unwrap_or_default())
                .or_default()
                .push(cleanup_candidate_from_media(item));
            cached_hashes += 1;
        }
    }
    let hash_items: Vec<MediaItem> = image_items
        .iter()
        .filter(|item| {
            !cleanup_cache
                .get(&item.path)
                .is_some_and(|cache| cache_matches_media(cache, item) && cache.sha256.is_some())
        })
        .cloned()
        .collect();
    emit_cleanup_progress(
        &app,
        &format!("Using cached exact hashes ({cached_hashes} cached)"),
        None,
        cached_hashes,
        Some(image_items.len()),
        errors.len(),
        false,
    );
    let hash_threads = cleanup_hash_thread_count();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(hash_threads)
        .build()
        .map_err(|e| format!("cleanup hash thread pool: {e}"))?;
    let (tx, rx) = mpsc::channel();
    let hash_total = hash_items.len();
    let hash_handle = thread::spawn(move || {
        pool.install(|| {
            hash_items.par_iter().for_each_with(tx, |tx, item| {
                let res = file_sha256(Path::new(&item.path))
                    .map(|hash| (hash, cleanup_candidate_from_media(item)));
                let _ = tx.send((item.path.clone(), res));
            });
        });
    });
    for (i, (path, res)) in rx.into_iter().enumerate() {
        match res {
            Ok((hash, candidate)) => {
                cache_updates.push(CleanupCacheUpdate {
                    path: candidate.path.clone(),
                    size_bytes: candidate.size_bytes,
                    modified_at: candidate.modified_at,
                    sha256: Some(hash.clone()),
                    width: None,
                    height: None,
                    ahash: None,
                    dhash: None,
                    phash: None,
                });
                by_hash.entry(hash).or_default().push(candidate);
            }
            Err(e) => errors.push(e),
        }
        if i % 25 == 0 {
            emit_cleanup_progress(
                &app,
                &format!("Hashing exact image duplicates ({hash_threads} workers)"),
                Some(path),
                cached_hashes + i + 1,
                Some(image_items.len()),
                errors.len(),
                false,
            );
        }
    }
    if hash_handle.join().is_err() {
        errors.push("exact hash worker thread panicked".into());
    }
    let mut duplicate_groups = vec![];
    let mut exact_paths = HashSet::new();
    for (_, candidates) in by_hash.into_iter().filter(|(_, v)| v.len() > 1) {
        let Some(keeper) = choose_cleanup_keeper(&candidates) else {
            continue;
        };
        for c in &candidates {
            exact_paths.insert(c.path.clone());
        }
        duplicate_groups.push(CleanupDuplicateGroup {
            id: format!("exact-{}", duplicate_groups.len() + 1),
            kind: "exact-image".into(),
            reason: "Byte-identical image file hash".into(),
            score: Some(0),
            default_keep_path: keeper,
            candidates,
        });
    }
    emit_cleanup_progress(
        &app,
        "Finding visual image duplicates",
        None,
        0,
        Some(image_items.len()),
        errors.len(),
        false,
    );
    if hash_total > 0 {
        emit_cleanup_progress(
            &app,
            &format!("Hashing exact image duplicates ({hash_threads} workers)"),
            None,
            image_items.len().min(cached_hashes + hash_total),
            Some(image_items.len()),
            errors.len(),
            false,
        );
    }
    let (mut visual_groups, mut visual_cache_updates) = build_visual_duplicate_groups(
        &app,
        &image_items,
        &exact_paths,
        &cleanup_cache,
        &mut errors,
    )?;
    cache_updates.append(&mut visual_cache_updates);
    write_cleanup_cache_updates(&mut conn, &cache_updates)?;
    duplicate_groups.append(&mut visual_groups);
    let duplicate_files: usize = duplicate_groups
        .iter()
        .map(|g| g.candidates.len().saturating_sub(1))
        .sum();
    let selected_duplicate_bytes: i64 = duplicate_groups
        .iter()
        .flat_map(|g| {
            g.candidates
                .iter()
                .filter(|c| c.path != g.default_keep_path)
                .map(|c| c.size_bytes.unwrap_or(0))
        })
        .sum();
    let ignored_bytes: i64 = ignored_files
        .iter()
        .map(|e| e.size_bytes.unwrap_or(0))
        .sum();
    let totals = CleanupTotals {
        ignored_files: ignored_files.len(),
        empty_folders: empty_folders.len(),
        duplicate_groups: duplicate_groups.len(),
        duplicate_files,
        selected_files: ignored_files.len() + duplicate_files,
        selected_folders: empty_folders.len(),
        selected_bytes: ignored_bytes + selected_duplicate_bytes,
        errors: errors.len(),
    };
    let plan = CleanupPlan {
        plan_id: cleanup_plan_id(),
        ignored_files,
        empty_folders,
        duplicate_groups,
        totals,
        errors,
    };
    cleanup_plans()
        .lock()
        .map_err(|_| "cleanup plan lock poisoned".to_string())?
        .insert(plan.plan_id.clone(), plan.clone());
    emit_cleanup_progress(
        &app,
        "Cleanup plan ready",
        None,
        plan.totals.selected_files,
        Some(plan.totals.selected_files),
        plan.errors.len(),
        true,
    );
    Ok(plan)
}

#[tauri::command]
async fn generate_cleanup_plan(
    app: tauri::AppHandle,
    paths: Option<Vec<String>>,
) -> Result<CleanupPlan, String> {
    tauri::async_runtime::spawn_blocking(move || generate_cleanup_plan_impl(app, paths))
        .await
        .map_err(|e| format!("cleanup plan task failed: {e}"))?
}

#[tauri::command]
fn get_cleanup_cache_info(
    app: tauri::AppHandle,
    paths: Option<Vec<String>>,
) -> Result<CleanupCacheInfo, String> {
    let roots = if let Some(paths) = paths.filter(|p| !p.is_empty()) {
        paths
    } else {
        list_library_folders(app.clone())?
    };
    let conn = open_db(&app)?;
    let cleanup_cache = load_cleanup_file_cache(&conn)?;
    let mut stmt = conn
        .prepare(
            r#"SELECT
                COUNT(*),
                SUM(CASE WHEN sha256 IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN ahash IS NOT NULL AND dhash IS NOT NULL AND phash IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN sha256 IS NOT NULL AND ahash IS NOT NULL AND dhash IS NOT NULL AND phash IS NOT NULL THEN 1 ELSE 0 END),
                SUM(LENGTH(path) + COALESCE(LENGTH(sha256),0) + COALESCE(LENGTH(ahash),0) + COALESCE(LENGTH(dhash),0) + COALESCE(LENGTH(phash),0) + 64),
                MIN(updated_at),
                MAX(updated_at)
               FROM cleanup_file_cache"#,
        )
        .map_err(|e| e.to_string())?;
    let (
        total_entries,
        exact_hash_entries,
        visual_fingerprint_entries,
        complete_entries,
        cache_bytes_estimate,
        oldest_updated_at,
        newest_updated_at,
    ) = stmt
        .query_row([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Option<i64>>(1)?.unwrap_or(0),
                row.get::<_, Option<i64>>(2)?.unwrap_or(0),
                row.get::<_, Option<i64>>(3)?.unwrap_or(0),
                row.get::<_, Option<i64>>(4)?.unwrap_or(0),
                row.get::<_, Option<i64>>(5)?,
                row.get::<_, Option<i64>>(6)?,
            ))
        })
        .map_err(|e| e.to_string())?;
    let image_items: Vec<MediaItem> = cleanup_image_items(&conn, &app)?
        .into_iter()
        .filter(|item| path_is_under_roots(&item.path, &roots))
        .collect();
    let mut selected_exact_cached = 0usize;
    let mut selected_visual_cached = 0usize;
    let mut selected_complete_cached = 0usize;
    let mut stale_selected_entries = 0usize;
    for item in &image_items {
        if let Some(cache) = cleanup_cache.get(&item.path) {
            if cache_matches_media(cache, item) {
                let exact = cache.sha256.is_some();
                let visual = cache.width.is_some()
                    && cache.height.is_some()
                    && cache.ahash.is_some()
                    && cache.dhash.is_some()
                    && cache.phash.is_some();
                if exact {
                    selected_exact_cached += 1;
                }
                if visual {
                    selected_visual_cached += 1;
                }
                if exact && visual {
                    selected_complete_cached += 1;
                }
            } else {
                stale_selected_entries += 1;
            }
        }
    }
    Ok(CleanupCacheInfo {
        total_entries,
        exact_hash_entries,
        visual_fingerprint_entries,
        complete_entries,
        selected_images: image_items.len(),
        selected_exact_cached,
        selected_visual_cached,
        selected_complete_cached,
        selected_uncached_exact: image_items.len().saturating_sub(selected_exact_cached),
        selected_uncached_visual: image_items.len().saturating_sub(selected_visual_cached),
        stale_selected_entries,
        cache_bytes_estimate,
        oldest_updated_at,
        newest_updated_at,
    })
}

#[tauri::command]
fn clear_cleanup_cache(app: tauri::AppHandle) -> Result<usize, String> {
    let conn = open_db(&app)?;
    let deleted = conn
        .execute("DELETE FROM cleanup_file_cache", [])
        .map_err(|e| format!("failed to clear cleanup cache: {e}"))?;
    Ok(deleted)
}

#[tauri::command]
fn cancel_cleanup_plan(plan_id: String) -> Result<(), String> {
    cleanup_plans()
        .lock()
        .map_err(|_| "cleanup plan lock poisoned".to_string())?
        .remove(&plan_id);
    Ok(())
}

#[tauri::command]
async fn apply_cleanup_plan(
    app: tauri::AppHandle,
    plan_id: String,
    selections: ApplyCleanupSelections,
) -> Result<ApplyCleanupResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let plan = cleanup_plans()
            .lock()
            .map_err(|_| "cleanup plan lock poisoned".to_string())?
            .get(&plan_id)
            .cloned()
            .ok_or_else(|| "cleanup plan expired; generate a new plan".to_string())?;
        let requested_remove: HashSet<String> = selections.remove_paths.iter().cloned().collect();
        for group in &plan.duplicate_groups {
            if group
                .candidates
                .iter()
                .all(|candidate| requested_remove.contains(&candidate.path))
            {
                return Err(format!(
                    "refusing to delete every file in duplicate group {}",
                    group.id
                ));
            }
        }
        let allowed_files: HashMap<String, i64> = plan
            .ignored_files
            .iter()
            .map(|e| (e.path.clone(), e.size_bytes.unwrap_or(0)))
            .chain(plan.duplicate_groups.iter().flat_map(|g| {
                g.candidates
                    .iter()
                    .map(|c| (c.path.clone(), c.size_bytes.unwrap_or(0)))
            }))
            .collect();
        let allowed_folders: HashSet<String> =
            plan.empty_folders.iter().map(|e| e.path.clone()).collect();
        let mut result = ApplyCleanupResult {
            files_deleted: 0,
            folders_deleted: 0,
            bytes_deleted: 0,
            rows_marked_missing: 0,
            errors: vec![],
        };
        let total = selections.remove_paths.len() + selections.empty_folders.len();
        for (i, path) in selections.remove_paths.iter().enumerate() {
            emit_cleanup_progress(
                &app,
                "Permanently deleting files",
                Some(path.clone()),
                i + 1,
                Some(total),
                result.errors.len(),
                false,
            );
            let Some(size) = allowed_files.get(path).copied() else {
                result
                    .errors
                    .push(format!("refusing unplanned file deletion: {path}"));
                continue;
            };
            match fs::remove_file(path) {
                Ok(_) => {
                    result.files_deleted += 1;
                    result.bytes_deleted += size;
                }
                Err(e) => result
                    .errors
                    .push(format!("failed to permanently delete {path}: {e}")),
            }
        }
        let mut folders = selections.empty_folders.clone();
        folders.sort_by_key(|p| std::cmp::Reverse(Path::new(p).components().count()));
        for folder in folders {
            if !allowed_folders.contains(&folder) {
                result
                    .errors
                    .push(format!("refusing unplanned folder deletion: {folder}"));
                continue;
            }
            match fs::read_dir(&folder) {
                Ok(mut rd) => {
                    if rd.next().is_none() {
                        match fs::remove_dir(&folder) {
                            Ok(_) => result.folders_deleted += 1,
                            Err(e) => result
                                .errors
                                .push(format!("failed to permanently delete folder {folder}: {e}")),
                        }
                    } else {
                        result
                            .errors
                            .push(format!("folder not empty after file cleanup: {folder}"));
                    }
                }
                Err(e) => result
                    .errors
                    .push(format!("failed to inspect folder {folder}: {e}")),
            }
        }
        let conn = open_db(&app)?;
        for path in selections.remove_paths {
            if allowed_files.contains_key(&path) {
                if conn
                    .execute(
                        "UPDATE media_items SET missing=1 WHERE path=?1",
                        params![path],
                    )
                    .map_err(|e| e.to_string())?
                    > 0
                {
                    result.rows_marked_missing += 1;
                }
            }
        }
        cleanup_plans()
            .lock()
            .map_err(|_| "cleanup plan lock poisoned".to_string())?
            .remove(&plan_id);
        emit_cleanup_progress(
            &app,
            "Cleanup complete",
            None,
            total,
            Some(total),
            result.errors.len(),
            true,
        );
        Ok(result)
    })
    .await
    .map_err(|e| format!("cleanup apply task failed: {e}"))?
}
#[tauri::command]
async fn rescan(app: tauri::AppHandle) -> Result<ScanSummary, String> {
    let paths = list_library_folders(app.clone())?;
    scan_library(app, paths).await
}
#[tauri::command]
fn get_media_item(app: tauri::AppHandle, id: i64) -> Result<Option<MediaItem>, String> {
    let mut item = open_db(&app)?
        .query_row(
            &format!("{MEDIA_SELECT} WHERE id=?1"),
            params![id],
            row_to_media,
        )
        .optional()
        .map_err(|e| e.to_string())?;
    if let Some(item) = item.as_mut() {
        enrich_display_paths(&app, std::slice::from_mut(item))?;
    }
    Ok(item)
}
#[tauri::command]
fn search_media(app: tauri::AppHandle, filter: SearchFilter) -> Result<Vec<MediaItem>, String> {
    let c = open_db(&app)?;
    let mut sql = format!("{MEDIA_SELECT} WHERE 1=1");
    sql.push_str(&sql_path_not_blacklisted_clause());
    let mut v: Vec<Box<dyn ToSql>> = vec![];
    let radius_filter = match (filter.lat, filter.lng, filter.radius_km) {
        (Some(lat), Some(lng), Some(radius_km))
            if (-90.0..=90.0).contains(&lat)
                && (-180.0..=180.0).contains(&lng)
                && radius_km.is_finite()
                && radius_km > 0.0 =>
        {
            Some((lat, lng, radius_km))
        }
        _ => None,
    };
    if filter.person_id.is_some() || filter.person_name.is_some() {
        sql.push_str(" AND media_items.id IN (SELECT DISTINCT f.media_item_id FROM faces f");
        if filter.person_name.is_some() {
            sql.push_str(" JOIN people p ON p.id=f.person_id");
        }
        sql.push_str(" WHERE 1=1");
        if let Some(id) = filter.person_id {
            sql.push_str(" AND f.person_id=?");
            v.push(Box::new(id));
        }
        if let Some(n) = filter.person_name {
            sql.push_str(" AND p.name LIKE ? COLLATE NOCASE");
            v.push(Box::new(format!("%{}%", n.trim())));
        }
        sql.push(')');
    }
    if let Some(q) = filter.query.filter(|q| !q.trim().is_empty()) {
        sql.push_str(" AND (file_name LIKE ? OR path LIKE ?)");
        let like = format!("%{}%", q.trim());
        v.push(Box::new(like.clone()));
        v.push(Box::new(like));
    }
    if let Some(x) = filter.media_type {
        sql.push_str(" AND media_type=?");
        v.push(Box::new(x));
    }
    if let Some(x) = filter.missing {
        sql.push_str(" AND missing=?");
        v.push(Box::new(if x { 1_i64 } else { 0_i64 }));
    }
    if let Some(x) = filter.from_ts {
        sql.push_str(" AND COALESCE(captured_at,modified_at,created_at)>=?");
        v.push(Box::new(x));
    }
    if let Some(x) = filter.to_ts {
        sql.push_str(" AND COALESCE(captured_at,modified_at,created_at)<=?");
        v.push(Box::new(x));
    }
    if let Some(cam) = filter.camera {
        sql.push_str(" AND ((camera_make||' '||camera_model) LIKE ? OR camera_model LIKE ?)");
        let like = format!("%{}%", cam);
        v.push(Box::new(like.clone()));
        v.push(Box::new(like));
    }
    if let Some(x) = filter.has_gps {
        sql.push_str(if x {
            " AND latitude IS NOT NULL AND longitude IS NOT NULL"
        } else {
            " AND (latitude IS NULL OR longitude IS NULL)"
        });
    }
    if let Some(x) = filter.has_camera {
        sql.push_str(if x {
            " AND (camera_make IS NOT NULL OR camera_model IS NOT NULL)"
        } else {
            " AND camera_make IS NULL AND camera_model IS NULL"
        });
    }
    if let Some((lat, lng, radius_km)) = radius_filter {
        let (min_lat, max_lat, lng_bounds) = radius_bounds(lat, lng, radius_km);
        sql.push_str(
            " AND latitude IS NOT NULL AND longitude IS NOT NULL AND latitude>=? AND latitude<=?",
        );
        v.push(Box::new(min_lat));
        v.push(Box::new(max_lat));
        if let Some((min_lng, max_lng)) = lng_bounds {
            sql.push_str(" AND longitude>=? AND longitude<=?");
            v.push(Box::new(min_lng));
            v.push(Box::new(max_lng));
        }
    }
    let sort_dir = if filter
        .sort_order
        .as_deref()
        .is_some_and(|order| order.eq_ignore_ascii_case("asc"))
    {
        "ASC"
    } else {
        "DESC"
    };
    sql.push_str(&format!(
        " ORDER BY COALESCE(captured_at,modified_at,created_at) {sort_dir},id {sort_dir}"
    ));
    if radius_filter.is_none() {
        sql.push_str(" LIMIT ? OFFSET ?");
        v.push(Box::new(filter.limit.unwrap_or(100).clamp(1, 500)));
        v.push(Box::new(filter.offset.unwrap_or(0).max(0)));
    }
    let refs: Vec<&dyn ToSql> = v.iter().map(|x| x.as_ref()).collect();
    let mut st = c.prepare(&sql).map_err(|e| e.to_string())?;
    let mut res = st
        .query_map(params_from_iter(refs), row_to_media)
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    if let Some((lat, lng, radius_km)) = radius_filter {
        let offset = filter.offset.unwrap_or(0).max(0) as usize;
        let limit = filter.limit.unwrap_or(100).clamp(1, 500) as usize;
        res = res
            .into_iter()
            .filter(|item| {
                item.latitude
                    .zip(item.longitude)
                    .is_some_and(|(item_lat, item_lng)| {
                        haversine_km(lat, lng, item_lat, item_lng) <= radius_km
                    })
            })
            .skip(offset)
            .take(limit)
            .collect();
    }
    enrich_display_paths(&app, &mut res)?;
    Ok(res)
}
#[tauri::command]
fn list_geo_points(app: tauri::AppHandle) -> Result<Vec<GeoPoint>, String> {
    let c = open_db(&app)?;
    let mut s = c
        .prepare(
            "SELECT latitude,longitude FROM media_items WHERE media_type='image' AND missing=0 AND latitude IS NOT NULL AND longitude IS NOT NULL AND lower(file_name) NOT LIKE 'facetile%'",
        )
        .map_err(|e| e.to_string())?;
    let points = s
        .query_map([], |r| {
            Ok(GeoPoint {
                latitude: r.get(0)?,
                longitude: r.get(1)?,
            })
        })
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    Ok(points)
}
#[tauri::command]
fn list_people(app: tauri::AppHandle) -> Result<Vec<Person>, String> {
    let c = open_db(&app)?;
    let mut s=c.prepare("SELECT p.id,p.name,p.created_at,COUNT(f.id) FROM people p LEFT JOIN faces f ON f.person_id=p.id GROUP BY p.id ORDER BY p.name").map_err(|e|e.to_string())?;
    let res = s
        .query_map([], |r| {
            Ok(Person {
                id: r.get(0)?,
                name: r.get(1)?,
                created_at: r.get(2)?,
                face_count: r.get(3)?,
            })
        })
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string());
    res
}
#[tauri::command]
fn rename_person(app: tauri::AppHandle, person_id: i64, name: String) -> Result<(), String> {
    open_db(&app)?
        .execute(
            "UPDATE people SET name=?1 WHERE id=?2",
            params![name, person_id],
        )
        .map_err(|e| e.to_string())?;
    Ok(())
}
#[tauri::command]
fn delete_person(app: tauri::AppHandle, person_id: i64) -> Result<(), String> {
    let conn = open_db(&app)?;
    let n = conn
        .execute("DELETE FROM people WHERE id=?1", params![person_id])
        .map_err(|e| e.to_string())?;
    if n == 0 {
        return Err("Person not found".to_string());
    }
    Ok(())
}
fn get_or_create_person(conn: &Connection, name: &str) -> Result<i64, String> {
    let trimmed = name.trim();
    let name = if trimmed.is_empty() {
        "Unnamed"
    } else {
        trimmed
    };
    conn.execute(
        "INSERT OR IGNORE INTO people(name,created_at) VALUES(?1,?2)",
        params![name, now_unix()],
    )
    .map_err(|e| e.to_string())?;
    conn.query_row("SELECT id FROM people WHERE name=?1", params![name], |r| {
        r.get(0)
    })
    .map_err(|e| e.to_string())
}
fn best_named_person_for_embedding(
    embedding: &[f32],
    named_faces: &[NamedFaceEmbedding],
    threshold: f64,
) -> Option<i64> {
    let mut best_score = threshold;
    let mut best_person_id = None;
    for named_face in named_faces {
        if named_face.embedding.len() != embedding.len() {
            continue;
        }
        let score = cosine(embedding, &named_face.embedding);
        if score >= best_score {
            best_score = score;
            best_person_id = Some(named_face.person_id);
        }
    }
    best_person_id
}
fn load_named_face_embeddings(conn: &Connection) -> Result<Vec<NamedFaceEmbedding>, String> {
    let mut s = conn
        .prepare(
            "SELECT f.person_id,e.vector FROM embeddings e JOIN faces f ON f.id=e.face_id \
             WHERE f.person_id IS NOT NULL",
        )
        .map_err(|e| e.to_string())?;
    let mut named_faces = vec![];
    for r in s
        .query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
        })
        .map_err(|e| e.to_string())?
    {
        let (pid, blob) = r.map_err(|e| e.to_string())?;
        if let Some(v) = parse_vec(blob) {
            named_faces.push(NamedFaceEmbedding {
                person_id: pid,
                embedding: v,
            });
        }
    }
    Ok(named_faces)
}
fn propagate_person(
    conn: &Connection,
    person_id: i64,
    seed: &[f32],
    threshold: f64,
    exclude_media_item_id: i64,
) -> Result<usize, String> {
    let mut s = conn
        .prepare(
            "SELECT e.face_id,e.vector,f.media_item_id FROM embeddings e JOIN faces f ON f.id=e.face_id \
             WHERE f.person_id IS NULL AND f.media_item_id != ?1",
        )
        .map_err(|e| e.to_string())?;
    let rows = s
        .query_map(params![exclude_media_item_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    let mut candidates: Vec<serde_json::Value> = vec![];
    for (face_id, blob, media_item_id) in rows {
        if let Some(v) = parse_vec(blob) {
            if v.len() != seed.len() {
                continue;
            }
            candidates.push(serde_json::json!({
                "face_id": face_id,
                "media_item_id": media_item_id,
                "embedding": v
            }));
        }
    }
    if candidates.is_empty() {
        return Ok(0);
    }
    let payload = serde_json::json!({
        "mode": "propagate",
        "seed": seed,
        "threshold": threshold,
        "exclude_media_item_id": exclude_media_item_id,
        "candidates": candidates
    });
    let res = run_sidecar_stdin(&payload.to_string())?;
    if !res.ok {
        return Err(format!("face-match sidecar failed: {}", res.stderr.trim()));
    }
    let data = parse_sidecar_ok_data(&res.stdout)?;
    let ids: Vec<i64> = data
        .get("matching_face_ids")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|x| x.as_i64()).collect::<Vec<_>>())
        .unwrap_or_default();
    let mut n = 0usize;
    for face_id in ids {
        conn.execute(
            "UPDATE faces SET person_id=?1 WHERE id=?2",
            params![person_id, face_id],
        )
        .map_err(|e| e.to_string())?;
        n += 1;
    }
    Ok(n)
}
#[tauri::command]
fn name_face(app: tauri::AppHandle, face_id: i64, name: String) -> Result<usize, String> {
    let conn = open_db(&app)?;
    let media_item_id: i64 = conn
        .query_row(
            "SELECT media_item_id FROM faces WHERE id=?1",
            params![face_id],
            |r| r.get(0),
        )
        .map_err(|e| e.to_string())?;
    let pid = get_or_create_person(&conn, &name)?;
    conn.execute(
        "UPDATE faces SET person_id=?1 WHERE id=?2",
        params![pid, face_id],
    )
    .map_err(|e| e.to_string())?;
    let blob: Vec<u8> = conn
        .query_row(
            "SELECT vector FROM embeddings WHERE face_id=?1 ORDER BY id DESC LIMIT 1",
            params![face_id],
            |r| r.get(0),
        )
        .optional()
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "face has no embedding yet".to_string())?;
    let seed = parse_vec(blob).ok_or_else(|| "invalid face embedding".to_string())?;
    propagate_person(
        &conn,
        pid,
        &seed,
        INSIGHTFACE_FACE_MATCH_THRESHOLD,
        media_item_id,
    )
}
#[tauri::command]
fn list_faces(
    app: tauri::AppHandle,
    media_item_id: Option<i64>,
    person_id: Option<i64>,
) -> Result<Vec<Face>, String> {
    let c = open_db(&app)?;
    let mut sql="SELECT f.id,f.media_item_id,f.person_id,p.name,f.x,f.y,f.width,f.height,f.confidence,f.created_at FROM faces f LEFT JOIN people p ON p.id=f.person_id WHERE 1=1".to_string();
    let mut v: Vec<Box<dyn ToSql>> = vec![];
    if let Some(x) = media_item_id {
        sql.push_str(" AND f.media_item_id=?");
        v.push(Box::new(x));
    }
    if let Some(x) = person_id {
        sql.push_str(" AND f.person_id=?");
        v.push(Box::new(x));
    }
    let refs: Vec<&dyn ToSql> = v.iter().map(|x| x.as_ref()).collect();
    let mut s = c.prepare(&sql).map_err(|e| e.to_string())?;
    let res = s
        .query_map(params_from_iter(refs), |r| {
            Ok(Face {
                id: r.get(0)?,
                media_item_id: r.get(1)?,
                person_id: r.get(2)?,
                person_name: r.get(3)?,
                x: r.get(4)?,
                y: r.get(5)?,
                width: r.get(6)?,
                height: r.get(7)?,
                confidence: r.get(8)?,
                created_at: r.get(9)?,
            })
        })
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string());
    res
}
fn sidecar_dir() -> Result<PathBuf, String> {
    Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| "failed to resolve repo root".to_string())?
        .join("python-sidecar"))
}
fn sidecar_python(dir: &Path) -> PathBuf {
    let windows_python = dir.join(".venv").join("Scripts").join("python.exe");
    if windows_python.exists() {
        return windows_python;
    }
    let unix_python = dir.join(".venv").join("bin").join("python");
    if unix_python.exists() {
        return unix_python;
    }
    PathBuf::from("python3")
}
fn sidecar_command(dir: &Path) -> Command {
    let mut command = Command::new(sidecar_python(dir));
    command
        .arg("-m")
        .arg("rich_media_sidecar")
        .current_dir(dir)
        .env("PYTHONPATH", dir.to_string_lossy().to_string());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    command
}
fn run_sidecar(args: Vec<String>) -> Result<SidecarResult, String> {
    let dir = sidecar_dir()?;
    backend_log(&format!("running python sidecar args={args:?}"));
    let out = sidecar_command(&dir).args(args).output().map_err(|e| {
        let msg = format!("failed to run python sidecar: {e}");
        backend_log(&msg);
        msg
    })?;
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    if !out.status.success() || !stderr.trim().is_empty() {
        backend_log(&format!(
            "python sidecar status={} stderr={}",
            out.status,
            stderr.trim()
        ));
    }
    Ok(SidecarResult {
        ok: out.status.success(),
        stdout,
        stderr,
    })
}

fn run_sidecar_json_payload(
    mut args: Vec<String>,
    payload: &serde_json::Value,
) -> Result<SidecarResult, String> {
    let mut path = std::env::temp_dir();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    path.push(format!("rich-media-viewer-sidecar-{unique}.json"));
    fs::write(&path, payload.to_string())
        .map_err(|e| format!("failed to write sidecar payload {}: {e}", path.display()))?;
    args.push("--json".into());
    args.push(format!("@{}", path.display()));
    let res = run_sidecar(args);
    let _ = fs::remove_file(&path);
    res
}

fn spawn_face_sidecar(dir: &Path) -> Result<FaceSidecarProcess, String> {
    backend_log("starting persistent python face sidecar");
    let mut child = sidecar_command(dir)
        .arg("serve-faces")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| {
            let msg = format!("failed to spawn persistent face sidecar: {e}");
            backend_log(&msg);
            msg
        })?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "failed to open face sidecar stdin".to_string())?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "failed to open face sidecar stdout".to_string())?;
    Ok(FaceSidecarProcess {
        child,
        stdin,
        stdout: BufReader::new(stdout),
    })
}
fn take_face_sidecar(dir: &Path, pool_size: usize) -> Result<(usize, FaceSidecarProcess), String> {
    let pool_size = pool_size.max(1);
    let pool = FACE_SIDECARS.get_or_init(|| Mutex::new(Vec::new()));
    loop {
        let mut guard = pool
            .lock()
            .map_err(|_| "face sidecar pool mutex poisoned".to_string())?;
        for (idx, slot) in guard.iter_mut().enumerate() {
            if !slot.in_use {
                slot.in_use = true;
                let mut process = match slot.process.take() {
                    Some(process) => process,
                    None => {
                        drop(guard);
                        return match spawn_face_sidecar(dir) {
                            Ok(process) => Ok((idx, process)),
                            Err(e) => {
                                release_face_sidecar_slot(idx)?;
                                Err(e)
                            }
                        };
                    }
                };
                let exited = match process.child.try_wait() {
                    Ok(status) => status.is_some(),
                    Err(e) => {
                        drop(guard);
                        release_face_sidecar_slot(idx)?;
                        return Err(format!("failed to check face sidecar status: {e}"));
                    }
                };
                if exited {
                    drop(guard);
                    return match spawn_face_sidecar(dir) {
                        Ok(process) => Ok((idx, process)),
                        Err(e) => {
                            release_face_sidecar_slot(idx)?;
                            Err(e)
                        }
                    };
                }
                return Ok((idx, process));
            }
        }
        if guard.len() < pool_size {
            let idx = guard.len();
            guard.push(FaceSidecarSlot {
                process: None,
                in_use: true,
            });
            drop(guard);
            return match spawn_face_sidecar(dir) {
                Ok(process) => Ok((idx, process)),
                Err(e) => {
                    release_face_sidecar_slot(idx)?;
                    Err(e)
                }
            };
        }
        drop(guard);
        thread::sleep(Duration::from_millis(10));
    }
}

fn return_face_sidecar(idx: usize, process: FaceSidecarProcess) -> Result<(), String> {
    let pool = FACE_SIDECARS.get_or_init(|| Mutex::new(Vec::new()));
    let mut guard = pool
        .lock()
        .map_err(|_| "face sidecar pool mutex poisoned".to_string())?;
    if idx >= guard.len() {
        guard.resize_with(idx + 1, || FaceSidecarSlot {
            process: None,
            in_use: false,
        });
    }
    guard[idx].process = Some(process);
    guard[idx].in_use = false;
    Ok(())
}

fn release_face_sidecar_slot(idx: usize) -> Result<(), String> {
    let pool = FACE_SIDECARS.get_or_init(|| Mutex::new(Vec::new()));
    let mut guard = pool
        .lock()
        .map_err(|_| "face sidecar pool mutex poisoned".to_string())?;
    if let Some(slot) = guard.get_mut(idx) {
        slot.process = None;
        slot.in_use = false;
    }
    Ok(())
}

fn run_face_sidecar(payload: &str, pool_size: usize) -> Result<SidecarResult, String> {
    let dir = sidecar_dir()?;
    let (idx, mut process) = take_face_sidecar(&dir, pool_size)?;
    if let Err(exc) = writeln!(process.stdin, "{payload}").and_then(|_| process.stdin.flush()) {
        backend_log(&format!(
            "restarting face sidecar after write failure: {exc}"
        ));
        process = spawn_face_sidecar(&dir)?;
        if let Err(e) = writeln!(process.stdin, "{payload}").and_then(|_| process.stdin.flush()) {
            release_face_sidecar_slot(idx)?;
            return Err(format!("failed to write face sidecar stdin: {e}"));
        }
    }
    let mut stdout = String::new();
    if let Err(e) = process.stdout.read_line(&mut stdout) {
        release_face_sidecar_slot(idx)?;
        return Err(format!("failed to read face sidecar stdout: {e}"));
    }
    if stdout.trim().is_empty() {
        release_face_sidecar_slot(idx)?;
        return Err("face sidecar exited without a response".to_string());
    }
    let ok = parse_sidecar_ok_data(&stdout).is_ok();
    return_face_sidecar(idx, process)?;
    Ok(SidecarResult {
        ok,
        stdout,
        stderr: String::new(),
    })
}

fn run_sidecar_stdin(stdin_body: &str) -> Result<SidecarResult, String> {
    let dir = sidecar_dir()?;
    backend_log("running python sidecar face-match --stdin");
    let mut child = sidecar_command(&dir)
        .arg("face-match")
        .arg("--stdin")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            let msg = format!("failed to spawn python sidecar: {e}");
            backend_log(&msg);
            msg
        })?;
    child
        .stdin
        .take()
        .ok_or_else(|| "failed to open sidecar stdin".to_string())?
        .write_all(stdin_body.as_bytes())
        .map_err(|e| format!("failed to write sidecar stdin: {e}"))?;
    let out = child
        .wait_with_output()
        .map_err(|e| format!("failed to read sidecar output: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    if !out.status.success() || !stderr.trim().is_empty() {
        backend_log(&format!(
            "python sidecar status={} stderr={}",
            out.status,
            stderr.trim()
        ));
    }
    Ok(SidecarResult {
        ok: out.status.success(),
        stdout,
        stderr,
    })
}

fn parse_sidecar_ok_data(stdout: &str) -> Result<serde_json::Value, String> {
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).map_err(|e| {
        format!(
            "invalid JSON from sidecar: {e}; stdout={}",
            stdout.chars().take(500).collect::<String>()
        )
    })?;
    match v.get("ok").and_then(|x| x.as_bool()) {
        Some(true) => v
            .get("data")
            .cloned()
            .ok_or_else(|| "sidecar response missing data".to_string()),
        _ => {
            let msg = v
                .pointer("/error/message")
                .and_then(|m| m.as_str())
                .unwrap_or("sidecar returned ok=false");
            Err(msg.to_string())
        }
    }
}
fn media_paths_for_ids(
    conn: &Connection,
    ids: Option<Vec<i64>>,
    images_only: bool,
) -> Result<Vec<(i64, String)>, String> {
    let mut sql = "SELECT id,path FROM media_items WHERE missing=0".to_string();
    sql.push_str(&sql_path_not_blacklisted_clause());
    if images_only {
        sql.push_str(" AND media_type='image'");
    }
    let mut out = vec![];
    if let Some(ids) = ids {
        for id in ids {
            if let Some(row) = conn
                .query_row(
                    &sql.replace(" WHERE missing=0", " WHERE missing=0 AND id=?1"),
                    params![id],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()
                .map_err(|e| e.to_string())?
            {
                out.push(row)
            }
        }
    } else {
        let mut s = conn.prepare(&sql).map_err(|e| e.to_string())?;
        out = s
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
    }
    Ok(out)
}
#[tauri::command]
fn cluster_faces(
    app: tauri::AppHandle,
    media_ids: Option<Vec<i64>>,
) -> Result<SidecarResult, String> {
    // Kept for compatibility, but intentionally capped so setup cannot accidentally process huge libraries.
    let ids=media_ids.ok_or_else(||"Use Guided Face Setup to process images incrementally instead of clustering the whole library.".to_string())?;
    if ids.len() > 25 {
        return Err("Too many images for one face-clustering call; use Guided Face Setup.".into());
    }
    process_face_paths(app, Some(ids), false, true, 1)
}
fn process_face_paths(
    app: tauri::AppHandle,
    media_ids: Option<Vec<i64>>,
    auto_name_clusters: bool,
    auto_assign_identity: bool,
    face_sidecar_pool_size: usize,
) -> Result<SidecarResult, String> {
    let mut conn = open_db(&app)?;
    let rows = sidecar_media_rows(&app, media_paths_for_ids(&conn, media_ids, true)?)?;
    let paths: Vec<String> = rows.iter().map(|(_, p)| p.clone()).collect();
    let payload = serde_json::json!({"paths":paths}).to_string();
    let res = run_face_sidecar(&payload, face_sidecar_pool_size)?;
    if res.ok {
        let root: serde_json::Value =
            serde_json::from_str(&res.stdout).map_err(|e| format!("invalid sidecar JSON: {e}"))?;
        if let Some(faces) = root.pointer("/data/faces").and_then(|v| v.as_array()) {
            let named_faces = if auto_assign_identity {
                load_named_face_embeddings(&conn)?
            } else {
                vec![]
            };
            let tx = conn.transaction().map_err(|e| e.to_string())?;
            for (mid, _) in &rows {
                tx.execute("DELETE FROM faces WHERE media_item_id=?1", params![mid])
                    .map_err(|e| e.to_string())?;
            }
            for face in faces {
                let path = face
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let Some((mid, _)) = rows.iter().find(|(_, p)| p == path) else {
                    continue;
                };
                let emb: Vec<f32> = serde_json::from_value(
                    face.get("embedding")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null),
                )
                .unwrap_or_default();
                let pid = if auto_name_clusters {
                    let cluster = face.get("cluster_id").and_then(|v| v.as_i64()).unwrap_or(0);
                    Some(get_or_create_person(
                        &tx,
                        &format!("Person {}", cluster + 1),
                    )?)
                } else if emb.is_empty() {
                    None
                } else if auto_assign_identity {
                    best_named_person_for_embedding(
                        &emb,
                        &named_faces,
                        INSIGHTFACE_FACE_MATCH_THRESHOLD,
                    )
                } else {
                    None
                };
                let bbox = face
                    .get("bbox")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let x = bbox.get(0).and_then(|v| v.as_f64()).unwrap_or(0.0);
                let y = bbox.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
                let w = bbox.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
                let h = bbox.get(3).and_then(|v| v.as_f64()).unwrap_or(0.0);
                let conf = face.get("confidence").and_then(|v| v.as_f64());
                tx.execute("INSERT INTO faces(media_item_id,person_id,x,y,width,height,confidence,created_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",params![mid,pid,x,y,w,h,conf,now_unix()]).map_err(|e|e.to_string())?;
                let face_id = tx.last_insert_rowid();
                if !emb.is_empty() {
                    tx.execute("INSERT INTO embeddings(face_id,model,vector,created_at) VALUES(?1,?2,?3,?4)",params![face_id,FACE_EMBEDDING_MODEL,vec_to_blob(&emb),now_unix()]).map_err(|e|e.to_string())?;
                }
            }
            tx.commit().map_err(|e| e.to_string())?;
        }
    }
    Ok(res)
}
#[tauri::command]
fn process_face_setup_image(app: tauri::AppHandle, media_id: i64) -> Result<SidecarResult, String> {
    process_face_paths(app, Some(vec![media_id]), false, true, 1)
}
#[tauri::command]
async fn generate_embeddings(
    app: tauri::AppHandle,
    media_ids: Option<Vec<i64>>,
    provider: Option<String>,
    model: Option<String>,
    image_max_width: Option<u32>,
) -> Result<SidecarResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        generate_embeddings_impl(app, media_ids, provider, model, image_max_width)
    })
    .await
    .map_err(|e| {
        let msg = format!("embedding task failed/thread panicked: {e}");
        backend_log(&msg);
        msg
    })?
}

fn generate_embeddings_impl(
    app: tauri::AppHandle,
    media_ids: Option<Vec<i64>>,
    provider: Option<String>,
    model: Option<String>,
    image_max_width: Option<u32>,
) -> Result<SidecarResult, String> {
    let mut conn = open_db(&app)?;
    let selected_provider = provider.unwrap_or_else(|| "fastembed".into());
    let selected_model = model.unwrap_or_else(|| "Qdrant/clip-ViT-B-32".into());
    let rows = sidecar_media_rows(
        &app,
        media_paths_for_ids(&conn, media_ids, selected_provider == "fastembed")?,
    )?;
    let total = rows.len();
    let mut summary = ScanSummary::default();
    emit_scan_progress(
        &app,
        &summary,
        "Generating embeddings",
        None,
        total,
        Some(total),
        0,
        None,
        false,
    );
    if total == 0 {
        emit_scan_progress(
            &app,
            &summary,
            "Embeddings complete",
            None,
            total,
            Some(total),
            0,
            None,
            true,
        );
        return Ok(SidecarResult {
            ok: true,
            stdout: "{\"ok\":true,\"data\":{\"embedded\":0,\"skipped\":0}}".into(),
            stderr: String::new(),
        });
    }
    let mut inserted = 0usize;
    let mut skipped = 0usize;
    let mut last_res = SidecarResult {
        ok: true,
        stdout: String::new(),
        stderr: String::new(),
    };
    for chunk in rows.chunks(EMBEDDING_BATCH_SIZE) {
        let paths: Vec<String> = chunk.iter().map(|(_, p)| p.clone()).collect();
        let payload = serde_json::json!({"paths":paths});
        let mut args = vec![
            "embed".into(),
            "--provider".into(),
            selected_provider.clone(),
            "--model".into(),
            selected_model.clone(),
            "--workers".into(),
            embedding_thread_count(&selected_provider).to_string(),
            "--batch-size".into(),
            embedding_batch_size(&selected_provider).to_string(),
        ];
        if let Some(width) = image_max_width.filter(|w| *w > 0) {
            args.push("--image-max-width".into());
            args.push(width.to_string());
        }
        let res = run_sidecar_json_payload(args, &payload)?;
        if !res.ok {
            summary.errors.push(res.stderr.clone());
            last_res = res;
            break;
        }
        let root: serde_json::Value =
            serde_json::from_str(&res.stdout).map_err(|e| format!("invalid sidecar JSON: {e}"))?;
        if let Some(embs) = root.pointer("/data/embeddings").and_then(|v| v.as_array()) {
            let path_to_id: HashMap<&str, i64> =
                chunk.iter().map(|(id, p)| (p.as_str(), *id)).collect();
            let tx = conn.transaction().map_err(|e| e.to_string())?;
            for emb in embs {
                let src = emb
                    .get("source")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let Some(mid) = path_to_id.get(src) else {
                    continue;
                };
                let model = emb
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let Some(vecv) = emb
                    .get("embedding")
                    .and_then(|v| v.as_array())
                    .filter(|v| !v.is_empty())
                else {
                    skipped += 1;
                    continue;
                };
                let vector: Vec<f32> =
                    serde_json::from_value(serde_json::Value::Array(vecv.clone()))
                        .map_err(|e| e.to_string())?;
                tx.execute("INSERT INTO embeddings(media_item_id,model,vector,created_at) VALUES(?1,?2,?3,?4)",params![mid,model,vec_to_blob(&vector),now_unix()]).map_err(|e|e.to_string())?;
                inserted += 1;
            }
            tx.commit().map_err(|e| e.to_string())?;
        }
        summary.scanned_files += chunk.len();
        summary.imported_or_updated = inserted;
        summary.skipped_files = skipped + summary.scanned_files.saturating_sub(inserted + skipped);
        emit_scan_progress(
            &app,
            &summary,
            "Generating embeddings",
            chunk.last().map(|(_, p)| p.clone()),
            total,
            Some(total),
            0,
            None,
            false,
        );
        last_res = res;
    }
    summary.imported_or_updated = inserted;
    summary.skipped_files = total.saturating_sub(inserted);
    emit_scan_progress(
        &app,
        &summary,
        "Embeddings complete",
        None,
        total,
        Some(total),
        0,
        None,
        true,
    );
    if last_res.ok {
        last_res.stdout = serde_json::json!({"ok":true,"data":{"embedded":inserted,"skipped":summary.skipped_files,"total":total}}).to_string();
    }
    Ok(last_res)
}
fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * std::mem::size_of::<f32>());
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
fn parse_vec(b: Vec<u8>) -> Option<Vec<f32>> {
    if b.len() % std::mem::size_of::<f32>() == 0 && !b.starts_with(b"[") {
        return Some(
            b.chunks_exact(std::mem::size_of::<f32>())
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
        );
    }
    serde_json::from_slice(&b).ok().or_else(|| {
        std::str::from_utf8(&b)
            .ok()
            .and_then(|s| serde_json::from_str(s).ok())
    })
}
fn cosine_with_norm(a: &[f32], a_norm: f32, b: &[f32]) -> f64 {
    let (mut dot, mut nb) = (0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += *x * *y;
        nb += *y * *y;
    }
    if a_norm == 0.0 || nb == 0.0 {
        0.0
    } else {
        (dot / (a_norm * nb.sqrt())) as f64
    }
}
fn score_vec_compacting(
    query: &[f32],
    query_norm: f32,
    b: &[u8],
) -> Option<(f64, Option<Vec<u8>>)> {
    if b.len() % std::mem::size_of::<f32>() == 0 && !b.starts_with(b"[") {
        if b.len() / std::mem::size_of::<f32>() != query.len() {
            return None;
        }
        let (mut dot, mut nb) = (0.0f32, 0.0f32);
        for (x, chunk) in query.iter().zip(b.chunks_exact(std::mem::size_of::<f32>())) {
            let y = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            dot += *x * y;
            nb += y * y;
        }
        let score = if query_norm == 0.0 || nb == 0.0 {
            0.0
        } else {
            (dot / (query_norm * nb.sqrt())) as f64
        };
        return Some((score, None));
    }
    let vector: Vec<f32> = serde_json::from_slice(b).ok().or_else(|| {
        std::str::from_utf8(b)
            .ok()
            .and_then(|s| serde_json::from_str(s).ok())
    })?;
    if vector.len() != query.len() {
        return None;
    }
    let score = cosine_with_norm(query, query_norm, &vector);
    let compacted = vec_to_blob(&vector);
    Some((score, Some(compacted)))
}
fn vector_norm(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    cosine_with_norm(a, vector_norm(a), b)
}

#[derive(Debug)]
struct SemanticCandidate {
    score: f64,
    media_id: i64,
}

impl PartialEq for SemanticCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.media_id == other.media_id
    }
}
impl Eq for SemanticCandidate {}
impl PartialOrd for SemanticCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        self.score.partial_cmp(&other.score)
    }
}
impl Ord for SemanticCandidate {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.partial_cmp(other).unwrap_or(CmpOrdering::Equal)
    }
}
fn search_semantic_impl(
    app: tauri::AppHandle,
    vector: Vec<f32>,
    model: Option<String>,
    limit: Option<i64>,
) -> Result<Vec<SemanticHit>, String> {
    let mut c = open_db(&app)?;
    let mut semantic_sql = format!(
        "SELECT e.id,e.media_item_id,e.vector FROM embeddings e JOIN media_items ON media_items.id=e.media_item_id WHERE e.media_item_id IS NOT NULL{}",
        sql_path_not_blacklisted_clause()
    );
    let mut params_vec: Vec<Box<dyn ToSql>> = vec![];
    if let Some(model) = model {
        semantic_sql.push_str(" AND e.model=?");
        params_vec.push(Box::new(model));
    }
    let max_hits = limit.unwrap_or(50).clamp(1, 200) as usize;
    let query_norm = vector_norm(&vector);
    let mut candidates: Vec<SemanticCandidate> = Vec::with_capacity(max_hits + 1);
    let mut compacted_vectors: Vec<(i64, Vec<u8>)> = vec![];
    {
        let refs: Vec<&dyn ToSql> = params_vec.iter().map(|x| x.as_ref()).collect();
        let mut s = c.prepare(&semantic_sql).map_err(|e| e.to_string())?;
        let rows = s
            .query_map(params_from_iter(refs), |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                ))
            })
            .map_err(|e| e.to_string())?;
        for r in rows {
            let (embedding_id, media_id, b) = r.map_err(|e| e.to_string())?;
            if let Some((score, compacted)) = score_vec_compacting(&vector, query_norm, &b) {
                if let Some(compacted) = compacted {
                    compacted_vectors.push((embedding_id, compacted));
                }
                if candidates.len() < max_hits {
                    candidates.push(SemanticCandidate { score, media_id });
                    if candidates.len() == max_hits {
                        candidates.sort_by(|a, b| {
                            a.score
                                .partial_cmp(&b.score)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                } else if score > candidates[0].score {
                    candidates[0] = SemanticCandidate { score, media_id };
                    candidates.sort_by(|a, b| {
                        a.score
                            .partial_cmp(&b.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
        }
    }
    if !compacted_vectors.is_empty() {
        let mut conn = c;
        let tx = conn.transaction().map_err(|e| e.to_string())?;
        for (embedding_id, compacted) in compacted_vectors {
            tx.execute(
                "UPDATE embeddings SET vector=?1 WHERE id=?2",
                params![compacted, embedding_id],
            )
            .map_err(|e| e.to_string())?;
        }
        tx.commit().map_err(|e| e.to_string())?;
        c = conn;
    }
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let ids: Vec<i64> = candidates.iter().map(|c| c.media_id).collect();
    if ids.is_empty() {
        return Ok(vec![]);
    }
    let placeholders = std::iter::repeat("?")
        .take(ids.len())
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!("{MEDIA_SELECT} WHERE id IN ({placeholders})");
    let mut stmt = c.prepare(&sql).map_err(|e| e.to_string())?;
    let media_rows = stmt
        .query_map(params_from_iter(ids.iter()), row_to_media)
        .map_err(|e| e.to_string())?;
    let mut media_by_id = HashMap::with_capacity(ids.len());
    for row in media_rows {
        let item = row.map_err(|e| e.to_string())?;
        media_by_id.insert(item.id, item);
    }
    let mut hits: Vec<SemanticHit> = candidates
        .into_iter()
        .filter_map(|candidate| {
            media_by_id
                .get(&candidate.media_id)
                .cloned()
                .map(|item| SemanticHit {
                    score: candidate.score,
                    item,
                })
        })
        .collect();
    let mut items: Vec<MediaItem> = hits.iter().map(|hit| hit.item.clone()).collect();
    enrich_display_paths(&app, &mut items)?;
    for (hit, item) in hits.iter_mut().zip(items.into_iter()) {
        hit.item = item;
    }
    Ok(hits)
}
#[tauri::command]
fn search_semantic(
    app: tauri::AppHandle,
    vector: Vec<f32>,
    limit: Option<i64>,
) -> Result<Vec<SemanticHit>, String> {
    search_semantic_impl(app, vector, None, limit)
}
#[tauri::command]
fn search_semantic_text(
    app: tauri::AppHandle,
    query: String,
    provider: Option<String>,
    model: Option<String>,
    limit: Option<i64>,
) -> Result<Vec<SemanticHit>, String> {
    let selected_model = model.unwrap_or_else(|| "Qdrant/clip-ViT-B-32".into());
    let c = open_db(&app)?;
    let n: i64 = c
        .query_row(
            "SELECT COUNT(*) FROM embeddings WHERE media_item_id IS NOT NULL AND model=?1",
            params![selected_model],
            |r| r.get(0),
        )
        .map_err(|e| e.to_string())?;
    if n == 0 {
        return Ok(vec![]);
    }
    drop(c);
    let args = vec![
        "embed".into(),
        "--provider".into(),
        provider.unwrap_or_else(|| "fastembed".into()),
        "--model".into(),
        selected_model.clone(),
        "--workers".into(),
        indexing_thread_count().to_string(),
        "--text".into(),
        query,
    ];
    let res = run_sidecar(args)?;
    if !res.ok {
        return Err(res.stderr);
    }
    let root: serde_json::Value =
        serde_json::from_str(&res.stdout).map_err(|e| format!("invalid sidecar JSON: {e}"))?;
    let vecv = root
        .pointer("/data/embeddings/0/embedding")
        .ok_or_else(|| "sidecar did not return a query embedding".to_string())?;
    let vector: Vec<f32> = serde_json::from_value(vecv.clone()).map_err(|e| e.to_string())?;
    search_semantic_impl(app, vector, Some(selected_model), limit)
}
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Welcome to {name}.")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(
        path: &str,
        captured_at: Option<i64>,
        modified_at: Option<i64>,
        pixels: i64,
    ) -> CleanupCandidate {
        CleanupCandidate {
            path: path.into(),
            display_path: None,
            file_name: path.into(),
            size_bytes: Some(1),
            created_at: None,
            modified_at,
            captured_at,
            width: Some(pixels),
            height: Some(1),
        }
    }

    #[test]
    fn cleanup_keeper_prefers_earliest_date_before_resolution() {
        let candidates = vec![
            candidate("large-later.jpg", Some(20), None, 4000),
            candidate("small-earlier.jpg", Some(10), None, 1000),
        ];
        assert_eq!(
            choose_cleanup_keeper(&candidates).as_deref(),
            Some("small-earlier.jpg")
        );
    }

    #[test]
    fn cleanup_keeper_uses_resolution_as_date_tiebreaker() {
        let candidates = vec![
            candidate("small.jpg", Some(10), None, 1000),
            candidate("large.jpg", Some(10), None, 4000),
        ];
        assert_eq!(
            choose_cleanup_keeper(&candidates).as_deref(),
            Some("large.jpg")
        );
    }

    #[test]
    fn hex_hamming_counts_different_bits() {
        assert_eq!(hex_hamming("00", "0f"), Some(4));
        assert_eq!(hex_hamming("ff", "00"), Some(8));
    }
}
pub fn run() {
    tauri::Builder::default()
        .plugin(
            tauri_plugin_window_state::Builder::default()
                .with_state_flags(StateFlags::POSITION | StateFlags::SIZE)
                .build(),
        )
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            greet,
            initialize_app,
            delete_current_index,
            choose_media_folder,
            get_settings,
            update_settings,
            list_library_folders,
            add_library_folder,
            remove_library_folder,
            scan_library,
            update_face_embeddings,
            search_media,
            get_media_item,
            mark_missing,
            rescan,
            list_people,
            rename_person,
            delete_person,
            name_face,
            list_faces,
            cluster_faces,
            process_face_setup_image,
            generate_embeddings,
            search_semantic,
            search_semantic_text,
            list_geo_points,
            generate_cleanup_plan,
            apply_cleanup_plan,
            cancel_cleanup_plan,
            get_cleanup_cache_info,
            clear_cleanup_cache
        ])
        .run(tauri::generate_context!())
        .expect("error while running Tauri application");
}
