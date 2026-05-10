use rusqlite::{params, params_from_iter, Connection, OptionalExtension, ToSql};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
#[cfg(not(debug_assertions))]
use tauri::Manager;
use walkdir::WalkDir;

const DB_FILE: &str = "rich-media-viewer.sqlite3";

#[derive(Debug, Serialize)]
struct AppInfo {
    data_dir: String,
    database_path: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct MediaItem {
    id: i64,
    path: String,
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
    latitude: Option<f64>,
    longitude: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct SearchFilter {
    query: Option<String>,
    media_type: Option<String>,
    missing: Option<bool>,
    limit: Option<i64>,
    offset: Option<i64>,
}

#[derive(Debug, Serialize)]
struct ScanSummary {
    scanned_files: usize,
    imported_or_updated: usize,
    skipped_files: usize,
    missing_marked: usize,
    errors: Vec<String>,
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

fn open_db(app: &tauri::AppHandle) -> Result<Connection, String> {
    let dir = app_data_dir(app)?;
    fs::create_dir_all(&dir).map_err(|e| format!("failed to create app data dir: {e}"))?;
    let conn = Connection::open(dir.join(DB_FILE)).map_err(|e| format!("failed to open database: {e}"))?;
    init_db(&conn)?;
    Ok(conn)
}

fn init_db(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        r#"
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS media_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            file_name TEXT NOT NULL,
            extension TEXT,
            media_type TEXT NOT NULL,
            size_bytes INTEGER,
            created_at INTEGER,
            modified_at INTEGER,
            imported_at INTEGER NOT NULL,
            missing INTEGER NOT NULL DEFAULT 0,
            camera_make TEXT,
            camera_model TEXT,
            latitude REAL,
            longitude REAL
        );
        CREATE INDEX IF NOT EXISTS idx_media_items_path ON media_items(path);
        CREATE INDEX IF NOT EXISTS idx_media_items_type ON media_items(media_type);

        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_item_id INTEGER NOT NULL,
            person_id INTEGER,
            x REAL NOT NULL,
            y REAL NOT NULL,
            width REAL NOT NULL,
            height REAL NOT NULL,
            confidence REAL,
            created_at INTEGER NOT NULL,
            FOREIGN KEY(media_item_id) REFERENCES media_items(id) ON DELETE CASCADE,
            FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_item_id INTEGER,
            face_id INTEGER,
            model TEXT NOT NULL,
            vector BLOB NOT NULL,
            created_at INTEGER NOT NULL,
            FOREIGN KEY(media_item_id) REFERENCES media_items(id) ON DELETE CASCADE,
            FOREIGN KEY(face_id) REFERENCES faces(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        );
        "#,
    )
    .map_err(|e| format!("failed to initialize database: {e}"))
}

fn unix_time(t: SystemTime) -> Option<i64> {
    t.duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs() as i64)
}

fn now_unix() -> i64 {
    unix_time(SystemTime::now()).unwrap_or(0)
}

fn media_type_for_ext(ext: Option<&str>) -> Option<&'static str> {
    match ext.unwrap_or_default().to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" | "png" | "gif" | "webp" | "bmp" | "tif" | "tiff" | "heic" | "heif" => Some("image"),
        "mp4" | "mov" | "m4v" | "avi" | "mkv" | "webm" => Some("video"),
        _ => None,
    }
}

fn media_from_path(path: &Path) -> Result<Option<MediaItem>, String> {
    let ext = path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase());
    let Some(media_type) = media_type_for_ext(ext.as_deref()) else { return Ok(None); };
    let md = fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !md.is_file() { return Ok(None); }
    Ok(Some(MediaItem {
        id: 0,
        path: path.to_string_lossy().to_string(),
        file_name: path.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_string(),
        extension: ext,
        media_type: media_type.to_string(),
        size_bytes: Some(md.len() as i64),
        created_at: md.created().ok().and_then(unix_time),
        modified_at: md.modified().ok().and_then(unix_time),
        imported_at: now_unix(),
        missing: false,
        // TODO: parse EXIF/XMP for camera and GPS metadata.
        camera_make: None,
        camera_model: None,
        latitude: None,
        longitude: None,
    }))
}

fn upsert_media(conn: &Connection, item: &MediaItem) -> Result<(), String> {
    conn.execute(
        r#"INSERT INTO media_items
        (path, file_name, extension, media_type, size_bytes, created_at, modified_at, imported_at, missing, camera_make, camera_model, latitude, longitude)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0, ?9, ?10, ?11, ?12)
        ON CONFLICT(path) DO UPDATE SET
          file_name=excluded.file_name, extension=excluded.extension, media_type=excluded.media_type,
          size_bytes=excluded.size_bytes, created_at=excluded.created_at, modified_at=excluded.modified_at,
          missing=0, camera_make=excluded.camera_make, camera_model=excluded.camera_model,
          latitude=excluded.latitude, longitude=excluded.longitude"#,
        params![item.path, item.file_name, item.extension, item.media_type, item.size_bytes, item.created_at, item.modified_at, item.imported_at, item.camera_make, item.camera_model, item.latitude, item.longitude],
    ).map_err(|e| format!("failed to upsert media item: {e}"))?;
    Ok(())
}

fn row_to_media(row: &rusqlite::Row<'_>) -> rusqlite::Result<MediaItem> {
    Ok(MediaItem {
        id: row.get(0)?, path: row.get(1)?, file_name: row.get(2)?, extension: row.get(3)?, media_type: row.get(4)?,
        size_bytes: row.get(5)?, created_at: row.get(6)?, modified_at: row.get(7)?, imported_at: row.get(8)?,
        missing: row.get::<_, i64>(9)? != 0, camera_make: row.get(10)?, camera_model: row.get(11)?, latitude: row.get(12)?, longitude: row.get(13)?,
    })
}

#[tauri::command]
fn initialize_app(app: tauri::AppHandle) -> Result<AppInfo, String> {
    let conn = open_db(&app)?;
    drop(conn);
    Ok(AppInfo { data_dir: app_data_dir(&app)?.to_string_lossy().to_string(), database_path: db_path(&app)?.to_string_lossy().to_string() })
}

#[tauri::command]
fn choose_media_folder(path: Option<String>) -> Result<Option<String>, String> {
    // Placeholder for a future native folder dialog/plugin. Frontend may pass a selected path for now.
    Ok(path.filter(|p| Path::new(p).is_dir()))
}

#[tauri::command]
fn scan_library(app: tauri::AppHandle, paths: Vec<String>) -> Result<ScanSummary, String> {
    let conn = open_db(&app)?;
    let mut summary = ScanSummary { scanned_files: 0, imported_or_updated: 0, skipped_files: 0, missing_marked: 0, errors: vec![] };
    for root in paths {
        for entry in WalkDir::new(&root).follow_links(false).into_iter().filter_map(Result::ok) {
            if !entry.file_type().is_file() { continue; }
            summary.scanned_files += 1;
            match media_from_path(entry.path()).and_then(|opt| { if let Some(item) = opt { upsert_media(&conn, &item).map(|_| true) } else { Ok(false) } }) {
                Ok(true) => summary.imported_or_updated += 1,
                Ok(false) => summary.skipped_files += 1,
                Err(e) => summary.errors.push(e),
            }
        }
    }
    summary.missing_marked = mark_missing_internal(&conn)?;
    Ok(summary)
}

fn mark_missing_internal(conn: &Connection) -> Result<usize, String> {
    let mut stmt = conn.prepare("SELECT id, path FROM media_items WHERE missing = 0").map_err(|e| e.to_string())?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))).map_err(|e| e.to_string())?;
    let mut count = 0;
    for row in rows {
        let (id, path) = row.map_err(|e| e.to_string())?;
        if !Path::new(&path).exists() {
            conn.execute("UPDATE media_items SET missing = 1 WHERE id = ?1", params![id]).map_err(|e| e.to_string())?;
            count += 1;
        }
    }
    Ok(count)
}

#[tauri::command]
fn mark_missing(app: tauri::AppHandle) -> Result<usize, String> {
    let conn = open_db(&app)?;
    mark_missing_internal(&conn)
}

#[tauri::command]
fn rescan(app: tauri::AppHandle) -> Result<ScanSummary, String> {
    let conn = open_db(&app)?;
    let mut stmt = conn.prepare("SELECT DISTINCT path FROM media_items").map_err(|e| e.to_string())?;
    let paths: Vec<String> = stmt.query_map([], |r| r.get::<_, String>(0)).map_err(|e| e.to_string())?.filter_map(Result::ok).collect();
    drop(stmt);
    scan_library(app, paths)
}

#[tauri::command]
fn get_media_item(app: tauri::AppHandle, id: i64) -> Result<Option<MediaItem>, String> {
    let conn = open_db(&app)?;
    conn.query_row("SELECT id,path,file_name,extension,media_type,size_bytes,created_at,modified_at,imported_at,missing,camera_make,camera_model,latitude,longitude FROM media_items WHERE id=?1", params![id], row_to_media)
        .optional().map_err(|e| e.to_string())
}

#[tauri::command]
fn search_media(app: tauri::AppHandle, filter: SearchFilter) -> Result<Vec<MediaItem>, String> {
    let conn = open_db(&app)?;
    let mut sql = "SELECT id,path,file_name,extension,media_type,size_bytes,created_at,modified_at,imported_at,missing,camera_make,camera_model,latitude,longitude FROM media_items WHERE 1=1".to_string();
    let mut values: Vec<Box<dyn ToSql>> = Vec::new();
    if let Some(q) = filter.query.filter(|q| !q.trim().is_empty()) {
        sql.push_str(" AND (file_name LIKE ? OR path LIKE ?)");
        let like = format!("%{}%", q.trim());
        values.push(Box::new(like.clone())); values.push(Box::new(like));
    }
    if let Some(mt) = filter.media_type { sql.push_str(" AND media_type = ?"); values.push(Box::new(mt)); }
    if let Some(missing) = filter.missing { sql.push_str(" AND missing = ?"); values.push(Box::new(if missing { 1_i64 } else { 0_i64 })); }
    sql.push_str(" ORDER BY modified_at DESC, id DESC LIMIT ? OFFSET ?");
    values.push(Box::new(filter.limit.unwrap_or(100).clamp(1, 500)));
    values.push(Box::new(filter.offset.unwrap_or(0).max(0)));
    let refs: Vec<&dyn ToSql> = values.iter().map(|v| v.as_ref()).collect();
    let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
    let items = stmt.query_map(params_from_iter(refs), row_to_media).map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())?;
    Ok(items)
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Welcome to {name}.")
}

pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            greet,
            initialize_app,
            choose_media_folder,
            scan_library,
            search_media,
            get_media_item,
            mark_missing,
            rescan
        ])
        .run(tauri::generate_context!())
        .expect("error while running Tauri application");
}
