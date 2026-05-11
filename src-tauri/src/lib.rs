use rayon::prelude::*;
use rusqlite::{params, params_from_iter, Connection, OptionalExtension, ToSql};
use serde::{Deserialize, Serialize};
use std::{collections::{HashSet, VecDeque}, fs, path::{Path, PathBuf}, process::Command, sync::{atomic::{AtomicUsize, Ordering}, mpsc, Arc, Mutex}, thread, time::{Duration, SystemTime, UNIX_EPOCH}};
use tauri::Emitter;
#[cfg(not(debug_assertions))]
use tauri::Manager;
use tauri_plugin_dialog::DialogExt;

const DB_FILE: &str = "rich-media-viewer.sqlite3";
const DISCOVERY_THREAD_MULTIPLIER: usize = 4;
const INDEXING_THREAD_DIVISOR: usize = 2;

fn backend_log(message:&str){eprintln!("[rich-media-viewer backend] {message}");}
fn available_threads()->usize{thread::available_parallelism().map(|n|n.get()).unwrap_or(4)}
fn discovery_thread_count()->usize{std::env::var("RMV_DISCOVERY_THREADS").ok().and_then(|v|v.parse().ok()).unwrap_or_else(||available_threads().saturating_mul(DISCOVERY_THREAD_MULTIPLIER)).clamp(8,128)}
fn indexing_thread_count()->usize{std::env::var("RMV_INDEXING_THREADS").ok().and_then(|v|v.parse().ok()).unwrap_or_else(||(available_threads()/INDEXING_THREAD_DIVISOR).max(1)).clamp(1,4)}

const BLACKLISTED_FOLDER_NAMES: &[&str]=&[
    "thumbnails", "thumbnail", "thumbs", ".thumbnails",
    "ipod photo cache", "apple tv photo cache", "photo cache",
    "previews", "preview", "derivatives", "renders", "proxies",
    "database", "contents", "backup", "private", "cpl", "cloudsharing", "journals",
    "__macosx", ".spotlight-v100", ".temporaryitems", ".trashes",
];
fn is_blacklisted_path(path:&Path)->bool{path.components().any(|c|c.as_os_str().to_str().map(|s|BLACKLISTED_FOLDER_NAMES.iter().any(|b|s.eq_ignore_ascii_case(b))).unwrap_or(false))}
fn is_dotfile_path(path:&Path)->bool{path.components().any(|c|c.as_os_str().to_str().map(|s|s.starts_with('.')).unwrap_or(false))}
fn sql_path_not_blacklisted_clause()->String{let mut clause:String=BLACKLISTED_FOLDER_NAMES.iter().map(|name|format!(" AND lower(path) NOT LIKE '%{}%'",name.replace("'","''"))).collect();clause.push_str(" AND file_name NOT LIKE '.%'");clause}

#[derive(Debug, Serialize)] struct AppInfo { data_dir: String, database_path: String }
#[derive(Debug, Serialize, Deserialize, Clone)]
struct MediaItem { id:i64,path:String,file_name:String,extension:Option<String>,media_type:String,size_bytes:Option<i64>,created_at:Option<i64>,modified_at:Option<i64>,imported_at:i64,missing:bool,camera_make:Option<String>,camera_model:Option<String>,lens_model:Option<String>,captured_at:Option<i64>,latitude:Option<f64>,longitude:Option<f64>,metadata_json:Option<String> }
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
struct AppSettings { library_folders: Vec<String> }
#[derive(Debug, Deserialize, Default)]
struct SearchFilter { query:Option<String>,media_type:Option<String>,missing:Option<bool>,from_ts:Option<i64>,to_ts:Option<i64>,camera:Option<String>,lat:Option<f64>,lng:Option<f64>,radius_km:Option<f64>,person_id:Option<i64>,person_name:Option<String>,has_gps:Option<bool>,has_camera:Option<bool>,limit:Option<i64>,offset:Option<i64> }
#[derive(Debug, Serialize)] struct ScanSummary { scanned_files:usize, imported_or_updated:usize, skipped_files:usize, missing_marked:usize, errors:Vec<String> }
#[derive(Debug, Serialize, Clone)] struct ScanProgress { phase:String, current_path:Option<String>, scanned_files:usize, imported_or_updated:usize, skipped_files:usize, missing_marked:usize, errors:usize, discovered_files:usize, total_files:Option<usize>, done:bool }
#[derive(Debug, Serialize)] struct Person { id:i64, name:String, created_at:i64, face_count:i64 }
#[derive(Debug, Serialize)] struct Face { id:i64, media_item_id:i64, person_id:Option<i64>, person_name:Option<String>, x:f64,y:f64,width:f64,height:f64,confidence:Option<f64>,created_at:i64 }
#[derive(Debug, Serialize)] struct SidecarResult { ok: bool, stdout: String, stderr: String }
#[derive(Debug, Serialize)] struct SemanticHit { item: MediaItem, score: f64 }

fn app_data_dir(_app:&tauri::AppHandle)->Result<PathBuf,String>{ #[cfg(debug_assertions)] { std::env::current_dir().map(|p|p.join("dev-data")).map_err(|e|format!("failed to resolve current dir: {e}")) } #[cfg(not(debug_assertions))] { _app.path().app_data_dir().map_err(|e|format!("failed to resolve app data dir: {e}")) } }
fn db_path(app:&tauri::AppHandle)->Result<PathBuf,String>{Ok(app_data_dir(app)?.join(DB_FILE))}
fn open_db(app:&tauri::AppHandle)->Result<Connection,String>{let dir=app_data_dir(app)?;fs::create_dir_all(&dir).map_err(|e|format!("failed to create app data dir: {e}"))?;let conn=Connection::open(dir.join(DB_FILE)).map_err(|e|format!("failed to open database: {e}"))?;init_db(&conn)?;Ok(conn)}
fn init_db(conn:&Connection)->Result<(),String>{conn.execute_batch(r#"
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS media_items(id INTEGER PRIMARY KEY AUTOINCREMENT,path TEXT NOT NULL UNIQUE,file_name TEXT NOT NULL,extension TEXT,media_type TEXT NOT NULL,size_bytes INTEGER,created_at INTEGER,modified_at INTEGER,imported_at INTEGER NOT NULL,missing INTEGER NOT NULL DEFAULT 0,camera_make TEXT,camera_model TEXT,latitude REAL,longitude REAL);
CREATE INDEX IF NOT EXISTS idx_media_items_path ON media_items(path); CREATE INDEX IF NOT EXISTS idx_media_items_type ON media_items(media_type);
CREATE TABLE IF NOT EXISTS people(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL,created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS faces(id INTEGER PRIMARY KEY AUTOINCREMENT,media_item_id INTEGER NOT NULL,person_id INTEGER,x REAL NOT NULL,y REAL NOT NULL,width REAL NOT NULL,height REAL NOT NULL,confidence REAL,created_at INTEGER NOT NULL,FOREIGN KEY(media_item_id) REFERENCES media_items(id) ON DELETE CASCADE,FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE SET NULL);
CREATE TABLE IF NOT EXISTS embeddings(id INTEGER PRIMARY KEY AUTOINCREMENT,media_item_id INTEGER,face_id INTEGER,model TEXT NOT NULL,vector BLOB NOT NULL,created_at INTEGER NOT NULL,FOREIGN KEY(media_item_id) REFERENCES media_items(id) ON DELETE CASCADE,FOREIGN KEY(face_id) REFERENCES faces(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL,updated_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS library_folders(id INTEGER PRIMARY KEY AUTOINCREMENT,path TEXT NOT NULL UNIQUE,created_at INTEGER NOT NULL);
"#).map_err(|e|format!("failed to initialize database: {e}"))?;
for col in ["lens_model TEXT","captured_at INTEGER","metadata_json TEXT"] { let _=conn.execute(&format!("ALTER TABLE media_items ADD COLUMN {col}"),[]); }
let _=conn.execute("CREATE INDEX IF NOT EXISTS idx_media_items_captured ON media_items(captured_at)",[]); Ok(())}
fn unix_time(t:SystemTime)->Option<i64>{t.duration_since(UNIX_EPOCH).ok().map(|d|d.as_secs() as i64)} fn now_unix()->i64{unix_time(SystemTime::now()).unwrap_or(0)}
fn clean_path_str(s:&str)->String{s.strip_prefix("\\\\?\\").unwrap_or(s).to_string()}
fn clean_path_string(path:&Path)->String{clean_path_str(&path.to_string_lossy())}
fn media_type_for_ext(ext:Option<&str>)->Option<&'static str>{match ext.unwrap_or_default().trim_start_matches('.').to_ascii_lowercase().as_str(){"jpg"|"jpeg"|"jpe"|"jfif"|"png"|"gif"|"webp"|"bmp"|"dib"|"tif"|"tiff"|"heic"|"heif"|"avif"|"svg"|"ico"|"raw"|"arw"|"cr2"|"cr3"|"nef"|"nrw"|"orf"|"rw2"|"raf"|"dng"|"pef"|"srw"=>Some("image"),"mp4"|"mov"|"m4v"|"avi"|"mkv"|"webm"|"wmv"|"mpg"|"mpeg"|"3gp"|"3g2"|"mts"|"m2ts"|"ts"=>Some("video"),_=>None}}
fn parse_exif(path:&Path)->(Option<String>,Option<String>,Option<String>,Option<i64>,Option<f64>,Option<f64>,Option<String>){let Some(file)=fs::File::open(path).ok() else { return (None,None,None,None,None,None,None) };let mut br=std::io::BufReader::new(file);let Some(exif)=exif::Reader::new().read_from_container(&mut br).ok() else { return (None,None,None,None,None,None,None) };use exif::{In,Tag,Value};let get=|tag| exif.get_field(tag,In::PRIMARY).map(|f|f.display_value().with_unit(&exif).to_string());let make=get(Tag::Make);let model=get(Tag::Model);let lens=get(Tag::LensModel);let dt=get(Tag::DateTimeOriginal).and_then(|s| chrono::NaiveDateTime::parse_from_str(&s,"%Y-%m-%d %H:%M:%S").or_else(|_|chrono::NaiveDateTime::parse_from_str(&s,"%Y:%m:%d %H:%M:%S")).ok()).map(|d|d.and_utc().timestamp());let gps=|tag| exif.get_field(tag,In::PRIMARY).and_then(|f|if let Value::Rational(v)=&f.value{ if v.len()>=3{Some(v[0].to_f64()+v[1].to_f64()/60.0+v[2].to_f64()/3600.0)}else{None}}else{None});let mut lat=gps(Tag::GPSLatitude);let mut lng=gps(Tag::GPSLongitude);if get(Tag::GPSLatitudeRef).unwrap_or_default().contains('S'){lat=lat.map(|v|-v)}; if get(Tag::GPSLongitudeRef).unwrap_or_default().contains('W'){lng=lng.map(|v|-v)};let json=serde_json::to_string(&serde_json::json!({"make":make,"model":model,"lens_model":lens,"captured_at":dt,"latitude":lat,"longitude":lng})).ok();(make,model,lens,dt,lat,lng,json)}
fn media_from_path(path:&Path)->Result<Option<MediaItem>,String>{if is_blacklisted_path(path)||is_dotfile_path(path){return Ok(None)}let ext=path.extension().and_then(|e|e.to_str()).map(|s|s.to_ascii_lowercase());let Some(mt)=media_type_for_ext(ext.as_deref()) else{return Ok(None)};let md=fs::metadata(path).map_err(|e|format!("{}: {e}",path.display()))?;if !md.is_file(){return Ok(None)};let (mk,mo,lens,cap,lat,lng,mjson)=if mt=="image"{parse_exif(path)}else{(None,None,None,None,None,None,None)};Ok(Some(MediaItem{id:0,path:clean_path_string(path),file_name:path.file_name().and_then(|n|n.to_str()).unwrap_or_default().to_string(),extension:ext,media_type:mt.to_string(),size_bytes:Some(md.len() as i64),created_at:md.created().ok().and_then(unix_time),modified_at:md.modified().ok().and_then(unix_time),imported_at:now_unix(),missing:false,camera_make:mk,camera_model:mo,lens_model:lens,captured_at:cap,latitude:lat,longitude:lng,metadata_json:mjson}))}
fn upsert_media(conn:&Connection,item:&MediaItem)->Result<(),String>{conn.execute(r#"INSERT INTO media_items(path,file_name,extension,media_type,size_bytes,created_at,modified_at,imported_at,missing,camera_make,camera_model,lens_model,captured_at,latitude,longitude,metadata_json) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,0,?9,?10,?11,?12,?13,?14,?15) ON CONFLICT(path) DO UPDATE SET file_name=excluded.file_name,extension=excluded.extension,media_type=excluded.media_type,size_bytes=excluded.size_bytes,created_at=excluded.created_at,modified_at=excluded.modified_at,missing=0,camera_make=excluded.camera_make,camera_model=excluded.camera_model,lens_model=excluded.lens_model,captured_at=excluded.captured_at,latitude=excluded.latitude,longitude=excluded.longitude,metadata_json=excluded.metadata_json"#,params![item.path,item.file_name,item.extension,item.media_type,item.size_bytes,item.created_at,item.modified_at,item.imported_at,item.camera_make,item.camera_model,item.lens_model,item.captured_at,item.latitude,item.longitude,item.metadata_json]).map_err(|e|format!("failed to upsert media item: {e}"))?;Ok(())}
const MEDIA_SELECT:&str="SELECT id,path,file_name,extension,media_type,size_bytes,created_at,modified_at,imported_at,missing,camera_make,camera_model,lens_model,captured_at,latitude,longitude,metadata_json FROM media_items";
fn row_to_media(row:&rusqlite::Row<'_>)->rusqlite::Result<MediaItem>{Ok(MediaItem{id:row.get(0)?,path:row.get(1)?,file_name:row.get(2)?,extension:row.get(3)?,media_type:row.get(4)?,size_bytes:row.get(5)?,created_at:row.get(6)?,modified_at:row.get(7)?,imported_at:row.get(8)?,missing:row.get::<_,i64>(9)?!=0,camera_make:row.get(10)?,camera_model:row.get(11)?,lens_model:row.get(12)?,captured_at:row.get(13)?,latitude:row.get(14)?,longitude:row.get(15)?,metadata_json:row.get(16)?})}
#[tauri::command] fn initialize_app(app:tauri::AppHandle)->Result<AppInfo,String>{let c=open_db(&app)?;drop(c);Ok(AppInfo{data_dir:app_data_dir(&app)?.to_string_lossy().to_string(),database_path:db_path(&app)?.to_string_lossy().to_string()})}
#[tauri::command] fn delete_current_index(app:tauri::AppHandle)->Result<AppInfo,String>{let path=db_path(&app)?;if path.exists(){fs::remove_file(&path).map_err(|e|format!("failed to delete database index {}: {e}",path.display()))?;}let c=open_db(&app)?;drop(c);Ok(AppInfo{data_dir:app_data_dir(&app)?.to_string_lossy().to_string(),database_path:path.to_string_lossy().to_string()})}
#[tauri::command] fn get_settings(app:tauri::AppHandle)->Result<AppSettings,String>{Ok(AppSettings{library_folders:list_library_folders(app)?})}
#[tauri::command] fn update_settings(app:tauri::AppHandle,settings:AppSettings)->Result<AppSettings,String>{let c=open_db(&app)?;c.execute("DELETE FROM library_folders",[]).map_err(|e|e.to_string())?;for p in &settings.library_folders{c.execute("INSERT OR IGNORE INTO library_folders(path,created_at) VALUES(?1,?2)",params![p,now_unix()]).map_err(|e|e.to_string())?;}Ok(settings)}
#[tauri::command] fn list_library_folders(app:tauri::AppHandle)->Result<Vec<String>,String>{let c=open_db(&app)?;let mut s=c.prepare("SELECT path FROM library_folders ORDER BY path").map_err(|e|e.to_string())?;let res=s.query_map([],|r|r.get::<_,String>(0)).map_err(|e|e.to_string())?.collect::<Result<Vec<_>,_>>().map_err(|e|e.to_string())?;Ok(res.into_iter().map(|p|clean_path_str(&p)).collect())}
#[tauri::command] fn add_library_folder(app:tauri::AppHandle,path:String)->Result<Vec<String>,String>{let c=open_db(&app)?;let root=Path::new(&path);if !root.is_dir(){return Err("not a directory".into())}let saved=clean_path_string(&root.canonicalize().unwrap_or_else(|_|root.to_path_buf()));c.execute("INSERT OR IGNORE INTO library_folders(path,created_at) VALUES(?1,?2)",params![saved,now_unix()]).map_err(|e|e.to_string())?;drop(c);list_library_folders(app)}
#[tauri::command] fn remove_library_folder(app:tauri::AppHandle,path:String)->Result<Vec<String>,String>{let c=open_db(&app)?;c.execute("DELETE FROM library_folders WHERE path=?1",params![path]).map_err(|e|e.to_string())?;drop(c);list_library_folders(app)}
#[tauri::command] async fn choose_media_folder(app:tauri::AppHandle,path:Option<String>)->Result<Option<String>,String>{if let Some(p)=path{return Ok(Path::new(&p).is_dir().then_some(clean_path_str(&p)))} ; Ok(app.dialog().file().blocking_pick_folder().map(|p|clean_path_str(&p.to_string())))}
fn discover_files(root:PathBuf,threads:usize,max_files:Option<usize>)->(Vec<PathBuf>,Vec<String>){
    backend_log(&format!("starting discovery root={} threads={threads} max_files={max_files:?}",root.display()));
    let queue=Arc::new(Mutex::new(VecDeque::from([root.clone()])));
    let files=Arc::new(Mutex::new(Vec::new()));
    let errors=Arc::new(Mutex::new(Vec::new()));
    let active=Arc::new(AtomicUsize::new(0));
    let found=Arc::new(AtomicUsize::new(0));
    let mut handles=Vec::with_capacity(threads);
    for _ in 0..threads{
        let queue=Arc::clone(&queue);let files=Arc::clone(&files);let errors=Arc::clone(&errors);let active=Arc::clone(&active);let found=Arc::clone(&found);let root=root.clone();
        handles.push(thread::spawn(move||loop{
            if max_files.is_some_and(|limit|found.load(Ordering::Relaxed)>=limit){break}
            let dir=queue.lock().unwrap().pop_front();
            let Some(dir)=dir else {
                if active.load(Ordering::SeqCst)==0{break}
                thread::sleep(Duration::from_millis(2));
                continue
            };
            active.fetch_add(1,Ordering::SeqCst);
            let read_dir=match fs::read_dir(&dir){Ok(rd)=>rd,Err(e)=>{let msg=format!("walk error reading {} under {}: {e}",dir.display(),root.display());backend_log(&msg);errors.lock().unwrap().push(msg);active.fetch_sub(1,Ordering::SeqCst);continue}};
            for entry in read_dir{
                let entry=match entry{Ok(e)=>e,Err(e)=>{let msg=format!("walk error under {}: {e}",root.display());backend_log(&msg);errors.lock().unwrap().push(msg);continue}};
                let path=entry.path();
                let Ok(ft)=entry.file_type() else { continue };
                if is_blacklisted_path(&path)||is_dotfile_path(&path){continue}
                if ft.is_dir(){queue.lock().unwrap().push_back(path)}else if ft.is_file(){
                    let ext=path.extension().and_then(|e|e.to_str());
                    if media_type_for_ext(ext).is_none(){continue}
                    let n=found.fetch_add(1,Ordering::Relaxed);
                    if max_files.is_some_and(|limit|n>=limit){break}
                    files.lock().unwrap().push(path)
                }
            }
            active.fetch_sub(1,Ordering::SeqCst);
        }));
    }
    for h in handles{if h.join().is_err(){let msg=format!("discovery worker thread panicked for root {}",root.display());backend_log(&msg);errors.lock().unwrap().push(msg);}}
    let files=match Arc::try_unwrap(files){Ok(m)=>m.into_inner().unwrap_or_default(),Err(a)=>a.lock().map(|v|v.clone()).unwrap_or_default()};
    let errors=match Arc::try_unwrap(errors){Ok(m)=>m.into_inner().unwrap_or_default(),Err(a)=>a.lock().map(|v|v.clone()).unwrap_or_else(|_|vec!["discovery error mutex poisoned".into()])};
    backend_log(&format!("finished discovery root={} files={} errors={}",root.display(),files.len(),errors.len()));
    (files,errors)
}
fn scan_library_impl(app:tauri::AppHandle,paths:Vec<String>)->Result<ScanSummary,String>{
    let mut conn=open_db(&app)?;
    let mut sum=ScanSummary{scanned_files:0,imported_or_updated:0,skipped_files:0,missing_marked:0,errors:vec![]};
    let mut seen=HashSet::new();
    let mut discovered_files=0usize;
    let mut total_files=None;
    let emit_progress=|sum:&ScanSummary,phase:&str,current_path:Option<String>,discovered_files:usize,total_files:Option<usize>,done:bool|{let _=app.emit("scan-progress",ScanProgress{phase:phase.into(),current_path,scanned_files:sum.scanned_files,imported_or_updated:sum.imported_or_updated,skipped_files:sum.skipped_files,missing_marked:sum.missing_marked,errors:sum.errors.len(),discovered_files,total_files,done});};
    emit_progress(&sum,"Starting scan",None,discovered_files,total_files,false);
    let discovery_threads=discovery_thread_count();
    let indexing_threads=indexing_thread_count();
    backend_log(&format!("scan started roots={} cpu_threads={} discovery_threads={} indexing_threads={}",paths.len(),available_threads(),discovery_threads,indexing_threads));
    for root in paths{
        emit_progress(&sum,&format!("Discovering files ({discovery_threads} discovery threads; {indexing_threads} indexing threads)"),Some(root.clone()),discovered_files,total_files,false);
        backend_log(&format!("scan root requested: {root}"));
        let root_path=Path::new(&root);
        if !root_path.exists(){let msg=format!("scan root does not exist: {root}");backend_log(&msg);sum.errors.push(msg);emit_progress(&sum,"Folder missing",Some(root),discovered_files,total_files,false);continue}
        if !root_path.is_dir(){let msg=format!("scan root is not a directory: {root}");backend_log(&msg);sum.errors.push(msg);emit_progress(&sum,"Not a directory",Some(root),discovered_files,total_files,false);continue}
        let (files,mut errors)=discover_files(root_path.to_path_buf(),discovery_threads,None);
        sum.errors.append(&mut errors);
        let discovered_count=files.len();
        discovered_files+=discovered_count;
        let files:Vec<PathBuf>=files.into_iter().filter(|p|seen.insert(p.clone())).collect();
        total_files=Some(total_files.unwrap_or(0)+files.len());
        backend_log(&format!("root {root}: discovered_files={discovered_count} unique_files={} total_errors={}",files.len(),sum.errors.len()));
        emit_progress(&sum,&format!("Indexing files ({indexing_threads} indexing threads; discovery used {discovery_threads})"),files.first().map(|p|clean_path_string(p)),discovered_files,total_files,false);
        let pool=rayon::ThreadPoolBuilder::new().num_threads(indexing_threads).build().map_err(|e|{let msg=format!("failed to build indexing thread pool: {e}");backend_log(&msg);msg})?;
        let (tx,rx)=mpsc::channel();
        let indexing_handle=thread::spawn(move||pool.install(||files.par_iter().for_each_with(tx,|tx,p|{let _=tx.send((p.clone(),media_from_path(p)));})));
        let mut results=0usize;
        let tx_db=conn.transaction().map_err(|e|format!("failed to begin index transaction: {e}"))?;
        for (scan_path,res) in rx{
            results+=1;
            sum.scanned_files+=1;
            match res.and_then(|o|if let Some(i)=o{upsert_media(&tx_db,&i).map(|_|true)}else{Ok(false)}){Ok(true)=>sum.imported_or_updated+=1,Ok(false)=>sum.skipped_files+=1,Err(e)=>{backend_log(&format!("index error for {}: {e}",scan_path.display()));sum.errors.push(e)}}
            if sum.scanned_files%50==0||sum.scanned_files==total_files.unwrap_or(0){emit_progress(&sum,"Indexing files",Some(clean_path_string(&scan_path)),discovered_files,total_files,false);}
        }
        tx_db.commit().map_err(|e|format!("failed to commit index transaction: {e}"))?;
        if indexing_handle.join().is_err(){let msg=format!("indexing worker thread panicked for root {root}");backend_log(&msg);sum.errors.push(msg);}
        backend_log(&format!("root {root}: metadata indexing complete results={results}"));
    }
    emit_progress(&sum,"Checking missing files",None,discovered_files,total_files,false);
    backend_log("checking missing files");
    sum.missing_marked=mark_missing_internal(&conn)?;
    backend_log(&format!("scan complete scanned={} imported_or_updated={} skipped={} missing_marked={} errors={}",sum.scanned_files,sum.imported_or_updated,sum.skipped_files,sum.missing_marked,sum.errors.len()));
    emit_progress(&sum,"Scan complete",None,discovered_files,total_files,true);
    Ok(sum)
}
#[tauri::command]
async fn scan_library(app:tauri::AppHandle,paths:Vec<String>)->Result<ScanSummary,String>{
    tauri::async_runtime::spawn_blocking(move||scan_library_impl(app,paths)).await.map_err(|e|{let msg=format!("scan task failed/thread panicked: {e}");backend_log(&msg);msg})?
}
fn mark_missing_internal(conn:&Connection)->Result<usize,String>{let mut s=conn.prepare("SELECT id,path FROM media_items WHERE missing=0").map_err(|e|e.to_string())?;let rows=s.query_map([],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,String>(1)?))).map_err(|e|e.to_string())?;let mut n=0;for r in rows{let(id,p)=r.map_err(|e|e.to_string())?;if !Path::new(&p).exists(){conn.execute("UPDATE media_items SET missing=1 WHERE id=?1",params![id]).map_err(|e|e.to_string())?;n+=1}}Ok(n)}
#[tauri::command] fn mark_missing(app:tauri::AppHandle)->Result<usize,String>{mark_missing_internal(&open_db(&app)?)}
#[tauri::command] async fn rescan(app:tauri::AppHandle)->Result<ScanSummary,String>{let paths=list_library_folders(app.clone())?;scan_library(app,paths).await}
#[tauri::command] fn get_media_item(app:tauri::AppHandle,id:i64)->Result<Option<MediaItem>,String>{open_db(&app)?.query_row(&format!("{MEDIA_SELECT} WHERE id=?1"),params![id],row_to_media).optional().map_err(|e|e.to_string())}
#[tauri::command] fn search_media(app:tauri::AppHandle,filter:SearchFilter)->Result<Vec<MediaItem>,String>{let c=open_db(&app)?;let mut sql=format!("{MEDIA_SELECT} WHERE 1=1");sql.push_str(&sql_path_not_blacklisted_clause());let mut v:Vec<Box<dyn ToSql>>=vec![];if filter.person_id.is_some()||filter.person_name.is_some(){sql.push_str(" AND EXISTS(SELECT 1 FROM faces f LEFT JOIN people p ON p.id=f.person_id WHERE f.media_item_id=media_items.id");if let Some(id)=filter.person_id{sql.push_str(" AND f.person_id=?");v.push(Box::new(id));}if let Some(n)=filter.person_name{sql.push_str(" AND p.name LIKE ?");v.push(Box::new(format!("%{}%",n)));}sql.push(')');}if let Some(q)=filter.query.filter(|q|!q.trim().is_empty()){sql.push_str(" AND (file_name LIKE ? OR path LIKE ?)");let like=format!("%{}%",q.trim());v.push(Box::new(like.clone()));v.push(Box::new(like));}if let Some(x)=filter.media_type{sql.push_str(" AND media_type=?");v.push(Box::new(x));}if let Some(x)=filter.missing{sql.push_str(" AND missing=?");v.push(Box::new(if x{1_i64}else{0_i64}));}if let Some(x)=filter.from_ts{sql.push_str(" AND COALESCE(captured_at,modified_at,created_at)>=?");v.push(Box::new(x));}if let Some(x)=filter.to_ts{sql.push_str(" AND COALESCE(captured_at,modified_at,created_at)<=?");v.push(Box::new(x));}if let Some(cam)=filter.camera{sql.push_str(" AND ((camera_make||' '||camera_model) LIKE ? OR camera_model LIKE ?)");let like=format!("%{}%",cam);v.push(Box::new(like.clone()));v.push(Box::new(like));}if let Some(x)=filter.has_gps{sql.push_str(if x{" AND latitude IS NOT NULL AND longitude IS NOT NULL"}else{" AND (latitude IS NULL OR longitude IS NULL)"});}if let Some(x)=filter.has_camera{sql.push_str(if x{" AND (camera_make IS NOT NULL OR camera_model IS NOT NULL)"}else{" AND camera_make IS NULL AND camera_model IS NULL"});}if let (Some(lat),Some(lng),Some(r))=(filter.lat,filter.lng,filter.radius_km){sql.push_str(" AND latitude IS NOT NULL AND longitude IS NOT NULL AND (111.045*sqrt((latitude-?)*(latitude-?) + ((longitude-?)*cos(?*0.0174532925199433))*((longitude-?)*cos(?*0.0174532925199433)))) <= ?");v.extend([Box::new(lat)as Box<dyn ToSql>,Box::new(lat),Box::new(lng),Box::new(lat),Box::new(lng),Box::new(lat),Box::new(r)]);}sql.push_str(" ORDER BY COALESCE(captured_at,modified_at,created_at) DESC,id DESC LIMIT ? OFFSET ?");v.push(Box::new(filter.limit.unwrap_or(100).clamp(1,500)));v.push(Box::new(filter.offset.unwrap_or(0).max(0)));let refs:Vec<&dyn ToSql>=v.iter().map(|x|x.as_ref()).collect();let mut st=c.prepare(&sql).map_err(|e|e.to_string())?;let res=st.query_map(params_from_iter(refs),row_to_media).map_err(|e|e.to_string())?.collect::<Result<Vec<_>,_>>().map_err(|e|e.to_string());res}
#[tauri::command] fn list_people(app:tauri::AppHandle)->Result<Vec<Person>,String>{let c=open_db(&app)?;let mut s=c.prepare("SELECT p.id,p.name,p.created_at,COUNT(f.id) FROM people p LEFT JOIN faces f ON f.person_id=p.id GROUP BY p.id ORDER BY p.name").map_err(|e|e.to_string())?;let res=s.query_map([],|r|Ok(Person{id:r.get(0)?,name:r.get(1)?,created_at:r.get(2)?,face_count:r.get(3)?})).map_err(|e|e.to_string())?.collect::<Result<Vec<_>,_>>().map_err(|e|e.to_string());res}
#[tauri::command] fn rename_person(app:tauri::AppHandle,person_id:i64,name:String)->Result<(),String>{open_db(&app)?.execute("UPDATE people SET name=?1 WHERE id=?2",params![name,person_id]).map_err(|e|e.to_string())?;Ok(())}
fn get_or_create_person(conn:&Connection,name:&str)->Result<i64,String>{let trimmed=name.trim();let name=if trimmed.is_empty(){"Unnamed"}else{trimmed};conn.execute("INSERT OR IGNORE INTO people(name,created_at) VALUES(?1,?2)",params![name,now_unix()]).map_err(|e|e.to_string())?;conn.query_row("SELECT id FROM people WHERE name=?1",params![name],|r|r.get(0)).map_err(|e|e.to_string())}
fn best_named_person_for_embedding(conn:&Connection,embedding:&[f32],threshold:f64)->Result<Option<i64>,String>{let mut s=conn.prepare("SELECT f.person_id,e.vector FROM embeddings e JOIN faces f ON f.id=e.face_id WHERE f.person_id IS NOT NULL").map_err(|e|e.to_string())?;let mut best=(threshold,None);for r in s.query_map([],|row|Ok((row.get::<_,i64>(0)?,row.get::<_,Vec<u8>>(1)?))).map_err(|e|e.to_string())?{let(pid,blob)=r.map_err(|e|e.to_string())?;if let Some(v)=parse_vec(blob){let score=cosine(embedding,&v);if score>best.0{best=(score,Some(pid));}}}Ok(best.1)}
fn propagate_person(conn:&Connection,person_id:i64,seed:&[f32],threshold:f64)->Result<usize,String>{let mut s=conn.prepare("SELECT e.face_id,e.vector FROM embeddings e JOIN faces f ON f.id=e.face_id WHERE f.person_id IS NULL").map_err(|e|e.to_string())?;let rows=s.query_map([],|row|Ok((row.get::<_,i64>(0)?,row.get::<_,Vec<u8>>(1)?))).map_err(|e|e.to_string())?.collect::<Result<Vec<_>,_>>().map_err(|e|e.to_string())?;let mut n=0;for(face_id,blob)in rows{if let Some(v)=parse_vec(blob){if cosine(seed,&v)>=threshold{conn.execute("UPDATE faces SET person_id=?1 WHERE id=?2",params![person_id,face_id]).map_err(|e|e.to_string())?;n+=1;}}}Ok(n)}
#[tauri::command] fn name_face(app:tauri::AppHandle,face_id:i64,name:String)->Result<usize,String>{let conn=open_db(&app)?;let pid=get_or_create_person(&conn,&name)?;conn.execute("UPDATE faces SET person_id=?1 WHERE id=?2",params![pid,face_id]).map_err(|e|e.to_string())?;let blob:Vec<u8>=conn.query_row("SELECT vector FROM embeddings WHERE face_id=?1 ORDER BY id DESC LIMIT 1",params![face_id],|r|r.get(0)).optional().map_err(|e|e.to_string())?.ok_or_else(||"face has no embedding yet".to_string())?;let seed=parse_vec(blob).ok_or_else(||"invalid face embedding".to_string())?;propagate_person(&conn,pid,&seed,0.88)
}
#[tauri::command] fn list_faces(app:tauri::AppHandle,media_item_id:Option<i64>,person_id:Option<i64>)->Result<Vec<Face>,String>{let c=open_db(&app)?;let mut sql="SELECT f.id,f.media_item_id,f.person_id,p.name,f.x,f.y,f.width,f.height,f.confidence,f.created_at FROM faces f LEFT JOIN people p ON p.id=f.person_id WHERE 1=1".to_string();let mut v:Vec<Box<dyn ToSql>>=vec![];if let Some(x)=media_item_id{sql.push_str(" AND f.media_item_id=?");v.push(Box::new(x));}if let Some(x)=person_id{sql.push_str(" AND f.person_id=?");v.push(Box::new(x));}let refs:Vec<&dyn ToSql>=v.iter().map(|x|x.as_ref()).collect();let mut s=c.prepare(&sql).map_err(|e|e.to_string())?;let res=s.query_map(params_from_iter(refs),|r|Ok(Face{id:r.get(0)?,media_item_id:r.get(1)?,person_id:r.get(2)?,person_name:r.get(3)?,x:r.get(4)?,y:r.get(5)?,width:r.get(6)?,height:r.get(7)?,confidence:r.get(8)?,created_at:r.get(9)?})).map_err(|e|e.to_string())?.collect::<Result<Vec<_>,_>>().map_err(|e|e.to_string());res}
fn sidecar_dir()->Result<PathBuf,String>{Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().ok_or_else(||"failed to resolve repo root".to_string())?.join("python-sidecar"))}
fn run_sidecar(args:Vec<String>)->Result<SidecarResult,String>{let dir=sidecar_dir()?;backend_log(&format!("running python sidecar args={args:?}"));let out=Command::new("python3").arg("-m").arg("rich_media_sidecar").args(args).current_dir(&dir).env("PYTHONPATH",dir.to_string_lossy().to_string()).output().map_err(|e|{let msg=format!("failed to run python sidecar: {e}");backend_log(&msg);msg})?;let stdout=String::from_utf8_lossy(&out.stdout).to_string();let stderr=String::from_utf8_lossy(&out.stderr).to_string();if !out.status.success()||!stderr.trim().is_empty(){backend_log(&format!("python sidecar status={} stderr={}",out.status,stderr.trim()));}Ok(SidecarResult{ok:out.status.success(),stdout,stderr})}
fn media_paths_for_ids(conn:&Connection,ids:Option<Vec<i64>>,images_only:bool)->Result<Vec<(i64,String)>,String>{let mut sql="SELECT id,path FROM media_items WHERE missing=0".to_string();sql.push_str(&sql_path_not_blacklisted_clause());if images_only{sql.push_str(" AND media_type='image'");}let mut out=vec![];if let Some(ids)=ids{for id in ids{if let Some(row)=conn.query_row(&sql.replace(" WHERE missing=0", " WHERE missing=0 AND id=?1"),params![id],|r|Ok((r.get(0)?,r.get(1)?))).optional().map_err(|e|e.to_string())?{out.push(row)}}}else{let mut s=conn.prepare(&sql).map_err(|e|e.to_string())?;out=s.query_map([],|r|Ok((r.get(0)?,r.get(1)?))).map_err(|e|e.to_string())?.collect::<Result<Vec<_>,_>>().map_err(|e|e.to_string())?;}Ok(out)}
#[tauri::command]
fn cluster_faces(app:tauri::AppHandle,media_ids:Option<Vec<i64>>)->Result<SidecarResult,String>{
    // Kept for compatibility, but intentionally capped so setup cannot accidentally process huge libraries.
    let ids=media_ids.ok_or_else(||"Use Guided Face Setup to process images incrementally instead of clustering the whole library.".to_string())?;
    if ids.len()>25{return Err("Too many images for one face-clustering call; use Guided Face Setup.".into())}
    process_face_paths(app,Some(ids),false)
}
fn process_face_paths(app:tauri::AppHandle,media_ids:Option<Vec<i64>>,auto_name_clusters:bool)->Result<SidecarResult,String>{
    let mut conn=open_db(&app)?;
    let rows=media_paths_for_ids(&conn,media_ids,true)?;
    let paths:Vec<String>=rows.iter().map(|(_,p)|p.clone()).collect();
    let payload=serde_json::json!({"paths":paths}).to_string();
    let res=run_sidecar(vec!["cluster-faces".into(),"--json".into(),payload])?;
    if res.ok{
        let root:serde_json::Value=serde_json::from_str(&res.stdout).map_err(|e|format!("invalid sidecar JSON: {e}"))?;
        if let Some(faces)=root.pointer("/data/faces").and_then(|v|v.as_array()){
            let tx=conn.transaction().map_err(|e|e.to_string())?;
            for (mid,_) in &rows{tx.execute("DELETE FROM faces WHERE media_item_id=?1",params![mid]).map_err(|e|e.to_string())?;}
            for face in faces{
                let path=face.get("path").and_then(|v|v.as_str()).unwrap_or_default();
                let Some((mid,_))=rows.iter().find(|(_,p)|p==path) else{continue};
                let emb:Vec<f32>=serde_json::from_value(face.get("embedding").cloned().unwrap_or(serde_json::Value::Null)).unwrap_or_default();
                let pid=if auto_name_clusters{let cluster=face.get("cluster_id").and_then(|v|v.as_i64()).unwrap_or(0);Some(get_or_create_person(&tx,&format!("Person {}",cluster+1))?)}else if emb.is_empty(){None}else{best_named_person_for_embedding(&tx,&emb,0.88)?};
                let bbox=face.get("bbox").and_then(|v|v.as_array()).cloned().unwrap_or_default();
                let x=bbox.get(0).and_then(|v|v.as_f64()).unwrap_or(0.0);let y=bbox.get(1).and_then(|v|v.as_f64()).unwrap_or(0.0);let w=bbox.get(2).and_then(|v|v.as_f64()).unwrap_or(0.0);let h=bbox.get(3).and_then(|v|v.as_f64()).unwrap_or(0.0);let conf=face.get("confidence").and_then(|v|v.as_f64());
                tx.execute("INSERT INTO faces(media_item_id,person_id,x,y,width,height,confidence,created_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",params![mid,pid,x,y,w,h,conf,now_unix()]).map_err(|e|e.to_string())?;
                let face_id=tx.last_insert_rowid();
                if !emb.is_empty(){let vec_json=serde_json::to_vec(&emb).map_err(|e|e.to_string())?;tx.execute("INSERT INTO embeddings(face_id,model,vector,created_at) VALUES(?1,'local-face',?2,?3)",params![face_id,vec_json,now_unix()]).map_err(|e|e.to_string())?;}
            }
            tx.commit().map_err(|e|e.to_string())?;
        }
    }
    Ok(res)
}
#[tauri::command] fn process_face_setup_image(app:tauri::AppHandle,media_id:i64)->Result<SidecarResult,String>{process_face_paths(app,Some(vec![media_id]),false)}
#[tauri::command]
fn generate_embeddings(app:tauri::AppHandle,media_ids:Option<Vec<i64>>,provider:Option<String>,model:Option<String>)->Result<SidecarResult,String>{
    let mut conn=open_db(&app)?;
    let rows=media_paths_for_ids(&conn,media_ids,false)?;
    let paths:Vec<String>=rows.iter().map(|(_,p)|p.clone()).collect();
    let payload=serde_json::json!({"paths":paths}).to_string();
    let args=vec!["embed".into(),"--provider".into(),provider.unwrap_or_else(||"ollama".into()),"--model".into(),model.unwrap_or_else(||"nomic-embed-text".into()),"--workers".into(),indexing_thread_count().to_string(),"--json".into(),payload];
    let res=run_sidecar(args)?;
    if res.ok{
        let root:serde_json::Value=serde_json::from_str(&res.stdout).map_err(|e|format!("invalid sidecar JSON: {e}"))?;
        if let Some(embs)=root.pointer("/data/embeddings").and_then(|v|v.as_array()){
            let tx=conn.transaction().map_err(|e|e.to_string())?;
            for emb in embs{
                let src=emb.get("source").and_then(|v|v.as_str()).unwrap_or_default();
                let Some((mid,_))=rows.iter().find(|(_,p)|p==src) else{continue};
                let model=emb.get("model").and_then(|v|v.as_str()).unwrap_or("unknown");
                let Some(vecv)=emb.get("embedding").and_then(|v|v.as_array()).filter(|v|!v.is_empty()) else { continue };
                let vec_json=serde_json::to_vec(vecv).map_err(|e|e.to_string())?;
                tx.execute("INSERT INTO embeddings(media_item_id,model,vector,created_at) VALUES(?1,?2,?3,?4)",params![mid,model,vec_json,now_unix()]).map_err(|e|e.to_string())?;
            }
            tx.commit().map_err(|e|e.to_string())?;
        }
    }
    Ok(res)
}
fn parse_vec(b:Vec<u8>)->Option<Vec<f32>>{serde_json::from_slice(&b).ok().or_else(||std::str::from_utf8(&b).ok().and_then(|s|serde_json::from_str(s).ok()))}
fn cosine(a:&[f32],b:&[f32])->f64{let(mut dot,mut na,mut nb)=(0.0,0.0,0.0);for(x,y)in a.iter().zip(b){dot+=(*x as f64)*(*y as f64);na+=(*x as f64)*(*x as f64);nb+=(*y as f64)*(*y as f64);}if na==0.0||nb==0.0{0.0}else{dot/(na.sqrt()*nb.sqrt())}}
fn search_semantic_impl(app:tauri::AppHandle,vector:Vec<f32>,model:Option<String>,limit:Option<i64>)->Result<Vec<SemanticHit>,String>{let c=open_db(&app)?;let mut semantic_sql=format!("SELECT e.vector,media_items.id,media_items.path,media_items.file_name,media_items.extension,media_items.media_type,media_items.size_bytes,media_items.created_at,media_items.modified_at,media_items.imported_at,media_items.missing,media_items.camera_make,media_items.camera_model,media_items.lens_model,media_items.captured_at,media_items.latitude,media_items.longitude,media_items.metadata_json FROM embeddings e JOIN media_items ON media_items.id=e.media_item_id WHERE e.media_item_id IS NOT NULL{}",sql_path_not_blacklisted_clause());let mut params_vec:Vec<Box<dyn ToSql>>=vec![];if let Some(model)=model{semantic_sql.push_str(" AND e.model=?");params_vec.push(Box::new(model));}let refs:Vec<&dyn ToSql>=params_vec.iter().map(|x|x.as_ref()).collect();let mut s=c.prepare(&semantic_sql).map_err(|e|e.to_string())?;let rows=s.query_map(params_from_iter(refs),|row|Ok((row.get::<_,Vec<u8>>(0)?,row_to_media_offset(row,1)?))).map_err(|e|e.to_string())?;let mut hits=vec![];for r in rows{let(b,item)=r.map_err(|e|e.to_string())?;if let Some(v)=parse_vec(b){if v.len()==vector.len(){hits.push(SemanticHit{score:cosine(&vector,&v),item});}}}hits.sort_by(|a,b|b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));hits.truncate(limit.unwrap_or(50).clamp(1,200) as usize);Ok(hits)}
#[tauri::command] fn search_semantic(app:tauri::AppHandle,vector:Vec<f32>,limit:Option<i64>)->Result<Vec<SemanticHit>,String>{search_semantic_impl(app,vector,None,limit)}
#[tauri::command] fn search_semantic_text(app:tauri::AppHandle,query:String,provider:Option<String>,model:Option<String>,limit:Option<i64>)->Result<Vec<SemanticHit>,String>{let selected_model=model.unwrap_or_else(||"nomic-embed-text".into());let c=open_db(&app)?;let n:i64=c.query_row("SELECT COUNT(*) FROM embeddings WHERE media_item_id IS NOT NULL AND model=?1",params![selected_model],|r|r.get(0)).map_err(|e|e.to_string())?;if n==0{return Ok(vec![])}drop(c);let args=vec!["embed".into(),"--provider".into(),provider.unwrap_or_else(||"ollama".into()),"--model".into(),selected_model.clone(),"--workers".into(),indexing_thread_count().to_string(),"--text".into(),query];let res=run_sidecar(args)?;if !res.ok{return Err(res.stderr)}let root:serde_json::Value=serde_json::from_str(&res.stdout).map_err(|e|format!("invalid sidecar JSON: {e}"))?;let vecv=root.pointer("/data/embeddings/0/embedding").ok_or_else(||"sidecar did not return a query embedding".to_string())?;let vector:Vec<f32>=serde_json::from_value(vecv.clone()).map_err(|e|e.to_string())?;search_semantic_impl(app,vector,Some(selected_model),limit)}
fn row_to_media_offset(row:&rusqlite::Row<'_>,o:usize)->rusqlite::Result<MediaItem>{Ok(MediaItem{id:row.get(o)?,path:row.get(o+1)?,file_name:row.get(o+2)?,extension:row.get(o+3)?,media_type:row.get(o+4)?,size_bytes:row.get(o+5)?,created_at:row.get(o+6)?,modified_at:row.get(o+7)?,imported_at:row.get(o+8)?,missing:row.get::<_,i64>(o+9)?!=0,camera_make:row.get(o+10)?,camera_model:row.get(o+11)?,lens_model:row.get(o+12)?,captured_at:row.get(o+13)?,latitude:row.get(o+14)?,longitude:row.get(o+15)?,metadata_json:row.get(o+16)?})}
#[tauri::command] fn greet(name:&str)->String{format!("Welcome to {name}.")}
pub fn run(){tauri::Builder::default().plugin(tauri_plugin_dialog::init()).invoke_handler(tauri::generate_handler![greet,initialize_app,delete_current_index,choose_media_folder,get_settings,update_settings,list_library_folders,add_library_folder,remove_library_folder,scan_library,search_media,get_media_item,mark_missing,rescan,list_people,rename_person,name_face,list_faces,cluster_faces,process_face_setup_image,generate_embeddings,search_semantic,search_semantic_text]).run(tauri::generate_context!()).expect("error while running Tauri application");}
