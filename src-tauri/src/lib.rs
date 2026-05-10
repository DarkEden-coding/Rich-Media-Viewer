use rusqlite::{params, params_from_iter, Connection, OptionalExtension, ToSql};
use serde::{Deserialize, Serialize};
use std::{fs, path::{Path, PathBuf}, process::Command, time::{SystemTime, UNIX_EPOCH}};
#[cfg(not(debug_assertions))]
use tauri::Manager;
use tauri_plugin_dialog::DialogExt;
use walkdir::WalkDir;

const DB_FILE: &str = "rich-media-viewer.sqlite3";

#[derive(Debug, Serialize)] struct AppInfo { data_dir: String, database_path: String }
#[derive(Debug, Serialize, Deserialize, Clone)]
struct MediaItem { id:i64,path:String,file_name:String,extension:Option<String>,media_type:String,size_bytes:Option<i64>,created_at:Option<i64>,modified_at:Option<i64>,imported_at:i64,missing:bool,camera_make:Option<String>,camera_model:Option<String>,lens_model:Option<String>,captured_at:Option<i64>,latitude:Option<f64>,longitude:Option<f64>,metadata_json:Option<String> }
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
struct AppSettings { library_folders: Vec<String> }
#[derive(Debug, Deserialize, Default)]
struct SearchFilter { query:Option<String>,media_type:Option<String>,missing:Option<bool>,from_ts:Option<i64>,to_ts:Option<i64>,camera:Option<String>,lat:Option<f64>,lng:Option<f64>,radius_km:Option<f64>,person_id:Option<i64>,person_name:Option<String>,has_gps:Option<bool>,has_camera:Option<bool>,limit:Option<i64>,offset:Option<i64> }
#[derive(Debug, Serialize)] struct ScanSummary { scanned_files:usize, imported_or_updated:usize, skipped_files:usize, missing_marked:usize, errors:Vec<String> }
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
fn media_type_for_ext(ext:Option<&str>)->Option<&'static str>{match ext.unwrap_or_default().to_ascii_lowercase().as_str(){"jpg"|"jpeg"|"png"|"gif"|"webp"|"bmp"|"tif"|"tiff"|"heic"|"heif"=>Some("image"),"mp4"|"mov"|"m4v"|"avi"|"mkv"|"webm"=>Some("video"),_=>None}}
fn parse_exif(path:&Path)->(Option<String>,Option<String>,Option<String>,Option<i64>,Option<f64>,Option<f64>,Option<String>){let Some(file)=fs::File::open(path).ok() else { return (None,None,None,None,None,None,None) };let mut br=std::io::BufReader::new(file);let Some(exif)=exif::Reader::new().read_from_container(&mut br).ok() else { return (None,None,None,None,None,None,None) };use exif::{In,Tag,Value};let get=|tag| exif.get_field(tag,In::PRIMARY).map(|f|f.display_value().with_unit(&exif).to_string());let make=get(Tag::Make);let model=get(Tag::Model);let lens=get(Tag::LensModel);let dt=get(Tag::DateTimeOriginal).and_then(|s| chrono::NaiveDateTime::parse_from_str(&s,"%Y-%m-%d %H:%M:%S").or_else(|_|chrono::NaiveDateTime::parse_from_str(&s,"%Y:%m:%d %H:%M:%S")).ok()).map(|d|d.and_utc().timestamp());let gps=|tag| exif.get_field(tag,In::PRIMARY).and_then(|f|if let Value::Rational(v)=&f.value{ if v.len()>=3{Some(v[0].to_f64()+v[1].to_f64()/60.0+v[2].to_f64()/3600.0)}else{None}}else{None});let mut lat=gps(Tag::GPSLatitude);let mut lng=gps(Tag::GPSLongitude);if get(Tag::GPSLatitudeRef).unwrap_or_default().contains('S'){lat=lat.map(|v|-v)}; if get(Tag::GPSLongitudeRef).unwrap_or_default().contains('W'){lng=lng.map(|v|-v)};let json=serde_json::to_string(&serde_json::json!({"make":make,"model":model,"lens_model":lens,"captured_at":dt,"latitude":lat,"longitude":lng})).ok();(make,model,lens,dt,lat,lng,json)}
fn media_from_path(path:&Path)->Result<Option<MediaItem>,String>{let ext=path.extension().and_then(|e|e.to_str()).map(|s|s.to_ascii_lowercase());let Some(mt)=media_type_for_ext(ext.as_deref()) else{return Ok(None)};let md=fs::metadata(path).map_err(|e|format!("{}: {e}",path.display()))?;if !md.is_file(){return Ok(None)};let (mk,mo,lens,cap,lat,lng,mjson)=if mt=="image"{parse_exif(path)}else{(None,None,None,None,None,None,None)};Ok(Some(MediaItem{id:0,path:path.to_string_lossy().to_string(),file_name:path.file_name().and_then(|n|n.to_str()).unwrap_or_default().to_string(),extension:ext,media_type:mt.to_string(),size_bytes:Some(md.len() as i64),created_at:md.created().ok().and_then(unix_time),modified_at:md.modified().ok().and_then(unix_time),imported_at:now_unix(),missing:false,camera_make:mk,camera_model:mo,lens_model:lens,captured_at:cap,latitude:lat,longitude:lng,metadata_json:mjson}))}
fn upsert_media(conn:&Connection,item:&MediaItem)->Result<(),String>{conn.execute(r#"INSERT INTO media_items(path,file_name,extension,media_type,size_bytes,created_at,modified_at,imported_at,missing,camera_make,camera_model,lens_model,captured_at,latitude,longitude,metadata_json) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,0,?9,?10,?11,?12,?13,?14,?15) ON CONFLICT(path) DO UPDATE SET file_name=excluded.file_name,extension=excluded.extension,media_type=excluded.media_type,size_bytes=excluded.size_bytes,created_at=excluded.created_at,modified_at=excluded.modified_at,missing=0,camera_make=excluded.camera_make,camera_model=excluded.camera_model,lens_model=excluded.lens_model,captured_at=excluded.captured_at,latitude=excluded.latitude,longitude=excluded.longitude,metadata_json=excluded.metadata_json"#,params![item.path,item.file_name,item.extension,item.media_type,item.size_bytes,item.created_at,item.modified_at,item.imported_at,item.camera_make,item.camera_model,item.lens_model,item.captured_at,item.latitude,item.longitude,item.metadata_json]).map_err(|e|format!("failed to upsert media item: {e}"))?;Ok(())}
const MEDIA_SELECT:&str="SELECT id,path,file_name,extension,media_type,size_bytes,created_at,modified_at,imported_at,missing,camera_make,camera_model,lens_model,captured_at,latitude,longitude,metadata_json FROM media_items";
fn row_to_media(row:&rusqlite::Row<'_>)->rusqlite::Result<MediaItem>{Ok(MediaItem{id:row.get(0)?,path:row.get(1)?,file_name:row.get(2)?,extension:row.get(3)?,media_type:row.get(4)?,size_bytes:row.get(5)?,created_at:row.get(6)?,modified_at:row.get(7)?,imported_at:row.get(8)?,missing:row.get::<_,i64>(9)?!=0,camera_make:row.get(10)?,camera_model:row.get(11)?,lens_model:row.get(12)?,captured_at:row.get(13)?,latitude:row.get(14)?,longitude:row.get(15)?,metadata_json:row.get(16)?})}
#[tauri::command] fn initialize_app(app:tauri::AppHandle)->Result<AppInfo,String>{let c=open_db(&app)?;drop(c);Ok(AppInfo{data_dir:app_data_dir(&app)?.to_string_lossy().to_string(),database_path:db_path(&app)?.to_string_lossy().to_string()})}
#[tauri::command] fn get_settings(app:tauri::AppHandle)->Result<AppSettings,String>{Ok(AppSettings{library_folders:list_library_folders(app)?})}
#[tauri::command] fn update_settings(app:tauri::AppHandle,settings:AppSettings)->Result<AppSettings,String>{let c=open_db(&app)?;c.execute("DELETE FROM library_folders",[]).map_err(|e|e.to_string())?;for p in &settings.library_folders{c.execute("INSERT OR IGNORE INTO library_folders(path,created_at) VALUES(?1,?2)",params![p,now_unix()]).map_err(|e|e.to_string())?;}Ok(settings)}
#[tauri::command] fn list_library_folders(app:tauri::AppHandle)->Result<Vec<String>,String>{let c=open_db(&app)?;let mut s=c.prepare("SELECT path FROM library_folders ORDER BY path").map_err(|e|e.to_string())?;let res=s.query_map([],|r|r.get(0)).map_err(|e|e.to_string())?.collect::<Result<Vec<_>,_>>().map_err(|e|e.to_string());res}
#[tauri::command] fn add_library_folder(app:tauri::AppHandle,path:String)->Result<Vec<String>,String>{let c=open_db(&app)?;if !Path::new(&path).is_dir(){return Err("not a directory".into())}c.execute("INSERT OR IGNORE INTO library_folders(path,created_at) VALUES(?1,?2)",params![path,now_unix()]).map_err(|e|e.to_string())?;drop(c);list_library_folders(app)}
#[tauri::command] fn remove_library_folder(app:tauri::AppHandle,path:String)->Result<Vec<String>,String>{let c=open_db(&app)?;c.execute("DELETE FROM library_folders WHERE path=?1",params![path]).map_err(|e|e.to_string())?;drop(c);list_library_folders(app)}
#[tauri::command] async fn choose_media_folder(app:tauri::AppHandle,path:Option<String>)->Result<Option<String>,String>{if let Some(p)=path{return Ok(Path::new(&p).is_dir().then_some(p))} ; Ok(app.dialog().file().blocking_pick_folder().map(|p|p.to_string()))}
#[tauri::command] fn scan_library(app:tauri::AppHandle,paths:Vec<String>)->Result<ScanSummary,String>{let conn=open_db(&app)?;let mut sum=ScanSummary{scanned_files:0,imported_or_updated:0,skipped_files:0,missing_marked:0,errors:vec![]};for root in paths{for entry in WalkDir::new(&root).follow_links(false).into_iter().filter_map(Result::ok){if !entry.file_type().is_file(){continue}sum.scanned_files+=1;match media_from_path(entry.path()).and_then(|o|if let Some(i)=o{upsert_media(&conn,&i).map(|_|true)}else{Ok(false)}){Ok(true)=>sum.imported_or_updated+=1,Ok(false)=>sum.skipped_files+=1,Err(e)=>sum.errors.push(e)}}}sum.missing_marked=mark_missing_internal(&conn)?;Ok(sum)}
fn mark_missing_internal(conn:&Connection)->Result<usize,String>{let mut s=conn.prepare("SELECT id,path FROM media_items WHERE missing=0").map_err(|e|e.to_string())?;let rows=s.query_map([],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,String>(1)?))).map_err(|e|e.to_string())?;let mut n=0;for r in rows{let(id,p)=r.map_err(|e|e.to_string())?;if !Path::new(&p).exists(){conn.execute("UPDATE media_items SET missing=1 WHERE id=?1",params![id]).map_err(|e|e.to_string())?;n+=1}}Ok(n)}
#[tauri::command] fn mark_missing(app:tauri::AppHandle)->Result<usize,String>{mark_missing_internal(&open_db(&app)?)}
#[tauri::command] fn rescan(app:tauri::AppHandle)->Result<ScanSummary,String>{let paths=list_library_folders(app.clone())?;scan_library(app,paths)}
#[tauri::command] fn get_media_item(app:tauri::AppHandle,id:i64)->Result<Option<MediaItem>,String>{open_db(&app)?.query_row(&format!("{MEDIA_SELECT} WHERE id=?1"),params![id],row_to_media).optional().map_err(|e|e.to_string())}
#[tauri::command] fn search_media(app:tauri::AppHandle,filter:SearchFilter)->Result<Vec<MediaItem>,String>{let c=open_db(&app)?;let mut sql=format!("{MEDIA_SELECT} WHERE 1=1");let mut v:Vec<Box<dyn ToSql>>=vec![];if filter.person_id.is_some()||filter.person_name.is_some(){sql.push_str(" AND EXISTS(SELECT 1 FROM faces f LEFT JOIN people p ON p.id=f.person_id WHERE f.media_item_id=media_items.id");if let Some(id)=filter.person_id{sql.push_str(" AND f.person_id=?");v.push(Box::new(id));}if let Some(n)=filter.person_name{sql.push_str(" AND p.name LIKE ?");v.push(Box::new(format!("%{}%",n)));}sql.push(')');}if let Some(q)=filter.query.filter(|q|!q.trim().is_empty()){sql.push_str(" AND (file_name LIKE ? OR path LIKE ?)");let like=format!("%{}%",q.trim());v.push(Box::new(like.clone()));v.push(Box::new(like));}if let Some(x)=filter.media_type{sql.push_str(" AND media_type=?");v.push(Box::new(x));}if let Some(x)=filter.missing{sql.push_str(" AND missing=?");v.push(Box::new(if x{1_i64}else{0_i64}));}if let Some(x)=filter.from_ts{sql.push_str(" AND COALESCE(captured_at,modified_at,created_at)>=?");v.push(Box::new(x));}if let Some(x)=filter.to_ts{sql.push_str(" AND COALESCE(captured_at,modified_at,created_at)<=?");v.push(Box::new(x));}if let Some(cam)=filter.camera{sql.push_str(" AND ((camera_make||' '||camera_model) LIKE ? OR camera_model LIKE ?)");let like=format!("%{}%",cam);v.push(Box::new(like.clone()));v.push(Box::new(like));}if let Some(x)=filter.has_gps{sql.push_str(if x{" AND latitude IS NOT NULL AND longitude IS NOT NULL"}else{" AND (latitude IS NULL OR longitude IS NULL)"});}if let Some(x)=filter.has_camera{sql.push_str(if x{" AND (camera_make IS NOT NULL OR camera_model IS NOT NULL)"}else{" AND camera_make IS NULL AND camera_model IS NULL"});}if let (Some(lat),Some(lng),Some(r))=(filter.lat,filter.lng,filter.radius_km){sql.push_str(" AND latitude IS NOT NULL AND longitude IS NOT NULL AND (111.045*sqrt((latitude-?)*(latitude-?) + ((longitude-?)*cos(?*0.0174532925199433))*((longitude-?)*cos(?*0.0174532925199433)))) <= ?");v.extend([Box::new(lat)as Box<dyn ToSql>,Box::new(lat),Box::new(lng),Box::new(lat),Box::new(lng),Box::new(lat),Box::new(r)]);}sql.push_str(" ORDER BY COALESCE(captured_at,modified_at,created_at) DESC,id DESC LIMIT ? OFFSET ?");v.push(Box::new(filter.limit.unwrap_or(100).clamp(1,500)));v.push(Box::new(filter.offset.unwrap_or(0).max(0)));let refs:Vec<&dyn ToSql>=v.iter().map(|x|x.as_ref()).collect();let mut st=c.prepare(&sql).map_err(|e|e.to_string())?;let res=st.query_map(params_from_iter(refs),row_to_media).map_err(|e|e.to_string())?.collect::<Result<Vec<_>,_>>().map_err(|e|e.to_string());res}
#[tauri::command] fn list_people(app:tauri::AppHandle)->Result<Vec<Person>,String>{let c=open_db(&app)?;let mut s=c.prepare("SELECT p.id,p.name,p.created_at,COUNT(f.id) FROM people p LEFT JOIN faces f ON f.person_id=p.id GROUP BY p.id ORDER BY p.name").map_err(|e|e.to_string())?;let res=s.query_map([],|r|Ok(Person{id:r.get(0)?,name:r.get(1)?,created_at:r.get(2)?,face_count:r.get(3)?})).map_err(|e|e.to_string())?.collect::<Result<Vec<_>,_>>().map_err(|e|e.to_string());res}
#[tauri::command] fn rename_person(app:tauri::AppHandle,person_id:i64,name:String)->Result<(),String>{open_db(&app)?.execute("UPDATE people SET name=?1 WHERE id=?2",params![name,person_id]).map_err(|e|e.to_string())?;Ok(())}
#[tauri::command] fn list_faces(app:tauri::AppHandle,media_item_id:Option<i64>,person_id:Option<i64>)->Result<Vec<Face>,String>{let c=open_db(&app)?;let mut sql="SELECT f.id,f.media_item_id,f.person_id,p.name,f.x,f.y,f.width,f.height,f.confidence,f.created_at FROM faces f LEFT JOIN people p ON p.id=f.person_id WHERE 1=1".to_string();let mut v:Vec<Box<dyn ToSql>>=vec![];if let Some(x)=media_item_id{sql.push_str(" AND f.media_item_id=?");v.push(Box::new(x));}if let Some(x)=person_id{sql.push_str(" AND f.person_id=?");v.push(Box::new(x));}let refs:Vec<&dyn ToSql>=v.iter().map(|x|x.as_ref()).collect();let mut s=c.prepare(&sql).map_err(|e|e.to_string())?;let res=s.query_map(params_from_iter(refs),|r|Ok(Face{id:r.get(0)?,media_item_id:r.get(1)?,person_id:r.get(2)?,person_name:r.get(3)?,x:r.get(4)?,y:r.get(5)?,width:r.get(6)?,height:r.get(7)?,confidence:r.get(8)?,created_at:r.get(9)?})).map_err(|e|e.to_string())?.collect::<Result<Vec<_>,_>>().map_err(|e|e.to_string());res}
fn sidecar_dir()->Result<PathBuf,String>{std::env::current_dir().map(|p|p.join("python-sidecar")).map_err(|e|e.to_string())}
fn run_sidecar(args:Vec<String>)->Result<SidecarResult,String>{let dir=sidecar_dir()?;let out=Command::new("python3").arg("-m").arg("rich_media_sidecar").args(args).current_dir(&dir).env("PYTHONPATH",dir.to_string_lossy().to_string()).output().map_err(|e|format!("failed to run python sidecar: {e}"))?;Ok(SidecarResult{ok:out.status.success(),stdout:String::from_utf8_lossy(&out.stdout).to_string(),stderr:String::from_utf8_lossy(&out.stderr).to_string()})}
fn media_paths_for_ids(conn:&Connection,ids:Option<Vec<i64>>,images_only:bool)->Result<Vec<(i64,String)>,String>{let mut sql="SELECT id,path FROM media_items WHERE missing=0".to_string();if images_only{sql.push_str(" AND media_type='image'");}let mut out=vec![];if let Some(ids)=ids{for id in ids{if let Some(row)=conn.query_row(&sql.replace(" WHERE missing=0", " WHERE missing=0 AND id=?1"),params![id],|r|Ok((r.get(0)?,r.get(1)?))).optional().map_err(|e|e.to_string())?{out.push(row)}}}else{let mut s=conn.prepare(&sql).map_err(|e|e.to_string())?;out=s.query_map([],|r|Ok((r.get(0)?,r.get(1)?))).map_err(|e|e.to_string())?.collect::<Result<Vec<_>,_>>().map_err(|e|e.to_string())?;}Ok(out)}
#[tauri::command]
fn cluster_faces(app:tauri::AppHandle,media_ids:Option<Vec<i64>>)->Result<SidecarResult,String>{
    let mut conn=open_db(&app)?;
    let rows=media_paths_for_ids(&conn,media_ids,true)?;
    let paths:Vec<String>=rows.iter().map(|(_,p)|p.clone()).collect();
    let payload=serde_json::json!({"paths":paths}).to_string();
    let res=run_sidecar(vec!["cluster-faces".into(),"--json".into(),payload])?;
    if res.ok{
        let root:serde_json::Value=serde_json::from_str(&res.stdout).map_err(|e|format!("invalid sidecar JSON: {e}"))?;
        if let Some(faces)=root.pointer("/data/faces").and_then(|v|v.as_array()){
            let tx=conn.transaction().map_err(|e|e.to_string())?;
            for face in faces{
                let path=face.get("path").and_then(|v|v.as_str()).unwrap_or_default();
                let Some((mid,_))=rows.iter().find(|(_,p)|p==path) else{continue};
                let cluster=face.get("cluster_id").and_then(|v|v.as_i64()).unwrap_or(0);
                let pname=format!("Person {}",cluster+1);
                tx.execute("INSERT OR IGNORE INTO people(name,created_at) VALUES(?1,?2)",params![pname,now_unix()]).map_err(|e|e.to_string())?;
                let pid:i64=tx.query_row("SELECT id FROM people WHERE name=?1",params![pname],|r|r.get(0)).map_err(|e|e.to_string())?;
                let bbox=face.get("bbox").and_then(|v|v.as_array()).cloned().unwrap_or_default();
                let x=bbox.get(0).and_then(|v|v.as_f64()).unwrap_or(0.0);
                let y=bbox.get(1).and_then(|v|v.as_f64()).unwrap_or(0.0);
                let w=bbox.get(2).and_then(|v|v.as_f64()).unwrap_or(0.0);
                let h=bbox.get(3).and_then(|v|v.as_f64()).unwrap_or(0.0);
                let conf=face.get("confidence").and_then(|v|v.as_f64());
                tx.execute("INSERT INTO faces(media_item_id,person_id,x,y,width,height,confidence,created_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",params![mid,pid,x,y,w,h,conf,now_unix()]).map_err(|e|e.to_string())?;
            }
            tx.commit().map_err(|e|e.to_string())?;
        }
    }
    Ok(res)
}
#[tauri::command]
fn generate_embeddings(app:tauri::AppHandle,media_ids:Option<Vec<i64>>,provider:Option<String>,allow_remote:Option<bool>)->Result<SidecarResult,String>{
    let mut conn=open_db(&app)?;
    let rows=media_paths_for_ids(&conn,media_ids,false)?;
    let paths:Vec<String>=rows.iter().map(|(_,p)|p.clone()).collect();
    let payload=serde_json::json!({"paths":paths}).to_string();
    let mut args=vec!["embed".into(),"--provider".into(),provider.unwrap_or_else(||"local".into())];
    if allow_remote.unwrap_or(false){args.push("--allow-remote".into())}
    args.extend(["--json".into(),payload]);
    let res=run_sidecar(args)?;
    if res.ok{
        let root:serde_json::Value=serde_json::from_str(&res.stdout).map_err(|e|format!("invalid sidecar JSON: {e}"))?;
        if let Some(embs)=root.pointer("/data/embeddings").and_then(|v|v.as_array()){
            let tx=conn.transaction().map_err(|e|e.to_string())?;
            for emb in embs{
                let src=emb.get("source").and_then(|v|v.as_str()).unwrap_or_default();
                let Some((mid,_))=rows.iter().find(|(_,p)|p==src) else{continue};
                let model=emb.get("model").and_then(|v|v.as_str()).unwrap_or("unknown");
                let vec_json=serde_json::to_vec(emb.get("embedding").unwrap_or(&serde_json::Value::Null)).map_err(|e|e.to_string())?;
                tx.execute("INSERT INTO embeddings(media_item_id,model,vector,created_at) VALUES(?1,?2,?3,?4)",params![mid,model,vec_json,now_unix()]).map_err(|e|e.to_string())?;
            }
            tx.commit().map_err(|e|e.to_string())?;
        }
    }
    Ok(res)
}
fn parse_vec(b:Vec<u8>)->Option<Vec<f32>>{serde_json::from_slice(&b).ok().or_else(||std::str::from_utf8(&b).ok().and_then(|s|serde_json::from_str(s).ok()))}
fn cosine(a:&[f32],b:&[f32])->f64{let(mut dot,mut na,mut nb)=(0.0,0.0,0.0);for(x,y)in a.iter().zip(b){dot+=(*x as f64)*(*y as f64);na+=(*x as f64)*(*x as f64);nb+=(*y as f64)*(*y as f64);}if na==0.0||nb==0.0{0.0}else{dot/(na.sqrt()*nb.sqrt())}}
#[tauri::command] fn search_semantic(app:tauri::AppHandle,vector:Vec<f32>,limit:Option<i64>)->Result<Vec<SemanticHit>,String>{let c=open_db(&app)?;let mut s=c.prepare("SELECT e.vector,media_items.id,media_items.path,media_items.file_name,media_items.extension,media_items.media_type,media_items.size_bytes,media_items.created_at,media_items.modified_at,media_items.imported_at,media_items.missing,media_items.camera_make,media_items.camera_model,media_items.lens_model,media_items.captured_at,media_items.latitude,media_items.longitude,media_items.metadata_json FROM embeddings e JOIN media_items ON media_items.id=e.media_item_id WHERE e.media_item_id IS NOT NULL").map_err(|e|e.to_string())?;let mut hits=vec![];for r in s.query_map([],|row|Ok((row.get::<_,Vec<u8>>(0)?,row_to_media_offset(row,1)?))).map_err(|e|e.to_string())?{let(b,item)=r.map_err(|e|e.to_string())?;if let Some(v)=parse_vec(b){hits.push(SemanticHit{score:cosine(&vector,&v),item});}}hits.sort_by(|a,b|b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));hits.truncate(limit.unwrap_or(50).clamp(1,200) as usize);Ok(hits)}
fn row_to_media_offset(row:&rusqlite::Row<'_>,o:usize)->rusqlite::Result<MediaItem>{Ok(MediaItem{id:row.get(o)?,path:row.get(o+1)?,file_name:row.get(o+2)?,extension:row.get(o+3)?,media_type:row.get(o+4)?,size_bytes:row.get(o+5)?,created_at:row.get(o+6)?,modified_at:row.get(o+7)?,imported_at:row.get(o+8)?,missing:row.get::<_,i64>(o+9)?!=0,camera_make:row.get(o+10)?,camera_model:row.get(o+11)?,lens_model:row.get(o+12)?,captured_at:row.get(o+13)?,latitude:row.get(o+14)?,longitude:row.get(o+15)?,metadata_json:row.get(o+16)?})}
#[tauri::command] fn greet(name:&str)->String{format!("Welcome to {name}.")}
pub fn run(){tauri::Builder::default().plugin(tauri_plugin_dialog::init()).invoke_handler(tauri::generate_handler![greet,initialize_app,choose_media_folder,get_settings,update_settings,list_library_folders,add_library_folder,remove_library_folder,scan_library,search_media,get_media_item,mark_missing,rescan,list_people,rename_person,list_faces,cluster_faces,generate_embeddings,search_semantic]).run(tauri::generate_context!()).expect("error while running Tauri application");}
