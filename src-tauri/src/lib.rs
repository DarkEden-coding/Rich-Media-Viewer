#[tauri::command]
fn greet(name: &str) -> String {
    format!("Welcome to {name}.")
}

pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running Tauri application");
}
