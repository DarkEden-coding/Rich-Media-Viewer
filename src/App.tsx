import { convertFileSrc, invoke } from '@tauri-apps/api/core';
import { useEffect, useMemo, useState } from 'react';

type AppInfo = { data_dir: string; database_path: string };
type MediaItem = {
  id: number;
  path: string;
  file_name: string;
  extension: string | null;
  media_type: 'image' | 'video' | string;
  size_bytes: number | null;
  created_at: number | null;
  modified_at: number | null;
  imported_at: number;
  missing: boolean;
  camera_make: string | null;
  camera_model: string | null;
  latitude: number | null;
  longitude: number | null;
};
type ScanSummary = { scanned_files: number; imported_or_updated: number; skipped_files: number; missing_marked: number; errors: string[] };
type ViewMode = 'grid' | 'list';

const emptyFilters = { query: '', mediaType: '', missing: '', from: '', to: '', lat: '', lng: '', radius: '', camera: '', person: '' };

function formatBytes(bytes?: number | null) {
  if (!bytes) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = bytes;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit += 1; }
  return `${size.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

function formatDate(seconds?: number | null) {
  return seconds ? new Date(seconds * 1000).toLocaleDateString() : 'Unknown';
}

function mediaUrl(item: MediaItem) {
  try { return convertFileSrc(item.path); } catch { return `file://${item.path}`; }
}

function App() {
  const [appInfo, setAppInfo] = useState<AppInfo | null>(null);
  const [setupOpen, setSetupOpen] = useState(true);
  const [folderInput, setFolderInput] = useState('');
  const [folders, setFolders] = useState<string[]>([]);
  const [provider, setProvider] = useState('Local metadata + filename search');
  const [privateMode, setPrivateMode] = useState(true);
  const [filters, setFilters] = useState(emptyFilters);
  const [items, setItems] = useState<MediaItem[]>([]);
  const [selected, setSelected] = useState<MediaItem | null>(null);
  const [view, setView] = useState<ViewMode>('grid');
  const [loading, setLoading] = useState(false);
  const [scan, setScan] = useState<ScanSummary | null>(null);
  const [notice, setNotice] = useState('Starting Rich Media Viewer…');

  const counts = useMemo(() => ({
    total: items.length,
    images: items.filter((i) => i.media_type === 'image').length,
    videos: items.filter((i) => i.media_type === 'video').length,
    missing: items.filter((i) => i.missing).length,
  }), [items]);

  async function runSearch(next = filters) {
    setLoading(true);
    try {
      const result = await invoke<MediaItem[]>('search_media', { filter: { query: next.query || undefined, media_type: next.mediaType || undefined, missing: next.missing === '' ? undefined : next.missing === 'true', limit: 120, offset: 0 } });
      setItems(result);
      setNotice(`${result.length} media items loaded`);
    } catch (error) { setNotice(`Search unavailable: ${String(error)}`); }
    finally { setLoading(false); }
  }

  useEffect(() => {
    invoke<AppInfo>('initialize_app')
      .then((info) => { setAppInfo(info); setNotice('Library database ready'); return runSearch(); })
      .catch((error) => setNotice(`Running in preview mode: ${String(error)}`));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function addFolder() {
    const path = folderInput.trim();
    if (!path) return;
    setFolders((prev) => Array.from(new Set([...prev, path])));
    setFolderInput('');
  }

  async function scanFolders(paths = folders) {
    if (!paths.length) { setNotice('Add at least one folder path before scanning.'); return; }
    setLoading(true); setNotice('Scanning library…');
    try {
      const summary = await invoke<ScanSummary>('scan_library', { paths });
      setScan(summary); setSetupOpen(false); setNotice(`Scan complete: ${summary.imported_or_updated} imported/updated`);
      await runSearch();
    } catch (error) { setNotice(`Scan failed: ${String(error)}`); }
    finally { setLoading(false); }
  }

  async function openItem(item: MediaItem) {
    setSelected(item);
    try { setSelected((await invoke<MediaItem | null>('get_media_item', { id: item.id })) ?? item); } catch { /* keep optimistic item */ }
  }

  const updateFilter = (key: keyof typeof emptyFilters, value: string) => setFilters((f) => ({ ...f, [key]: value }));

  return <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,#155e75_0,#020617_35%,#020617_100%)] text-slate-100">
    <div className="flex min-h-screen">
      <aside className="w-96 shrink-0 border-r border-white/10 bg-slate-950/80 p-5 backdrop-blur">
        <div className="mb-6">
          <p className="text-xs font-semibold uppercase tracking-[0.35em] text-cyan-300">Private library</p>
          <h1 className="mt-2 text-3xl font-bold">Rich Media Viewer</h1>
          <p className="mt-2 text-sm text-slate-400">Semantic-ready search shell with local-first indexing.</p>
        </div>
        <button onClick={() => setSetupOpen(true)} className="mb-4 w-full rounded-xl border border-cyan-400/30 bg-cyan-400/10 px-4 py-3 text-left text-sm font-semibold text-cyan-100 hover:bg-cyan-400/20">Setup folders / providers</button>
        <section className="space-y-4 rounded-2xl border border-white/10 bg-white/[0.04] p-4 shadow-2xl">
          <label className="block"><span className="text-xs uppercase text-slate-400">Semantic query</span><input value={filters.query} onChange={(e) => updateFilter('query', e.target.value)} placeholder="sunset beach, dog, receipt…" className="mt-1 w-full rounded-lg border border-white/10 bg-slate-900 px-3 py-2 outline-none focus:border-cyan-300" /></label>
          <div className="grid grid-cols-2 gap-3"><label><span className="text-xs uppercase text-slate-400">From</span><input type="date" value={filters.from} onChange={(e) => updateFilter('from', e.target.value)} className="mt-1 w-full rounded-lg border border-white/10 bg-slate-900 px-3 py-2" /></label><label><span className="text-xs uppercase text-slate-400">To</span><input type="date" value={filters.to} onChange={(e) => updateFilter('to', e.target.value)} className="mt-1 w-full rounded-lg border border-white/10 bg-slate-900 px-3 py-2" /></label></div>
          <div className="grid grid-cols-3 gap-2"><input value={filters.lat} onChange={(e) => updateFilter('lat', e.target.value)} placeholder="Lat" className="rounded-lg border border-white/10 bg-slate-900 px-3 py-2"/><input value={filters.lng} onChange={(e) => updateFilter('lng', e.target.value)} placeholder="Lng" className="rounded-lg border border-white/10 bg-slate-900 px-3 py-2"/><input value={filters.radius} onChange={(e) => updateFilter('radius', e.target.value)} placeholder="Mi" className="rounded-lg border border-white/10 bg-slate-900 px-3 py-2"/></div>
          <select value={filters.person} onChange={(e) => updateFilter('person', e.target.value)} className="w-full rounded-lg border border-white/10 bg-slate-900 px-3 py-2"><option value="">People selector (coming soon)</option><option>Unassigned faces</option></select>
          <div className="grid grid-cols-2 gap-3"><select value={filters.mediaType} onChange={(e) => updateFilter('mediaType', e.target.value)} className="rounded-lg border border-white/10 bg-slate-900 px-3 py-2"><option value="">All media</option><option value="image">Images</option><option value="video">Videos</option></select><select value={filters.missing} onChange={(e) => updateFilter('missing', e.target.value)} className="rounded-lg border border-white/10 bg-slate-900 px-3 py-2"><option value="">Any status</option><option value="false">Available</option><option value="true">Missing</option></select></div>
          <input value={filters.camera} onChange={(e) => updateFilter('camera', e.target.value)} placeholder="Camera / lens metadata (placeholder)" className="w-full rounded-lg border border-white/10 bg-slate-900 px-3 py-2" />
          <button onClick={() => runSearch()} disabled={loading} className="w-full rounded-xl bg-cyan-400 px-4 py-3 font-bold text-slate-950 hover:bg-cyan-300 disabled:opacity-60">Search library</button>
        </section>
      </aside>

      <section className="flex-1 p-6">
        <header className="mb-5 flex items-center justify-between"><div><p className="text-sm text-slate-400">{notice}</p><p className="text-xs text-slate-500">DB: {appInfo?.database_path ?? 'not connected'}</p></div><div className="rounded-xl border border-white/10 bg-slate-900 p-1"><button onClick={() => setView('grid')} className={`rounded-lg px-3 py-2 ${view === 'grid' ? 'bg-cyan-400 text-slate-950' : ''}`}>Grid</button><button onClick={() => setView('list')} className={`rounded-lg px-3 py-2 ${view === 'list' ? 'bg-cyan-400 text-slate-950' : ''}`}>List</button></div></header>
        <div className="mb-5 grid grid-cols-4 gap-4">{[['Items', counts.total], ['Images', counts.images], ['Videos', counts.videos], ['Missing', counts.missing]].map(([k,v]) => <div key={k} className="rounded-2xl border border-white/10 bg-white/[0.05] p-4"><p className="text-sm text-slate-400">{k}</p><p className="text-3xl font-bold">{v}</p></div>)}</div>
        {scan && <div className="mb-5 rounded-2xl border border-emerald-400/20 bg-emerald-400/10 p-4 text-sm text-emerald-100">Scanned {scan.scanned_files} files, imported/updated {scan.imported_or_updated}, skipped {scan.skipped_files}, missing marked {scan.missing_marked}.</div>}
        <div className={view === 'grid' ? 'grid grid-cols-2 gap-4 lg:grid-cols-3 xl:grid-cols-4' : 'space-y-3'}>
          {items.map((item) => <button key={item.id} onClick={() => openItem(item)} className={`overflow-hidden rounded-2xl border border-white/10 bg-slate-900/80 text-left shadow-xl hover:border-cyan-300/60 ${view === 'list' ? 'flex w-full items-center gap-4 p-3' : ''}`}>{item.media_type === 'image' ? <img src={mediaUrl(item)} className={view === 'grid' ? 'h-44 w-full object-cover' : 'h-20 w-28 rounded-xl object-cover'} /> : <div className={view === 'grid' ? 'flex h-44 items-center justify-center bg-slate-800' : 'flex h-20 w-28 items-center justify-center rounded-xl bg-slate-800'}>▶</div>}<div className="p-3"><p className="truncate font-semibold">{item.file_name}</p><p className="text-xs text-slate-400">{item.media_type} · {formatBytes(item.size_bytes)} · {formatDate(item.modified_at)}</p></div></button>)}
        </div>
      </section>
    </div>

    {setupOpen && <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/70 p-6"><div className="w-full max-w-3xl rounded-3xl border border-white/10 bg-slate-950 p-6 shadow-2xl"><div className="flex items-start justify-between"><div><p className="text-xs uppercase tracking-[0.35em] text-cyan-300">Setup wizard</p><h2 className="mt-2 text-2xl font-bold">Folders, provider, privacy</h2></div><button onClick={() => setSetupOpen(false)} className="text-slate-400 hover:text-white">✕</button></div><div className="mt-6 grid gap-5 md:grid-cols-3"><section className="md:col-span-2 rounded-2xl bg-white/[0.04] p-4"><h3 className="font-semibold">1. Media folders</h3><p className="text-sm text-slate-400">Native folder picker is not wired yet; paste absolute paths manually.</p><div className="mt-3 flex gap-2"><input value={folderInput} onChange={(e) => setFolderInput(e.target.value)} placeholder="/Users/you/Pictures" className="flex-1 rounded-xl border border-white/10 bg-slate-900 px-3 py-2"/><button onClick={addFolder} className="rounded-xl bg-cyan-400 px-4 font-bold text-slate-950">Add</button></div><div className="mt-3 space-y-2">{folders.map((f) => <div key={f} className="rounded-lg bg-slate-900 px-3 py-2 text-sm">{f}</div>)}</div></section><section className="rounded-2xl bg-white/[0.04] p-4"><h3 className="font-semibold">2. Provider</h3><select value={provider} onChange={(e) => setProvider(e.target.value)} className="mt-3 w-full rounded-xl border border-white/10 bg-slate-900 px-3 py-2"><option>Local metadata + filename search</option><option disabled>Local embeddings (coming soon)</option><option disabled>Cloud captions (disabled)</option></select><label className="mt-4 flex items-center gap-2 text-sm"><input type="checkbox" checked={privateMode} onChange={(e) => setPrivateMode(e.target.checked)} /> Keep processing local/private</label></section></div><div className="mt-6 flex justify-end gap-3"><button onClick={() => setSetupOpen(false)} className="rounded-xl border border-white/10 px-4 py-2">Later</button><button onClick={() => scanFolders()} className="rounded-xl bg-cyan-400 px-5 py-2 font-bold text-slate-950">Scan now</button></div></div></div>}

    {selected && <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6" onClick={() => setSelected(null)}><div onClick={(e) => e.stopPropagation()} className="max-h-[92vh] w-full max-w-5xl overflow-hidden rounded-3xl border border-white/10 bg-slate-950 shadow-2xl"><div className="flex items-center justify-between border-b border-white/10 p-4"><div><h2 className="font-bold">{selected.file_name}</h2><p className="text-xs text-slate-400">{selected.path}</p></div><button onClick={() => setSelected(null)} className="text-slate-400 hover:text-white">✕</button></div><div className="grid gap-4 p-4 lg:grid-cols-[1fr_280px]">{selected.media_type === 'video' ? <video src={mediaUrl(selected)} controls className="max-h-[70vh] w-full rounded-2xl bg-black" /> : <img src={mediaUrl(selected)} className="max-h-[70vh] w-full rounded-2xl object-contain bg-black" />}<aside className="space-y-3 rounded-2xl bg-white/[0.04] p-4 text-sm"><p><b>Type:</b> {selected.media_type}</p><p><b>Size:</b> {formatBytes(selected.size_bytes)}</p><p><b>Modified:</b> {formatDate(selected.modified_at)}</p><p><b>Camera:</b> {selected.camera_make ?? '—'} {selected.camera_model ?? ''}</p><p><b>GPS:</b> {selected.latitude ?? '—'}, {selected.longitude ?? '—'}</p><p><b>Status:</b> {selected.missing ? 'Missing' : 'Available'}</p></aside></div></div></div>}
  </main>;
}

export default App;
