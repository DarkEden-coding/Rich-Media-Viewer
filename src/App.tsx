import { convertFileSrc, invoke } from '@tauri-apps/api/core';
import { useEffect, useMemo, useState } from 'react';

type AppInfo = { data_dir: string; database_path: string };
type MediaItem = { id: number; path: string; file_name: string; extension: string | null; media_type: 'image' | 'video' | string; size_bytes: number | null; created_at: number | null; modified_at: number | null; imported_at: number; missing: boolean; camera_make: string | null; camera_model: string | null; latitude: number | null; longitude: number | null; };
type ScanSummary = { scanned_files: number; imported_or_updated: number; skipped_files: number; missing_marked: number; errors: string[] };
type ViewMode = 'grid' | 'list';

const emptyFilters = { query: '', mediaType: '', missing: '', from: '', to: '', lat: '', lng: '', radius: '', camera: '', person: '' };
const icons = { folder: '▱', search: '⌕', image: '▧', video: '▻', missing: '?', items: '□', calendar: '◷', pin: '⌖', mountain: '△', people: '♙', camera: '▣', grid: '▦', list: '☷', box: '▱' };

function formatBytes(bytes?: number | null) { if (!bytes) return '—'; const units = ['B', 'KB', 'MB', 'GB', 'TB']; let size = bytes; let unit = 0; while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit += 1; } return `${size.toFixed(unit ? 1 : 0)} ${units[unit]}`; }
function formatDate(seconds?: number | null) { return seconds ? new Date(seconds * 1000).toLocaleDateString() : 'Unknown'; }
function mediaUrl(item: MediaItem) { try { return convertFileSrc(item.path); } catch { return `file://${item.path}`; } }

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

  const counts = useMemo(() => ({ total: items.length, images: items.filter((i) => i.media_type === 'image').length, videos: items.filter((i) => i.media_type === 'video').length, missing: items.filter((i) => i.missing).length }), [items]);
  const updateFilter = (key: keyof typeof emptyFilters, value: string) => setFilters((f) => ({ ...f, [key]: value }));

  async function runSearch(next = filters) { setLoading(true); try { const result = await invoke<MediaItem[]>('search_media', { filter: { query: next.query || undefined, media_type: next.mediaType || undefined, missing: next.missing === '' ? undefined : next.missing === 'true', limit: 120, offset: 0 } }); setItems(result); setNotice(`${result.length} media items loaded`); } catch (error) { setNotice(`Search unavailable: ${String(error)}`); } finally { setLoading(false); } }
  useEffect(() => { invoke<AppInfo>('initialize_app').then((info) => { setAppInfo(info); setNotice('Library database ready'); return runSearch(); }).catch((error) => setNotice(`Running in preview mode: ${String(error)}`)); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, []);
  async function addFolder() { const path = folderInput.trim(); if (!path) return; setFolders((prev) => Array.from(new Set([...prev, path]))); setFolderInput(''); }
  async function scanFolders(paths = folders) { if (!paths.length) { setNotice('Add at least one folder path before scanning.'); return; } setLoading(true); setNotice('Scanning library…'); try { const summary = await invoke<ScanSummary>('scan_library', { paths }); setScan(summary); setSetupOpen(false); setNotice(`Scan complete: ${summary.imported_or_updated} imported/updated`); await runSearch(); } catch (error) { setNotice(`Scan failed: ${String(error)}`); } finally { setLoading(false); } }
  async function openItem(item: MediaItem) { setSelected(item); try { setSelected((await invoke<MediaItem | null>('get_media_item', { id: item.id })) ?? item); } catch { /* keep optimistic item */ } }

  const statData = [[icons.folder, 'Items', counts.total], [icons.image, 'Images', counts.images], [icons.video, 'Videos', counts.videos], [icons.missing, 'Missing', counts.missing]] as const;

  return <main className="app-shell">
    <div className="traffic-lights"><span /><span /><span /></div>
    <div className="flex h-[calc(100vh-35px)] min-h-0">
      <aside className="w-[450px] shrink-0 px-4 pb-8">
        <div className="glass-panel h-full overflow-y-auto rounded-[13px] p-6">
          <p className="text-[13px] font-black uppercase tracking-[0.42em] text-[#4da8ff]">Private library</p>
          <h1 className="mt-3 text-[34px] font-black leading-none tracking-[-0.04em]">Rich Media Viewer</h1>
          <p className="mt-4 text-[17px] text-slate-400">Semantic-ready search shell with local-first indexing.</p>
          <button onClick={() => setSetupOpen(true)} className="mt-7 flex w-full items-center justify-between rounded-lg border border-slate-500/25 bg-slate-950/20 px-4 py-4 text-left text-[16px] font-extrabold hover:border-blue-400/60"><span><span className="mr-4 text-2xl text-blue-400">{icons.folder}</span>Setup folders / providers</span><span className="text-2xl text-slate-300">›</span></button>

          <section className="glass-panel mt-6 rounded-xl p-4">
            <label><span className="label">Semantic query</span><div className="relative"><span className="absolute left-4 top-1/2 -translate-y-1/2 text-2xl text-slate-300">{icons.search}</span><input value={filters.query} onChange={(e) => updateFilter('query', e.target.value)} placeholder="sunset beach, dog, receipt..." className="field pl-12 text-[16px]" /></div></label>
            <div className="mt-6"><span className="label">Date range</span><div className="grid grid-cols-2 gap-5"><label><span className="sub-label">From</span><input type="date" value={filters.from} onChange={(e) => updateFilter('from', e.target.value)} className="field" /></label><label><span className="sub-label">To</span><input type="date" value={filters.to} onChange={(e) => updateFilter('to', e.target.value)} className="field" /></label></div></div>
            <div className="mt-5"><span className="label">Location</span><div className="grid grid-cols-3 gap-4"><input value={filters.lat} onChange={(e) => updateFilter('lat', e.target.value)} placeholder="⌖   Lat" className="field"/><input value={filters.lng} onChange={(e) => updateFilter('lng', e.target.value)} placeholder="⌖   Lng" className="field"/><input value={filters.radius} onChange={(e) => updateFilter('radius', e.target.value)} placeholder="△   Mi" className="field"/></div></div>
            <label className="mt-5 block"><span className="label">People</span><select value={filters.person} onChange={(e) => updateFilter('person', e.target.value)} className="select-field"><option value="">♙   People selector (coming soon)</option><option>Unassigned faces</option></select></label>
            <div className="mt-5 grid grid-cols-2 gap-5"><label><span className="label">Media type</span><select value={filters.mediaType} onChange={(e) => updateFilter('mediaType', e.target.value)} className="select-field"><option value="">▧   All media</option><option value="image">Images</option><option value="video">Videos</option></select></label><label><span className="label">Status</span><select value={filters.missing} onChange={(e) => updateFilter('missing', e.target.value)} className="select-field"><option value="">◉   Any status</option><option value="false">Available</option><option value="true">Missing</option></select></label></div>
            <label className="mt-5 block"><span className="label">Metadata</span><input value={filters.camera} onChange={(e) => updateFilter('camera', e.target.value)} placeholder="▣   Camera / lens metadata (placeholder)" className="field" /></label>
            <button onClick={() => runSearch()} disabled={loading} className="primary-btn mt-7 flex w-full items-center justify-center gap-4 px-4 py-4 text-[17px]"><span className="text-3xl leading-none">{icons.search}</span>Search library</button>
          </section>
        </div>
      </aside>

      <section className="flex-1 overflow-y-auto pl-7 pr-10 pt-8">
        <header className="mb-7 flex items-start justify-between"><div><p className="text-[18px] font-semibold underline decoration-slate-300/70 underline-offset-2">{notice}</p><p className="mt-2 text-[16px] text-slate-400">DB: {appInfo?.database_path ?? 'not connected'}</p></div><div className="glass-panel rounded-lg p-1"><button onClick={() => setView('grid')} className={`rounded-md px-5 py-3 text-[16px] font-bold ${view === 'grid' ? 'primary-btn' : 'text-slate-200'}`}>{icons.grid} &nbsp;Grid</button><button onClick={() => setView('list')} className={`rounded-md px-5 py-3 text-[16px] font-bold ${view === 'list' ? 'primary-btn' : 'text-slate-200'}`}>{icons.list} &nbsp;List</button></div></header>
        <div className="mb-7 grid grid-cols-4 gap-5">{statData.map(([icon,k,v]) => <div key={k} className="stat-card flex items-center gap-5 p-5"><span className="icon-badge text-3xl">{icon}</span><div><p className="text-[16px] text-slate-300">{k}</p><p className="mt-1 text-[32px] font-black">{v}</p></div></div>)}</div>
        {scan && <div className="mb-5 rounded-xl border border-emerald-400/20 bg-emerald-400/10 p-4 text-sm text-emerald-100">Scanned {scan.scanned_files} files, imported/updated {scan.imported_or_updated}, skipped {scan.skipped_files}, missing marked {scan.missing_marked}.</div>}
        <div className="divider-line -mx-10 mb-0" />
        {items.length === 0 ? <div className="flex min-h-[590px] flex-col items-center justify-center text-center"><div className="empty-orb text-8xl">{icons.box}</div><h2 className="mt-5 text-2xl font-black">Your media library is ready</h2><p className="mt-4 text-[17px] leading-7 text-slate-400">Use the filters on the left to search your library.<br />Results will appear here.</p></div> : <div className={view === 'grid' ? 'grid grid-cols-2 gap-4 lg:grid-cols-3 xl:grid-cols-4' : 'space-y-3'}>{items.map((item) => <button key={item.id} onClick={() => openItem(item)} className={`glass-panel overflow-hidden rounded-xl text-left hover:border-blue-400/70 ${view === 'list' ? 'flex w-full items-center gap-4 p-3' : ''}`}>{item.media_type === 'image' ? <img src={mediaUrl(item)} className={view === 'grid' ? 'h-44 w-full object-cover' : 'h-20 w-28 rounded-xl object-cover'} /> : <div className={view === 'grid' ? 'flex h-44 items-center justify-center bg-slate-800' : 'flex h-20 w-28 items-center justify-center rounded-xl bg-slate-800'}>▶</div>}<div className="p-3"><p className="truncate font-semibold">{item.file_name}</p><p className="text-xs text-slate-400">{item.media_type} · {formatBytes(item.size_bytes)} · {formatDate(item.modified_at)}</p></div></button>)}</div>}
      </section>
    </div>

    {setupOpen && <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/70 p-6"><div className="glass-panel w-full max-w-3xl rounded-3xl p-6"><div className="flex items-start justify-between"><div><p className="text-xs uppercase tracking-[0.35em] text-blue-300">Setup wizard</p><h2 className="mt-2 text-2xl font-black">Folders, provider, privacy</h2></div><button onClick={() => setSetupOpen(false)} className="text-slate-400 hover:text-white">✕</button></div><div className="mt-6 grid gap-5 md:grid-cols-3"><section className="glass-panel rounded-2xl p-4 md:col-span-2"><h3 className="font-semibold">1. Media folders</h3><p className="text-sm text-slate-400">Native folder picker is not wired yet; paste absolute paths manually.</p><div className="mt-3 flex gap-2"><input value={folderInput} onChange={(e) => setFolderInput(e.target.value)} placeholder="/Users/you/Pictures" className="field flex-1"/><button onClick={addFolder} className="primary-btn px-4">Add</button></div><div className="mt-3 space-y-2">{folders.map((f) => <div key={f} className="rounded-lg bg-slate-950/60 px-3 py-2 text-sm">{f}</div>)}</div></section><section className="glass-panel rounded-2xl p-4"><h3 className="font-semibold">2. Provider</h3><select value={provider} onChange={(e) => setProvider(e.target.value)} className="select-field mt-3"><option>Local metadata + filename search</option><option disabled>Local embeddings (coming soon)</option><option disabled>Cloud captions (disabled)</option></select><label className="mt-4 flex items-center gap-2 text-sm"><input type="checkbox" checked={privateMode} onChange={(e) => setPrivateMode(e.target.checked)} /> Keep processing local/private</label></section></div><div className="mt-6 flex justify-end gap-3"><button onClick={() => setSetupOpen(false)} className="rounded-xl border border-white/10 px-4 py-2">Later</button><button onClick={() => scanFolders()} className="primary-btn px-5 py-2">Scan now</button></div></div></div>}

    {selected && <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6" onClick={() => setSelected(null)}><div onClick={(e) => e.stopPropagation()} className="glass-panel max-h-[92vh] w-full max-w-5xl overflow-hidden rounded-3xl"><div className="flex items-center justify-between border-b border-white/10 p-4"><div><h2 className="font-bold">{selected.file_name}</h2><p className="text-xs text-slate-400">{selected.path}</p></div><button onClick={() => setSelected(null)} className="text-slate-400 hover:text-white">✕</button></div><div className="grid gap-4 p-4 lg:grid-cols-[1fr_280px]">{selected.media_type === 'video' ? <video src={mediaUrl(selected)} controls className="max-h-[70vh] w-full rounded-2xl bg-black" /> : <img src={mediaUrl(selected)} className="max-h-[70vh] w-full rounded-2xl bg-black object-contain" />}<aside className="glass-panel space-y-3 rounded-2xl p-4 text-sm"><p><b>Type:</b> {selected.media_type}</p><p><b>Size:</b> {formatBytes(selected.size_bytes)}</p><p><b>Modified:</b> {formatDate(selected.modified_at)}</p><p><b>Camera:</b> {selected.camera_make ?? '—'} {selected.camera_model ?? ''}</p><p><b>GPS:</b> {selected.latitude ?? '—'}, {selected.longitude ?? '—'}</p><p><b>Status:</b> {selected.missing ? 'Missing' : 'Available'}</p></aside></div></div></div>}
  </main>;
}

export default App;
