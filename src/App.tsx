import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import {
  memo,
  startTransition,
  useEffect,
  useMemo,
  useRef,
  useState,
  type UIEvent,
} from "react";

type AppInfo = {
  data_dir: string;
  database_path: string;
  index_exists: boolean;
};
type MediaItem = {
  id: number;
  path: string;
  display_path: string | null;
  file_name: string;
  extension: string | null;
  media_type: string;
  size_bytes: number | null;
  created_at: number | null;
  modified_at: number | null;
  imported_at: number;
  missing: boolean;
  camera_make: string | null;
  camera_model: string | null;
  lens_model: string | null;
  captured_at: number | null;
  latitude: number | null;
  longitude: number | null;
  metadata_json: string | null;
};
type ScanSummary = {
  scanned_files: number;
  imported_or_updated: number;
  skipped_files: number;
  missing_marked: number;
  errors: string[];
};
type ScanProgress = {
  phase: string;
  current_path: string | null;
  scanned_files: number;
  imported_or_updated: number;
  skipped_files: number;
  missing_marked: number;
  errors: number;
  discovered_files?: number;
  total_files?: number | null;
  faces_done?: number;
  faces_total?: number | null;
  done: boolean;
};
type Person = {
  id: number;
  name: string;
  created_at: number;
  face_count: number;
};
type SidecarResult = { ok: boolean; stdout: string; stderr: string };
type Face = {
  id: number;
  media_item_id: number;
  person_id: number | null;
  person_name: string | null;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number | null;
  created_at: number;
};
type GeoPoint = {
  latitude: number;
  longitude: number;
};
type ViewMode = "grid" | "list";
type SortOrder = "desc" | "asc";

const emptyFilters = {
  fileNameQuery: "",
  semanticQuery: "",
  mediaType: "",
  missing: "",
  from: "",
  to: "",
  lat: "",
  lng: "",
  radius: "",
  camera: "",
  personId: "",
  personName: "",
  hasGps: "",
  hasCamera: "",
};
const icons = {
  folder: (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="1em"
      height="1em"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z" />
    </svg>
  ),
  search: (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="1em"
      height="1em"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  ),
  image: (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="1em"
      height="1em"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
      <circle cx="9" cy="9" r="2" />
      <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
    </svg>
  ),
  video: (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="1em"
      height="1em"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="m22 8-6 4 6 4V8Z" />
      <rect width="14" height="12" x="2" y="6" rx="2" ry="2" />
    </svg>
  ),
  missing: (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="1em"
      height="1em"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="12" x2="12" y1="8" y2="12" />
      <line x1="12" x2="12.01" y1="16" y2="16" />
    </svg>
  ),
  box: (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="1em"
      height="1em"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z" />
      <path d="m3.3 7 8.7 5 8.7-5" />
      <path d="M12 22V12" />
    </svg>
  ),
  grid: (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="1em"
      height="1em"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect width="7" height="7" x="3" y="3" rx="1" />
      <rect width="7" height="7" x="14" y="3" rx="1" />
      <rect width="7" height="7" x="14" y="14" rx="1" />
      <rect width="7" height="7" x="3" y="14" rx="1" />
    </svg>
  ),
  list: (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="1em"
      height="1em"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="8" x2="21" y1="6" y2="6" />
      <line x1="8" x2="21" y1="12" y2="12" />
      <line x1="8" x2="21" y1="18" y2="18" />
      <line x1="3" x2="3.01" y1="6" y2="6" />
      <line x1="3" x2="3.01" y1="12" y2="12" />
      <line x1="3" x2="3.01" y1="18" y2="18" />
    </svg>
  ),
  user: (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="1em"
      height="1em"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  ),
  sparkles: (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="1em"
      height="1em"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
      <path d="M5 3v4" />
      <path d="M3 5h4" />
    </svg>
  ),
  map: (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="1em"
      height="1em"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M14.1 4.8 9.9 3.2a2 2 0 0 0-1.5.1L4 5.2a2 2 0 0 0-1.2 1.9v11.2a1 1 0 0 0 1.4.9l4.7-2 5.2 2 5.5-2.4a2 2 0 0 0 1.2-1.8V3.8a1 1 0 0 0-1.4-.9l-5.3 2Z" />
      <path d="M9 3v14" />
      <path d="M15 5v14" />
    </svg>
  ),
};
const providers = ["fastembed", "google", "openrouter"];
const fastEmbedEmbeddingModels = [
  "Qdrant/clip-ViT-B-32",
];
const googleEmbeddingModels = [
  "gemini-embedding-2",
  "gemini-embedding-001",
  "text-embedding-004",
];
const openRouterEmbeddingModels = [
  "google/gemini-embedding-2-preview",
  "openai/text-embedding-3-small",
  "openai/text-embedding-3-large",
];
const embeddingModelsByProvider: Record<string, string[]> = {
  fastembed: fastEmbedEmbeddingModels,
  google: googleEmbeddingModels,
  openrouter: openRouterEmbeddingModels,
};
const PAGE_SIZE = 120;

function formatBytes(bytes?: number | null) {
  if (!bytes) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = bytes,
    unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit++;
  }
  return `${size.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}
function formatDate(seconds?: number | null) {
  return seconds ? new Date(seconds * 1000).toLocaleString() : "Unknown";
}
function captureSortTime(item: MediaItem) {
  return item.captured_at ?? item.modified_at ?? item.created_at ?? 0;
}
function compareMediaByDate(a: MediaItem, b: MediaItem, order: SortOrder) {
  const dir = order === "asc" ? 1 : -1;
  const byDate = (captureSortTime(a) - captureSortTime(b)) * dir;
  return byDate || (a.id - b.id) * dir;
}
function mediaMonthLabel(seconds: number | null | undefined) {
  if (!seconds) return "Unknown date";
  return new Date(seconds * 1000).toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
  });
}
function groupMediaByMonth(items: MediaItem[]) {
  const groups: { key: string; label: string; items: MediaItem[] }[] = [];
  for (const item of items) {
    const seconds = captureSortTime(item);
    const key = seconds
      ? new Date(seconds * 1000).toISOString().slice(0, 7)
      : "unknown";
    const last = groups[groups.length - 1];
    if (last?.key === key) {
      last.items.push(item);
    } else {
      groups.push({
        key,
        label: mediaMonthLabel(seconds),
        items: [item],
      });
    }
  }
  return groups;
}
function fileUrl(path: string) {
  try {
    return convertFileSrc(path);
  } catch {
    return `file://${path}`;
  }
}
function mediaUrl(item: MediaItem) {
  return fileUrl(item.display_path || item.path);
}
function dateToEpoch(value: string, end = false) {
  if (!value) return undefined;
  const d = new Date(`${value}T${end ? "23:59:59" : "00:00:00"}`);
  return Number.isNaN(d.getTime()) ? undefined : Math.floor(d.getTime() / 1000);
}
function num(value: string) {
  const n = Number(value);
  return value.trim() === "" || Number.isNaN(n) ? undefined : n;
}
function metaRows(item: MediaItem) {
  return [
    ["Captured", formatDate(item.captured_at)],
    ["Created", formatDate(item.created_at)],
    ["Modified", formatDate(item.modified_at)],
    [
      "Camera",
      `${item.camera_make ?? ""} ${item.camera_model ?? ""}`.trim() || "—",
    ],
    ["Lens", item.lens_model ?? "—"],
    [
      "GPS",
      item.latitude != null && item.longitude != null
        ? `${item.latitude.toFixed(5)}, ${item.longitude.toFixed(5)}`
        : "—",
    ],
  ];
}
function scanPercent(p: ScanProgress) {
  const ft = p.faces_total;
  if (ft != null && ft > 0) {
    const done = p.faces_done ?? 0;
    return Math.min(100, Math.max(0, (done / ft) * 100));
  }
  const completed = p.scanned_files;
  return p.total_files
    ? Math.min(100, Math.max(0, (completed / p.total_files) * 100))
    : 0;
}
function scanProgressNotice(p: ScanProgress) {
  const faceLine =
    p.faces_total != null && p.faces_total > 0
      ? ` · faces ${p.faces_done ?? 0}/${p.faces_total}`
      : "";
  return `${p.phase}: ${p.scanned_files} scanned, ${p.imported_or_updated} imported${faceLine}`;
}
function formatCoord(value: string, fallback = "Not set") {
  const n = Number(value);
  return value.trim() === "" || Number.isNaN(n) ? fallback : n.toFixed(4);
}

function EarthRegionMap({
  points,
  lat,
  lng,
  radius,
  onPick,
}: {
  points: GeoPoint[];
  lat: string;
  lng: string;
  radius: string;
  onPick: (lat: number, lng: number) => void;
}) {
  const mapEl = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const pinLayerRef = useRef<L.LayerGroup | null>(null);
  const selectionLayerRef = useRef<L.LayerGroup | null>(null);
  const centerLat = num(lat);
  const centerLng = num(lng);
  const radiusKm = num(radius);
  useEffect(() => {
    if (!mapEl.current || mapRef.current) return;
    const initialCenter: L.LatLngExpression =
      centerLat != null && centerLng != null
        ? [centerLat, centerLng]
        : points[0]
          ? [points[0].latitude, points[0].longitude]
          : [20, 0];
    const map = L.map(mapEl.current, {
      center: initialCenter,
      zoom: centerLat != null && centerLng != null ? 10 : points.length ? 3 : 2,
      worldCopyJump: true,
      zoomControl: true,
    });
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);
    pinLayerRef.current = L.layerGroup().addTo(map);
    selectionLayerRef.current = L.layerGroup().addTo(map);
    map.on("click", (event: L.LeafletMouseEvent) => {
      onPick(event.latlng.lat, event.latlng.lng);
    });
    mapRef.current = map;
    setTimeout(() => map.invalidateSize(), 0);
    return () => {
      map.remove();
      mapRef.current = null;
      pinLayerRef.current = null;
      selectionLayerRef.current = null;
    };
    // The map click handler intentionally uses the initial onPick callback.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    const layer = pinLayerRef.current;
    if (!layer) return;
    layer.clearLayers();
    points.slice(0, 2500).forEach((point) => {
      L.circleMarker([point.latitude, point.longitude], {
        radius: 4,
        stroke: true,
        color: "#0f172a",
        weight: 1.2,
        fillColor: "#facc15",
        fillOpacity: 0.92,
      }).addTo(layer);
    });
  }, [points]);
  useEffect(() => {
    const layer = selectionLayerRef.current;
    if (!layer) return;
    layer.clearLayers();
    if (centerLat == null || centerLng == null) return;
    const center: L.LatLngExpression = [centerLat, centerLng];
    L.circle(center, {
      radius: Math.max(1, radiusKm ?? 1) * 1000,
      color: "#2563eb",
      weight: 2,
      fillColor: "#60a5fa",
      fillOpacity: 0.16,
    }).addTo(layer);
    L.circleMarker(center, {
      radius: 8,
      color: "#dbeafe",
      weight: 2,
      fillColor: "#2563eb",
      fillOpacity: 1,
    }).addTo(layer);
  }, [centerLat, centerLng, radiusKm]);
  useEffect(() => {
    const map = mapRef.current;
    if (!map || centerLat == null || centerLng == null) return;
    map.panTo([centerLat, centerLng], { animate: true });
  }, [centerLat, centerLng]);
  return (
    <div
      ref={mapEl}
      role="application"
      aria-label="OpenStreetMap location picker"
      className="region-map h-full min-h-[420px] w-full cursor-crosshair rounded-xl"
    />
  );
}

const LazyMediaThumb = memo(function LazyMediaThumb({
  item,
  view,
}: {
  item: MediaItem;
  view: ViewMode;
}) {
  const sizeClass =
    view === "grid" ? "h-44 w-full" : "h-20 w-28 shrink-0 rounded-xl";
  if (item.media_type !== "image")
    return (
      <div
        className={`${view === "grid" ? "flex h-44" : "flex h-20 w-28 shrink-0"} items-center justify-center rounded-xl bg-slate-800`}
      >
        ▶
      </div>
    );

  return (
    <div
      className={`thumb-placeholder relative overflow-hidden bg-slate-800 ${sizeClass}`}
    >
      <img
          src={mediaUrl(item)}
          loading="lazy"
          decoding="async"
          fetchPriority="low"
          onError={(e) =>
            e.currentTarget.parentElement?.classList.add("is-failed")
          }
          className="absolute inset-0 h-full w-full object-cover"
        />
      <div className="thumb-error absolute inset-0 z-10 hidden items-center justify-center px-3 text-center text-xs font-semibold text-slate-400">
        Preview unavailable
      </div>
    </div>
  );
});

function App() {
  const [appInfo, setAppInfo] = useState<AppInfo | null>(null);
  const [indexExists, setIndexExists] = useState(false);
  const [setupOpen, setSetupOpen] = useState(true);
  const [folderInput, setFolderInput] = useState("");
  const [folders, setFolders] = useState<string[]>([]);
  const [provider, setProvider] = useState(
    providers.includes(localStorage.getItem("rmv.provider") || "")
      ? localStorage.getItem("rmv.provider")!
      : "fastembed",
  );
  const [embeddingModel, setEmbeddingModel] = useState(
    (() => {
      const stored = localStorage.getItem("rmv.embeddingModel") || "";
      const models = embeddingModelsByProvider[provider] ?? [];
      return models.includes(stored)
        ? stored
        : (models[0] ?? "Qdrant/clip-ViT-B-32");
    })(),
  );
  const [embeddingImageMaxWidth, setEmbeddingImageMaxWidth] = useState(
    localStorage.getItem("rmv.embeddingImageMaxWidth") || "1024",
  );
  const [filters, setFilters] = useState(emptyFilters);
  const [geoPoints, setGeoPoints] = useState<GeoPoint[]>([]);
  const [regionOpen, setRegionOpen] = useState(false);
  const [regionDraft, setRegionDraft] = useState({
    lat: "",
    lng: "",
    radius: "50",
  });
  const [people, setPeople] = useState<Person[]>([]);
  const [items, setItems] = useState<MediaItem[]>([]);
  const [hasMoreItems, setHasMoreItems] = useState(true);
  const [selected, setSelected] = useState<MediaItem | null>(null);
  const [view, setView] = useState<ViewMode>("grid");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");
  const [loading, setLoading] = useState(false);
  const [scan, setScan] = useState<ScanSummary | null>(null);
  const [scanProgress, setScanProgress] = useState<ScanProgress | null>(null);
  const [semanticSearchOverlayVisible, setSemanticSearchOverlayVisible] =
    useState(false);
  const [notice, setNotice] = useState("Starting Rich Media Viewer…");
  const [rename, setRename] = useState<Record<number, string>>({});
  const [faceSetupOpen, setFaceSetupOpen] = useState(false);
  const [faceItems, setFaceItems] = useState<MediaItem[]>([]);
  const [faceIndex, setFaceIndex] = useState(0);
  const [faces, setFaces] = useState<Face[]>([]);
  const [faceNames, setFaceNames] = useState<Record<number, string>>({});
  const [faceBusy, setFaceBusy] = useState(false);
  const [faceImageSize, setFaceImageSize] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const [faceImageRect, setFaceImageRect] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const faceRequestRef = useRef(0);
  const faceImageRef = useRef<HTMLImageElement | null>(null);
  const searchRequestRef = useRef(0);
  const searchOffsetRef = useRef(0);
  const activeFilterRef = useRef(emptyFilters);
  const activeSortOrderRef = useRef<SortOrder>("desc");
  const loadingMoreRef = useRef(false);
  const pendingScanProgressRef = useRef<ScanProgress | null>(null);
  const scanProgressTimerRef = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );
  const lastScanProgressFlushRef = useRef(0);

  const counts = useMemo(
    () => ({
      total: items.length,
      images: items.filter((i) => i.media_type === "image").length,
      videos: items.filter((i) => i.media_type === "video").length,
      missing: items.filter((i) => i.missing).length,
    }),
    [items],
  );
  const groupedItems = useMemo(() => groupMediaByMonth(items), [items]);
  const updateFilter = (key: keyof typeof emptyFilters, value: string) =>
    setFilters((f) => ({ ...f, [key]: value }));
  const statData = [
    [icons.folder, "Items", counts.total],
    [icons.image, "Images", counts.images],
    [icons.video, "Videos", counts.videos],
    [icons.missing, "Missing", counts.missing],
  ] as const;
  const currentFaceItem = faceItems[faceIndex];
  const visibleFaces = currentFaceItem
    ? faces.filter((f) => f.media_item_id === currentFaceItem.id)
    : [];
  const unnamedFacesHere = visibleFaces.filter((f) => f.person_id == null);
  const namedFacesHere = visibleFaces.filter((f) => f.person_id != null);

  async function loadSettings() {
    const s = await invoke<{ library_folders: string[] }>("get_settings");
    setFolders(s.library_folders || []);
  }
  async function loadPeople() {
    try {
      const p = await invoke<Person[]>("list_people");
      setPeople(p);
      setRename(Object.fromEntries(p.map((x) => [x.id, x.name])));
    } catch {
      /* no people yet */
    }
  }
  async function loadGeoPoints() {
    try {
      setGeoPoints(await invoke<GeoPoint[]>("list_geo_points"));
    } catch {
      setGeoPoints([]);
    }
  }
  function openRegionModal() {
    setRegionDraft({
      lat: filters.lat,
      lng: filters.lng,
      radius: filters.radius || "50",
    });
    setRegionOpen(true);
  }
  function applyRegion() {
    const next = {
      ...filters,
      lat: regionDraft.lat,
      lng: regionDraft.lng,
      radius: regionDraft.radius,
      hasGps: regionDraft.lat && regionDraft.lng ? "true" : filters.hasGps,
    };
    setFilters(next);
    setRegionOpen(false);
    void runSearch(next);
  }
  function buildSearchFilter(next = filters, offset = 0, order = sortOrder) {
    const fq = next.fileNameQuery.trim();
    return {
      query: fq || undefined,
      media_type: next.mediaType || undefined,
      missing: next.missing === "" ? undefined : next.missing === "true",
      from_ts: dateToEpoch(next.from),
      to_ts: dateToEpoch(next.to, true),
      camera: next.camera || undefined,
      lat: num(next.lat),
      lng: num(next.lng),
      radius_km: num(next.radius),
      person_id: num(next.personId),
      person_name: next.personName || undefined,
      has_gps: next.hasGps === "" ? undefined : next.hasGps === "true",
      has_camera: next.hasCamera === "" ? undefined : next.hasCamera === "true",
      sort_order: order,
      limit: PAGE_SIZE,
      offset,
    };
  }
  async function runSearch(next = filters, order = sortOrder) {
    const requestId = ++searchRequestRef.current;
    activeFilterRef.current = next;
    activeSortOrderRef.current = order;
    searchOffsetRef.current = 0;
    loadingMoreRef.current = false;
    setHasMoreItems(true);
    setLoading(true);
    setSemanticSearchOverlayVisible(false);
    let semanticProgressTimer: ReturnType<typeof setTimeout> | null = null;
    const fileQ = next.fileNameQuery.trim();
    const semQ = next.semanticQuery.trim();
    try {
      if (semQ) {
        semanticProgressTimer = setTimeout(() => {
          if (requestId === searchRequestRef.current) {
            setSemanticSearchOverlayVisible(true);
          }
        }, 250);
        try {
          const semantic = await invoke<{ item: MediaItem; score: number }[]>(
            "search_semantic_text",
            {
              query: semQ,
              provider,
              model: embeddingModel,
              limit: 120,
            },
          );
          if (semanticProgressTimer) {
            clearTimeout(semanticProgressTimer);
            semanticProgressTimer = null;
          }
          let result = semantic.map((hit) => hit.item);
          if (fileQ) {
            const fl = fileQ.toLowerCase();
            result = result.filter(
              (item) =>
                item.file_name.toLowerCase().includes(fl) ||
                item.path.toLowerCase().includes(fl),
            );
          }
          result.sort((a, b) => compareMediaByDate(a, b, order));
          if (requestId !== searchRequestRef.current) return;
          setSemanticSearchOverlayVisible(false);
          setItems(result);
          searchOffsetRef.current = result.length;
          setHasMoreItems(false);
          if (fileQ) {
            setNotice(
              result.length
                ? `${result.length} semantic matches after filename filter`
                : "No semantic matches also match the filename filter",
            );
          } else {
            setNotice(
              result.length
                ? `Semantic search found ${result.length} embedded matches`
                : "No close semantic matches for that description",
            );
          }
          return;
        } catch {
          if (semanticProgressTimer) {
            clearTimeout(semanticProgressTimer);
            semanticProgressTimer = null;
          }
          if (!fileQ) {
            if (requestId !== searchRequestRef.current) return;
            setSemanticSearchOverlayVisible(false);
            setItems([]);
            searchOffsetRef.current = 0;
            setHasMoreItems(false);
            setNotice(
              "Semantic search unavailable (embeddings or provider issue)",
            );
            return;
          }
        }
        setSemanticSearchOverlayVisible(false);
      }
      setItems([]);
      const result = await invoke<MediaItem[]>("search_media", {
        filter: buildSearchFilter(next, 0, order),
      });
      if (requestId !== searchRequestRef.current) return;
      searchOffsetRef.current = result.length;
      setItems(result);
      setHasMoreItems(result.length === PAGE_SIZE);
      setNotice(
        fileQ
          ? `${result.length} media items matching filename / path`
          : `${result.length} nearby media items loaded`,
      );
    } catch (error) {
      if (requestId !== searchRequestRef.current) return;
      setNotice(`Search unavailable: ${String(error)}`);
    } finally {
      if (semanticProgressTimer) clearTimeout(semanticProgressTimer);
      if (requestId === searchRequestRef.current) {
        setLoading(false);
        setSemanticSearchOverlayVisible(false);
      }
    }
  }
  async function loadMoreItems() {
    if (loadingMoreRef.current || !hasMoreItems) return;
    const requestId = searchRequestRef.current;
    const offset = searchOffsetRef.current;
    loadingMoreRef.current = true;
    try {
      const result = await invoke<MediaItem[]>("search_media", {
        filter: buildSearchFilter(
          activeFilterRef.current,
          offset,
          activeSortOrderRef.current,
        ),
      });
      if (requestId !== searchRequestRef.current) return;
      searchOffsetRef.current = offset + result.length;
      startTransition(() => setItems((prev) => [...prev, ...result]));
      setHasMoreItems(result.length === PAGE_SIZE);
      if (result.length) {
        setNotice(`${offset + result.length} media items loaded`);
      }
    } catch (error) {
      if (requestId !== searchRequestRef.current) return;
      setNotice(`Search unavailable: ${String(error)}`);
    } finally {
      if (requestId === searchRequestRef.current) loadingMoreRef.current = false;
    }
  }
  function maybeLoadMore(e: UIEvent<HTMLDivElement>) {
    const el = e.currentTarget;
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 1600)
      void loadMoreItems();
  }
  function changeSortOrder(order: SortOrder) {
    setSortOrder(order);
    void runSearch(activeFilterRef.current, order);
  }
  useEffect(() => {
    invoke<AppInfo>("initialize_app")
      .then(async (info) => {
        setAppInfo(info);
        setIndexExists(info.index_exists);
        await loadSettings();
        await loadPeople();
        await loadGeoPoints();
        await runSearch();
        setNotice("Library database ready");
      })
      .catch((e) =>
        setNotice(`Running in preview mode: ${String(e)}`),
      ); /* eslint-disable-next-line react-hooks/exhaustive-deps */
  }, []);
  useEffect(() => {
    const flushScanProgress = () => {
      const progress = pendingScanProgressRef.current;
      if (!progress) return;
      pendingScanProgressRef.current = null;
      scanProgressTimerRef.current = null;
      lastScanProgressFlushRef.current = Date.now();
      setScanProgress(progress);
      setNotice(scanProgressNotice(progress));
    };
    const unlisten = listen<ScanProgress>("scan-progress", (e) => {
      pendingScanProgressRef.current = e.payload;
      if (e.payload.done) {
        if (scanProgressTimerRef.current) {
          clearTimeout(scanProgressTimerRef.current);
          scanProgressTimerRef.current = null;
        }
        flushScanProgress();
        return;
      }
      const elapsed = Date.now() - lastScanProgressFlushRef.current;
      if (elapsed >= 150) {
        if (scanProgressTimerRef.current) {
          clearTimeout(scanProgressTimerRef.current);
          scanProgressTimerRef.current = null;
        }
        flushScanProgress();
      } else if (!scanProgressTimerRef.current) {
        scanProgressTimerRef.current = setTimeout(
          flushScanProgress,
          150 - elapsed,
        );
      }
    });
    return () => {
      if (scanProgressTimerRef.current) {
        clearTimeout(scanProgressTimerRef.current);
        scanProgressTimerRef.current = null;
      }
      unlisten.then((f) => f());
    };
  }, []);
  useEffect(() => {
    if (!faceSetupOpen) return;
    const updateFaceImageRect = () => {
      const image = faceImageRef.current;
      if (!image) return;
      const rect = image.getBoundingClientRect();
      setFaceImageRect({ width: rect.width, height: rect.height });
    };
    updateFaceImageRect();
    window.addEventListener("resize", updateFaceImageRect);
    return () => window.removeEventListener("resize", updateFaceImageRect);
  }, [faceSetupOpen, currentFaceItem?.id]);
  useEffect(() => {
    localStorage.setItem("rmv.provider", provider);
  }, [provider]);
  useEffect(() => {
    localStorage.setItem("rmv.embeddingModel", embeddingModel);
  }, [embeddingModel]);
  useEffect(() => {
    localStorage.setItem("rmv.embeddingImageMaxWidth", embeddingImageMaxWidth);
  }, [embeddingImageMaxWidth]);
  useEffect(() => {
    const models = embeddingModelsByProvider[provider] ?? [];
    if (models.length && !models.includes(embeddingModel))
      setEmbeddingModel(models[0]);
  }, [provider, embeddingModel]);
  async function addFolder(path = folderInput.trim()) {
    if (!path) return;
    const list = await invoke<string[]>("add_library_folder", { path });
    setFolders(list);
    setFolderInput("");
    await invoke("update_settings", { settings: { library_folders: list } });
  }
  async function chooseFolder() {
    const path = await invoke<string | null>("choose_media_folder", {
      path: null,
    });
    if (path) await addFolder(path);
  }
  async function removeFolder(path: string) {
    const list = await invoke<string[]>("remove_library_folder", { path });
    setFolders(list);
    await invoke("update_settings", { settings: { library_folders: list } });
  }
  async function scanFolders(paths = folders) {
    if (!paths.length)
      return setNotice("Add at least one folder path before scanning.");
    setLoading(true);
      setScanProgress({
      phase: "Starting scan",
      current_path: null,
      scanned_files: 0,
      imported_or_updated: 0,
      skipped_files: 0,
      missing_marked: 0,
      errors: 0,
      discovered_files: 0,
      total_files: null,
      faces_done: 0,
      faces_total: null,
      done: false,
    });
    setNotice("Scanning library…");
    try {
      const summary = await invoke<ScanSummary>("scan_library", { paths });
      setScan(summary);
      setIndexExists(true);
      setSetupOpen(false);
      setNotice(
        `Scan complete: ${summary.imported_or_updated} imported/updated`,
      );
      setLoading(false);
      setScanProgress(null);
      await runSearch();
      void loadGeoPoints();
    } catch (e) {
      setNotice(`Scan failed: ${String(e)}`);
    } finally {
      setLoading(false);
      setScanProgress(null);
    }
  }
  async function updateFaceEmbeddings() {
    if (!indexExists) {
      setNotice("Scan the library first to build an index.");
      return;
    }
    setLoading(true);
    setScanProgress({
      phase: "Updating face embeddings",
      current_path: null,
      scanned_files: 0,
      imported_or_updated: 0,
      skipped_files: 0,
      missing_marked: 0,
      errors: 0,
      discovered_files: 0,
      total_files: null,
      faces_done: 0,
      faces_total: null,
      done: false,
    });
    setNotice("Updating face embeddings…");
    try {
      const summary = await invoke<ScanSummary>("update_face_embeddings");
      const errN = summary.errors?.length ?? 0;
      setNotice(
        errN
          ? `Face embeddings updated with ${errN} error(s). Check the index log if faces look wrong.`
          : "Face embeddings updated for all indexed images.",
      );
      await loadPeople();
      await runSearch();
    } catch (e) {
      setNotice(`Face embedding update failed: ${String(e)}`);
    } finally {
      setLoading(false);
      setScanProgress(null);
    }
  }
  async function deleteIndex() {
    if (
      !confirm(
        "Delete the current Rich Media Viewer index database? This removes indexed media records, people/faces, embeddings, and saved folder settings. Original media files will not be deleted.",
      )
    )
      return;
    setIndexExists(false);
    setLoading(true);
    try {
      const info = await invoke<AppInfo>("delete_current_index");
      setAppInfo(info);
      setIndexExists(info.index_exists);
      setFolders([]);
      setItems([]);
      setPeople([]);
      setFaces([]);
      setFaceItems([]);
      setSelected(null);
      setScan(null);
      setScanProgress(null);
      setGeoPoints([]);
      setHasMoreItems(false);
      setNotice(
        "Index database deleted. Add folders and scan to rebuild it.",
      );
    } catch (e) {
      setNotice(`Delete index failed: ${String(e)}`);
    } finally {
      setLoading(false);
    }
  }
  async function openItem(item: MediaItem) {
    setSelected(item);
    try {
      setSelected(
        (await invoke<MediaItem | null>("get_media_item", { id: item.id })) ??
          item,
      );
    } catch {
      /* keep optimistic item */
    }
  }
  async function renamePerson(id: number) {
    await invoke("rename_person", {
      personId: id,
      name: rename[id] || "Unnamed",
    });
    await loadPeople();
    await runSearch();
  }
  async function deletePerson(id: number, displayName: string) {
    if (
      !window.confirm(
        `Remove "${displayName}" from people? Tagged faces will become unnamed; face regions are kept.`,
      )
    ) {
      return;
    }
    try {
      await invoke("delete_person", { personId: id });
      setRename((r) => {
        const next = { ...r };
        delete next[id];
        return next;
      });
      await loadPeople();
      await runSearch();
      setNotice(`Removed "${displayName}" from people.`);
    } catch (e) {
      setNotice(`Failed to remove person: ${String(e)}`);
    }
  }
  async function sidecar(cmd: "generate_embeddings") {
    setLoading(true);
    setScanProgress({
      phase: "Generating embeddings",
      current_path: null,
      scanned_files: 0,
      imported_or_updated: 0,
      skipped_files: 0,
      missing_marked: 0,
      errors: 0,
      discovered_files: 0,
      total_files: null,
      faces_done: 0,
      faces_total: null,
      done: false,
    });
    const imageMaxWidth = num(embeddingImageMaxWidth);
    try {
      const res = await invoke<SidecarResult>(cmd, {
        mediaIds: null,
        provider,
        model: embeddingModel,
        imageMaxWidth:
          (provider === "fastembed" ||
            provider === "google" ||
            provider === "openrouter") &&
          imageMaxWidth &&
          imageMaxWidth > 0
            ? Math.floor(imageMaxWidth)
            : null,
      });
      let detail = "";
      try {
        const parsed = JSON.parse(res.stdout);
        const data = parsed.data;
        if (data && typeof data.embedded === "number") {
          detail = `: ${data.embedded} embedded, ${data.skipped ?? 0} skipped`;
        }
      } catch {
        detail = res.stderr ? `: ${res.stderr}` : "";
      }
      setNotice(`${cmd}: ${res.ok ? "complete" : "failed"}${detail}`);
      await loadPeople();
    } catch (e) {
      setNotice(`${cmd} failed: ${String(e)}`);
    } finally {
      setLoading(false);
      setScanProgress(null);
    }
  }
  async function openFaceSetup() {
    setFaceSetupOpen(true);
    setSetupOpen(false);
    setFaceBusy(true);
    setFaces([]);
    setFaceNames({});
    setFaceImageSize(null);
    setFaceImageRect(null);
    setNotice("Guided face setup opened. Faces process one image at a time.");
    try {
      const result = await invoke<MediaItem[]>("search_media", {
        filter: { media_type: "image", missing: false, limit: 500, offset: 0 },
      });
      setFaceItems(result);
      let foundFace = false;
      for (let i = 0; i < result.length; i += 1) {
        setFaceIndex(i);
        const faceCount = await processFaceImage(result[i].id);
        if (faceCount === null || faceCount > 0) {
          foundFace = faceCount !== null;
          break;
        }
      }
      if (result.length && !foundFace) setNotice("No images with faces found.");
    } catch (e) {
      setNotice(`Face setup unavailable: ${String(e)}`);
    } finally {
      setFaceBusy(false);
    }
  }
  async function applyFaceListForMedia(mediaId: number, requestId: number) {
    const f = await invoke<Face[]>("list_faces", {
      mediaItemId: mediaId,
      personId: null,
    });
    const mediaFaces = f.filter((x) => x.media_item_id === mediaId);
    if (requestId !== faceRequestRef.current) return mediaFaces.length;
    setFaces(mediaFaces);
    setFaceNames(
      Object.fromEntries(mediaFaces.map((x) => [x.id, x.person_name || ""])),
    );
    return mediaFaces.length;
  }
  async function processFaceImage(mediaId: number) {
    const requestId = ++faceRequestRef.current;
    setFaces([]);
    setFaceNames({});
    setFaceImageSize(null);
    setFaceImageRect(null);
    setFaceBusy(true);
    try {
      await invoke<SidecarResult>("process_face_setup_image", { mediaId });
      const faceCount = await applyFaceListForMedia(mediaId, requestId);
      await loadPeople();
      return faceCount;
    } catch (e) {
      if (requestId === faceRequestRef.current) {
        setNotice(`Face processing failed: ${String(e)}`);
        setFaces([]);
        setFaceNames({});
      }
      return null;
    } finally {
      if (requestId === faceRequestRef.current) setFaceBusy(false);
    }
  }
  async function goFace(delta: number) {
    if (!faceItems.length) return;

    const step = delta >= 0 ? 1 : -1;
    let next = Math.min(Math.max(faceIndex + step, 0), faceItems.length - 1);

    while (next >= 0 && next < faceItems.length) {
      setFaceIndex(next);
      const item = faceItems[next];
      const faceCount = item ? await processFaceImage(item.id) : null;

      if (delta < 0 || faceCount === null || faceCount > 0) return;

      next += step;
    }

    setNotice("No more images with faces found.");
  }
  async function saveFaceName(faceId: number) {
    const name = (faceNames[faceId] || "").trim();
    if (!name) return;
    const requestId = ++faceRequestRef.current;
    setFaceBusy(true);
    try {
      const matched = await invoke<number>("name_face", { faceId, name });
      setNotice(
        `Named face and matched ${matched} similar unnamed face(s) in other photos.`,
      );
      const item = faceItems[faceIndex];
      if (item) await applyFaceListForMedia(item.id, requestId);
      await loadPeople();
    } catch (e) {
      setNotice(`Naming face failed: ${String(e)}`);
    } finally {
      if (requestId === faceRequestRef.current) setFaceBusy(false);
    }
  }
  return (
    <main className="app-shell flex h-dvh min-h-0 flex-col">
      <div className="flex min-h-0 flex-1">
        <aside className="w-[min(420px,40vw)] min-w-[300px] shrink-0 px-3 pb-6 pt-4">
          <div className="glass-panel flex h-full min-h-0 flex-col overflow-hidden rounded-[13px]">
            <div className="overflow-y-auto px-6 pb-6 pt-8">
              <div className="flex items-baseline justify-between gap-3">
                <p className="text-[13px] font-black uppercase tracking-[0.42em] text-[#4da8ff]">
                  Private library
                </p>
                <span className="shrink-0 text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">
                  Local index
                </span>
              </div>
              <h1 className="mt-4 text-[32px] font-black leading-tight tracking-[-0.04em]">
                Rich Media Viewer
              </h1>
              <p className="mt-3 text-[15px] leading-snug text-slate-400">
                Local-first media indexing with optional faces and embeddings.
              </p>
              <button
                onClick={() => setSetupOpen(true)}
                className="mt-6 flex w-full items-center justify-between rounded-lg border border-slate-500/25 bg-slate-950/20 px-4 py-3.5 text-left text-[15px] font-extrabold hover:border-blue-400/60"
              >
                <span className="flex items-center gap-2">
                  <span className="text-lg">{icons.folder}</span> Setup folders
                  / providers
                </span>
                <span>›</span>
              </button>
              <section className="glass-panel mt-5 rounded-xl p-4">
                <label>
                  <span className="label">Describe content (semantic)</span>
                  <input
                    value={filters.semanticQuery}
                    onChange={(e) =>
                      updateFilter("semanticQuery", e.target.value)
                    }
                    placeholder="sunset beach, dog, receipt…"
                    className="field"
                  />
                </label>
                <label className="mt-4 block">
                  <span className="label">File name / path</span>
                  <input
                    value={filters.fileNameQuery}
                    onChange={(e) =>
                      updateFilter("fileNameQuery", e.target.value)
                    }
                    placeholder="IMG_2024, vacation, .mp4…"
                    className="field"
                  />
                </label>
                <p className="mt-2 text-xs text-slate-400">
                  Semantic search uses image embeddings only. Filename search
                  matches the indexed file name and full path in the library
                  database (SQL). With both filled, semantic hits are narrowed
                  by the filename filter.
                </p>
                <div className="mt-5 grid grid-cols-2 gap-4">
                  <label>
                    <span className="sub-label">From</span>
                    <input
                      type="date"
                      value={filters.from}
                      onChange={(e) => updateFilter("from", e.target.value)}
                      className="field"
                    />
                  </label>
                  <label>
                    <span className="sub-label">To</span>
                    <input
                      type="date"
                      value={filters.to}
                      onChange={(e) => updateFilter("to", e.target.value)}
                      className="field"
                    />
                  </label>
                </div>
                <div className="mt-5 rounded-lg border border-white/10 bg-slate-950/35 p-3">
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <span className="sub-label">Search region</span>
                      <p className="truncate text-xs text-slate-400">
                        {filters.lat && filters.lng
                          ? `${formatCoord(filters.lat)}, ${formatCoord(filters.lng)} · ${filters.radius || "0"} km`
                          : "No map region selected"}
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={openRegionModal}
                      className="flex shrink-0 items-center gap-2 rounded-lg bg-white/10 px-3 py-2 text-sm font-bold text-slate-100 transition-colors hover:bg-white/15"
                    >
                      <span className="text-base">{icons.map}</span>
                      Search region
                    </button>
                  </div>
                </div>
                <div className="mt-5 grid grid-cols-2 gap-4">
                  <select
                    value={filters.personId}
                    onChange={(e) => updateFilter("personId", e.target.value)}
                    className="select-field min-w-0"
                  >
                    <option value="">All people</option>
                    {people.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name} ({p.face_count})
                      </option>
                    ))}
                  </select>
                  <input
                    value={filters.personName}
                    onChange={(e) => updateFilter("personName", e.target.value)}
                    placeholder="Person name contains"
                    className="field min-w-0"
                  />
                </div>
                <div className="mt-5 grid grid-cols-2 gap-4">
                  <select
                    value={filters.mediaType}
                    onChange={(e) => updateFilter("mediaType", e.target.value)}
                    className="select-field min-w-0"
                  >
                    <option value="">All media</option>
                    <option value="image">Images</option>
                    <option value="video">Videos</option>
                  </select>
                  <select
                    value={filters.missing}
                    onChange={(e) => updateFilter("missing", e.target.value)}
                    className="select-field min-w-0"
                  >
                    <option value="">Any status</option>
                    <option value="false">Available</option>
                    <option value="true">Missing</option>
                  </select>
                </div>
                <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-3">
                  <input
                    value={filters.camera}
                    onChange={(e) => updateFilter("camera", e.target.value)}
                    placeholder="Camera"
                    className="field min-w-0 sm:col-span-1"
                  />
                  <select
                    value={filters.hasGps}
                    onChange={(e) => updateFilter("hasGps", e.target.value)}
                    className="select-field min-w-0"
                  >
                    <option value="">GPS any</option>
                    <option value="true">Has GPS</option>
                    <option value="false">No GPS</option>
                  </select>
                  <select
                    value={filters.hasCamera}
                    onChange={(e) => updateFilter("hasCamera", e.target.value)}
                    className="select-field min-w-0"
                  >
                    <option value="">Camera any</option>
                    <option value="true">Has camera</option>
                    <option value="false">No camera</option>
                  </select>
                </div>
                <button
                  onClick={() => runSearch()}
                  disabled={loading}
                  className="primary-btn mt-6 w-full px-4 py-3.5 flex items-center justify-center gap-2"
                >
                  <span className="text-lg">{icons.search}</span> Search library
                </button>
              </section>
            </div>
          </div>
        </aside>
        <section className="flex min-h-0 flex-1 flex-col overflow-hidden pl-5 pr-8 pt-5 pb-4">
          <header className="mb-5 flex shrink-0 flex-wrap items-start justify-between gap-4">
            <div className="min-w-0 flex-1 pt-1">
              <p className="text-[17px] font-semibold text-slate-100 underline decoration-slate-500/40 underline-offset-4">
                {notice}
              </p>
              <p
                className="mt-2 truncate font-mono text-[12px] leading-relaxed text-slate-500"
                title={appInfo?.database_path}
              >
                {appInfo?.database_path
                  ? `DB · ${appInfo.database_path}`
                  : "DB · not connected"}
              </p>
            </div>
            <div className="flex shrink-0 flex-wrap items-center gap-2">
              <div className="glass-panel rounded-lg p-1 flex items-center">
                <button
                  type="button"
                  onClick={() => changeSortOrder("desc")}
                  className={`rounded-md px-4 py-2.5 text-[15px] font-bold ${sortOrder === "desc" ? "primary-btn" : "text-slate-200"}`}
                  title="Newest first"
                >
                  Newest
                </button>
                <button
                  type="button"
                  onClick={() => changeSortOrder("asc")}
                  className={`rounded-md px-4 py-2.5 text-[15px] font-bold ${sortOrder === "asc" ? "primary-btn" : "text-slate-200"}`}
                  title="Oldest first"
                >
                  Oldest
                </button>
              </div>
              <div className="glass-panel rounded-lg p-1 flex items-center">
                <button
                  type="button"
                  onClick={() => setView("grid")}
                  className={`flex items-center gap-2 rounded-md px-4 py-2.5 text-[15px] font-bold ${view === "grid" ? "primary-btn" : "text-slate-200"}`}
                >
                  <span className="text-lg">{icons.grid}</span> Grid
                </button>
                <button
                  type="button"
                  onClick={() => setView("list")}
                  className={`flex items-center gap-2 rounded-md px-4 py-2.5 text-[15px] font-bold ${view === "list" ? "primary-btn" : "text-slate-200"}`}
                >
                  <span className="text-lg">{icons.list}</span> List
                </button>
              </div>
            </div>
          </header>
          <div className="mb-5 grid shrink-0 grid-cols-2 gap-3 lg:grid-cols-4">
            {statData.map(([icon, k, v]) => (
              <div
                key={k}
                className="stat-card flex items-center gap-4 px-5 py-6"
              >
                <span className="icon-badge text-3xl">{icon}</span>
                <div className="min-w-0">
                  <p className="text-[14px] text-slate-300">{k}</p>
                  <p className="mt-0.5 text-[30px] font-black tabular-nums leading-none">
                    {v}
                  </p>
                </div>
              </div>
            ))}
          </div>
          {scanProgress && loading && (
            <div className="mb-4 shrink-0 rounded-xl border border-blue-400/20 bg-blue-400/10 p-4 text-sm text-blue-100">
              <div className="mb-2 flex justify-between gap-3">
                <span>{scanProgress.phase}</span>
                <span>
                  {(scanProgress.total_files != null &&
                    scanProgress.total_files > 0) ||
                  (scanProgress.faces_total != null &&
                    scanProgress.faces_total > 0)
                    ? `${scanPercent(scanProgress).toFixed(1)}%`
                    : `${scanProgress.discovered_files ?? 0} discovered`}
                </span>
              </div>
              <div className="mb-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-blue-100/80 sm:grid-cols-6">
                <span>{scanProgress.discovered_files ?? 0} discovered</span>
                <span>{scanProgress.total_files ?? "…"} total</span>
                <span>{scanProgress.scanned_files} indexed</span>
                <span>{scanProgress.imported_or_updated} imported</span>
                <span>
                  {scanProgress.skipped_files} skipped · {scanProgress.errors}{" "}
                  errors
                </span>
                {(scanProgress.faces_total ?? 0) > 0 && (
                  <span className="sm:col-span-2">
                    Faces {scanProgress.faces_done ?? 0} /{" "}
                    {scanProgress.faces_total}
                  </span>
                )}
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-slate-900">
                <div
                  className={`h-full rounded-full bg-blue-400 ${
                    (scanProgress.total_files != null &&
                      scanProgress.total_files > 0) ||
                    (scanProgress.faces_total != null &&
                      scanProgress.faces_total > 0)
                      ? ""
                      : "w-1/3 animate-pulse"
                  }`}
                  style={
                    (scanProgress.total_files != null &&
                      scanProgress.total_files > 0) ||
                    (scanProgress.faces_total != null &&
                      scanProgress.faces_total > 0)
                      ? { width: `${scanPercent(scanProgress)}%` }
                      : undefined
                  }
                />
              </div>
              {scanProgress.current_path && (
                <p className="mt-2 truncate font-mono text-xs text-blue-200/70">
                  {scanProgress.current_path}
                </p>
              )}
            </div>
          )}
          {scan && (
            <div className="mb-4 shrink-0 rounded-xl border border-emerald-400/20 bg-emerald-400/10 p-4 text-sm text-emerald-100">
              Scanned {scan.scanned_files}, imported/updated{" "}
              {scan.imported_or_updated}, skipped {scan.skipped_files}, missing
              marked {scan.missing_marked}.
            </div>
          )}
          <div className="divider-line -ml-5 -mr-8 mb-3 shrink-0" />
          <div className="relative min-h-0 flex flex-1 flex-col">
            {semanticSearchOverlayVisible && (
              <div className="absolute inset-0 z-20 flex items-start justify-center overflow-hidden bg-slate-950/55 p-6 pt-10 backdrop-blur-[2px]">
                <div
                  role="status"
                  aria-live="polite"
                  aria-busy="true"
                  className="glass-panel flex flex-col items-center gap-4 rounded-2xl border border-blue-400/25 px-8 py-7 shadow-xl"
                >
                  <div
                    className="h-11 w-11 shrink-0 animate-spin rounded-full border-2 border-blue-400/25 border-t-blue-400"
                    aria-hidden
                  />
                  <p className="text-center text-base font-semibold text-slate-100">
                    Searching Images
                  </p>
                </div>
              </div>
            )}
            <div
              ref={scrollerRef}
              onScroll={maybeLoadMore}
              className="flex min-h-0 flex-1 flex-col overflow-y-auto"
            >
            {items.length === 0 ? (
              <div className="flex flex-1 flex-col items-center justify-center gap-4 py-10 text-center">
                <div className="empty-orb text-7xl">{icons.box}</div>
                <h2 className="text-xl font-black text-slate-100 sm:text-2xl">
                  Your media library is ready
                </h2>
                <p className="max-w-md px-4 text-sm text-slate-500">
                  Add folders in setup, run a scan, then search or browse your
                  grid.
                </p>
              </div>
            ) : (
              <div className="space-y-7 pb-4">
                {groupedItems.map((group) => (
                  <section key={group.key}>
                    <div className="mb-3 flex items-center gap-3">
                      <h2 className="text-sm font-black uppercase tracking-[0.22em] text-slate-300">
                        {group.label}
                      </h2>
                      <span className="h-px flex-1 bg-white/10" />
                      <span className="text-xs font-semibold text-slate-500">
                        {group.items.length}
                      </span>
                    </div>
                    <div
                      className={
                        view === "grid"
                          ? "grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-3 xl:grid-cols-4"
                          : "space-y-3"
                      }
                    >
                      {group.items.map((item) => (
                        <button
                          key={item.id}
                          onClick={() => openItem(item)}
                          className={`media-card glass-panel overflow-hidden rounded-xl text-left hover:border-blue-400/70 ${view === "list" ? "flex w-full items-center gap-4 p-3" : ""}`}
                        >
                          <LazyMediaThumb item={item} view={view} />
                          <div className="min-w-0 p-3">
                            <p className="truncate font-semibold">
                              {item.file_name}
                            </p>
                            <p className="text-xs text-slate-400">
                              {item.media_type} ·{" "}
                              {formatBytes(item.size_bytes)} ·{" "}
                              {formatDate(
                                item.captured_at ||
                                  item.modified_at ||
                                  item.created_at,
                              )}
                            </p>
                            <p className="truncate text-xs text-slate-500">
                              {item.camera_model ||
                                item.lens_model ||
                                (item.latitude != null
                                  ? "GPS tagged"
                                  : "No metadata")}
                            </p>
                          </div>
                        </button>
                      ))}
                    </div>
                  </section>
                ))}
                {hasMoreItems && (
                  <div className="py-4 text-center text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                    More media loads as you scroll
                  </div>
                )}
              </div>
            )}
            </div>
          </div>
        </section>
      </div>
      {setupOpen && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 sm:p-6">
          <div className="glass-panel w-full max-w-5xl rounded-[24px] shadow-2xl overflow-hidden flex flex-col max-h-full">
            {/* Header */}
            <div className="flex items-center justify-between px-8 py-6 border-b border-white/5 bg-white/[0.02]">
              <div>
                <p className="text-[11px] font-bold uppercase tracking-[0.35em] text-[#4da8ff]">
                  Setup Wizard
                </p>
                <h2 className="mt-1.5 text-[28px] font-black tracking-tight text-white">
                  Folders, Provider & Privacy
                </h2>
              </div>
              <button
                onClick={() => setSetupOpen(false)}
                className="flex h-10 w-10 items-center justify-center rounded-full bg-white/5 text-slate-400 transition-colors hover:bg-white/10 hover:text-white"
              >
                ✕
              </button>
            </div>

            {/* Body */}
            <div className="flex-1 overflow-y-auto px-8 py-6 custom-scrollbar">
              <div className="grid gap-6 md:grid-cols-[1fr_320px]">
                {/* Left Column: Folders & People */}
                <div className="flex flex-col gap-6">
                  <section className="glass-panel rounded-2xl p-6 border border-white/[0.04]">
                    <div className="flex items-center gap-3 mb-4">
                      <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-500/20 text-blue-400">
                        {icons.folder}
                      </span>
                      <h3 className="text-lg font-bold">Media Folders</h3>
                    </div>
                    <p className="text-sm text-slate-400 mb-5 leading-relaxed">
                      Add folders containing your photos and videos. We'll scan
                      them locally without uploading your original files
                      anywhere.
                    </p>
                    <div className="flex flex-col sm:flex-row gap-3">
                      <input
                        value={folderInput}
                        onChange={(e) => setFolderInput(e.target.value)}
                        placeholder="/Users/you/Pictures"
                        className="field flex-1 text-[15px] h-11 px-4"
                      />
                      <div className="flex gap-2">
                        <button
                          onClick={() => addFolder()}
                          className="rounded-xl bg-white/10 px-5 font-semibold text-white transition-colors hover:bg-white/20 h-11"
                        >
                          Add
                        </button>
                        <button
                          onClick={chooseFolder}
                          className="primary-btn px-6 h-11"
                        >
                          Pick Folder
                        </button>
                      </div>
                    </div>
                    {folders.length > 0 && (
                      <div className="mt-5 space-y-2 rounded-xl bg-slate-900/50 p-2 border border-black/20">
                        {folders.map((f) => (
                          <div
                            key={f}
                            className="group flex items-center justify-between rounded-lg bg-slate-800/40 px-4 py-3 text-[14px] transition-colors hover:bg-slate-800/60"
                          >
                            <span className="truncate pr-4 text-slate-200">
                              {f}
                            </span>
                            <button
                              onClick={() => removeFolder(f)}
                              className="shrink-0 text-sm font-medium text-red-400 opacity-80 transition-opacity hover:opacity-100"
                            >
                              Remove
                            </button>
                          </div>
                ))}
                {hasMoreItems && (
                  <div className="col-span-full py-4 text-center text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                    More media loads as you scroll
                  </div>
                )}
              </div>
            )}
                  </section>

                  <section className="glass-panel rounded-2xl p-6 border border-white/[0.04]">
                    <div className="flex items-center gap-3 mb-4">
                      <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/20 text-emerald-400 text-lg">
                        {icons.user}
                      </span>
                      <h3 className="text-lg font-bold">People Recognition</h3>
                    </div>
                    <p className="text-sm text-slate-400 mb-5 leading-relaxed">
                      Library scans only index media metadata. Use Guided Face
                      Setup or Update face embeddings when you want face
                      recognition.
                    </p>

                    {people.length === 0 ? (
                      <div className="rounded-xl border border-dashed border-white/10 p-6 text-center text-sm text-slate-500">
                        No people found yet. Run Guided Face Setup or Update
                        face embeddings after indexing your library.
                      </div>
                    ) : (
                      <div className="grid gap-3 sm:grid-cols-2">
                        {people.map((p) => (
                          <div
                            key={p.id}
                            className="flex gap-2 items-center rounded-xl bg-slate-900/30 p-2"
                          >
                            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-slate-800 text-xs font-bold text-slate-300">
                              {p.face_count}
                            </div>
                            <input
                              className="field h-9 flex-1 text-sm bg-transparent border-none shadow-none focus:bg-white/5"
                              value={rename[p.id] ?? p.name}
                              onChange={(e) =>
                                setRename((r) => ({
                                  ...r,
                                  [p.id]: e.target.value,
                                }))
                              }
                              placeholder="Unnamed person"
                            />
                            <button
                              type="button"
                              title="Remove person"
                              aria-label={`Remove ${p.name} from people`}
                              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-red-500/35 bg-red-950/40 text-lg font-bold leading-none text-red-300 transition-colors hover:border-red-400/60 hover:bg-red-950/70 hover:text-red-200"
                              onClick={() =>
                                deletePerson(p.id, rename[p.id] ?? p.name)
                              }
                            >
                              −
                            </button>
                            <button
                              className="h-9 rounded-lg bg-white/10 px-3 text-xs font-bold text-white transition-colors hover:bg-white/20"
                              onClick={() => renamePerson(p.id)}
                            >
                              Save
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </section>
                </div>

                {/* Right Column: AI & Processing */}
                <div className="flex flex-col gap-6">
                  <section className="glass-panel rounded-2xl p-6 border border-white/[0.04] bg-slate-900/40">
                    <div className="flex items-center gap-3 mb-4">
                      <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-purple-500/20 text-purple-400 text-lg">
                        {icons.sparkles}
                      </span>
                      <h3 className="text-lg font-bold">AI Features</h3>
                    </div>

                    <div className="space-y-5">
                      <div>
                        <label className="text-sm font-medium text-slate-300 mb-2 block">
                          AI Provider
                        </label>
                        <select
                          value={provider}
                          onChange={(e) => setProvider(e.target.value)}
                          className="select-field w-full h-11"
                        >
                          {providers.map((p) => (
                            <option key={p} value={p}>
                              {p.charAt(0).toUpperCase() + p.slice(1)}
                            </option>
                          ))}
                        </select>
                        <p className="mt-2 text-xs text-slate-500">
                          FastEmbed runs local CLIP image embeddings for text
                          search. Google Gemini Embedding 2 can embed supported
                          images, video, audio, and PDFs. OpenRouter supports
                          multimodal embedding models such as Gemini Embedding 2
                          Preview for text and images.
                        </p>
                      </div>

                      <div>
                        <label className="text-sm font-medium text-slate-300 mb-2 block">
                          Embedding Model
                        </label>
                        <select
                          value={embeddingModel}
                          onChange={(e) => setEmbeddingModel(e.target.value)}
                          className="select-field w-full h-11"
                        >
                          {(embeddingModelsByProvider[provider] ?? []).map(
                            (m) => (
                              <option key={m} value={m}>
                                {m}
                              </option>
                            ),
                          )}
                        </select>
                        <p className="mt-2 text-xs text-slate-500">
                          {provider === "google"
                            ? "Set GOOGLE_API_KEY or GEMINI_API_KEY. Text-only Google models skip media files."
                            : provider === "openrouter"
                              ? "Set OPENROUTER_API_KEY. Models without image support skip media files."
                              : "Downloads the selected FastEmbed CLIP model on first use. Non-image media files are skipped."}
                        </p>
                      </div>

                      {(provider === "fastembed" || provider === "google" || provider === "openrouter") && (
                        <div>
                          <label className="text-sm font-medium text-slate-300 mb-2 block">
                            Image Downscale Max Width
                          </label>
                          <input
                            type="number"
                            min="1"
                            step="1"
                            value={embeddingImageMaxWidth}
                            onChange={(e) =>
                              setEmbeddingImageMaxWidth(e.target.value)
                            }
                            className="field w-full h-11"
                            placeholder="Original size"
                          />
                          <p className="mt-2 text-xs text-slate-500">
                            Images wider than this are resized before
                            embedding. Leave empty to use original dimensions.
                          </p>
                        </div>
                      )}

                      <div className="pt-2 space-y-3">
                        <button
                          onClick={() => sidecar("generate_embeddings")}
                          className="flex w-full items-center justify-center gap-2 rounded-xl bg-slate-800 px-4 py-3.5 text-[14px] font-bold text-slate-200 transition-colors hover:bg-slate-700 hover:text-white border border-white/5"
                        >
                          Generate Embeddings
                        </button>
                        <button
                          onClick={openFaceSetup}
                          className="flex w-full items-center justify-center gap-2 rounded-xl bg-slate-800 px-4 py-3.5 text-[14px] font-bold text-slate-200 transition-colors hover:bg-slate-700 hover:text-white border border-white/5"
                        >
                          Guided Face Setup
                        </button>
                        <button
                          type="button"
                          disabled={loading}
                          onClick={updateFaceEmbeddings}
                          className="flex w-full items-center justify-center gap-2 rounded-xl border border-emerald-500/30 bg-emerald-950/40 px-4 py-3.5 text-[14px] font-bold text-emerald-100 transition-colors hover:bg-emerald-900/50 disabled:opacity-50"
                        >
                          Update face embeddings
                        </button>
                      </div>
                    </div>
                  </section>
                </div>
              </div>
            </div>

            {/* Footer */}
            <div className="border-t border-white/5 bg-slate-950/50 px-8 py-5">
              {scanProgress && loading && (
                <div className="mb-4 rounded-xl border border-blue-400/20 bg-blue-400/10 p-3 text-sm text-blue-100">
                  <div className="mb-2 flex justify-between gap-3">
                    <span>{scanProgress.phase}</span>
                    <span>
                      {(scanProgress.total_files != null &&
                        scanProgress.total_files > 0) ||
                      (scanProgress.faces_total != null &&
                        scanProgress.faces_total > 0)
                        ? `${scanPercent(scanProgress).toFixed(1)}%`
                        : `${scanProgress.discovered_files ?? 0} discovered`}{" "}
                      · {scanProgress.scanned_files} indexed
                      {(scanProgress.faces_total ?? 0) > 0 && (
                        <>
                          {" "}
                          · faces {scanProgress.faces_done ?? 0}/
                          {scanProgress.faces_total}
                        </>
                      )}
                    </span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-slate-900">
                    <div
                      className={`h-full rounded-full bg-blue-400 ${
                        (scanProgress.total_files != null &&
                          scanProgress.total_files > 0) ||
                        (scanProgress.faces_total != null &&
                          scanProgress.faces_total > 0)
                          ? ""
                          : "w-1/3 animate-pulse"
                      }`}
                      style={
                        (scanProgress.total_files != null &&
                          scanProgress.total_files > 0) ||
                        (scanProgress.faces_total != null &&
                          scanProgress.faces_total > 0)
                          ? { width: `${scanPercent(scanProgress)}%` }
                          : undefined
                      }
                    />
                  </div>
                  {scanProgress.current_path && (
                    <p className="mt-2 truncate font-mono text-xs text-blue-200/70">
                      {scanProgress.current_path}
                    </p>
                  )}
                </div>
              )}
              <div className="flex items-center justify-between">
                <p className="text-sm text-slate-500 hidden sm:block">
                  Selecting a folder only saves it. Files are read when you
                  scan.
                </p>
                <div className="flex flex-1 sm:flex-none justify-end gap-3">
                  {indexExists && (
                    <button
                      disabled={loading}
                      onClick={deleteIndex}
                      className="rounded-xl border border-red-400/30 bg-red-500/10 px-5 py-3 font-semibold text-red-200 transition-colors hover:bg-red-500/20 disabled:opacity-50"
                    >
                      Delete Index
                    </button>
                  )}
                  <button
                    disabled={loading}
                    onClick={() => setSetupOpen(false)}
                    className="rounded-xl px-6 py-3 font-semibold text-slate-300 transition-colors hover:bg-white/10 hover:text-white disabled:opacity-50"
                  >
                    Close
                  </button>
                  <button
                    disabled={loading}
                    onClick={() => scanFolders()}
                    className="primary-btn px-8 py-3 text-[15px] disabled:opacity-50"
                  >
                    {loading ? "Scanning…" : "Scan Library Now"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      {regionOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
          onClick={() => setRegionOpen(false)}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="glass-panel grid max-h-[92vh] w-full max-w-5xl overflow-hidden rounded-2xl lg:grid-cols-[1fr_320px]"
          >
            <section className="min-h-0 p-4">
              <EarthRegionMap
                points={geoPoints}
                lat={regionDraft.lat}
                lng={regionDraft.lng}
                radius={regionDraft.radius}
                onPick={(lat, lng) =>
                  setRegionDraft((draft) => ({
                    ...draft,
                    lat: lat.toFixed(5),
                    lng: lng.toFixed(5),
                  }))
                }
              />
            </section>
            <aside className="border-t border-white/10 bg-slate-950/40 p-5 lg:border-l lg:border-t-0">
              <p className="text-xs font-black uppercase tracking-[0.28em] text-blue-300">
                Map filter
              </p>
              <h2 className="mt-2 text-2xl font-black">Search region</h2>
              <p className="mt-2 text-sm text-slate-400">
                Zoom and pan the OpenStreetMap view, then click to place the
                center. Indexed geolocated images appear as small pins only.
              </p>
              <div className="mt-6 grid grid-cols-2 gap-3">
                <label>
                  <span className="sub-label">Latitude</span>
                  <input
                    value={regionDraft.lat}
                    onChange={(e) =>
                      setRegionDraft((draft) => ({ ...draft, lat: e.target.value }))
                    }
                    className="field"
                    placeholder="0.00000"
                  />
                </label>
                <label>
                  <span className="sub-label">Longitude</span>
                  <input
                    value={regionDraft.lng}
                    onChange={(e) =>
                      setRegionDraft((draft) => ({ ...draft, lng: e.target.value }))
                    }
                    className="field"
                    placeholder="0.00000"
                  />
                </label>
              </div>
              <label className="mt-5 block">
                <span className="sub-label">Radius kilometers</span>
                <input
                  type="range"
                  min="1"
                  max="2000"
                  step="1"
                  value={regionDraft.radius || "1"}
                  onChange={(e) =>
                    setRegionDraft((draft) => ({
                      ...draft,
                      radius: e.target.value,
                    }))
                  }
                  className="mt-2 w-full"
                />
                <input
                  value={regionDraft.radius}
                  onChange={(e) =>
                    setRegionDraft((draft) => ({
                      ...draft,
                      radius: e.target.value,
                    }))
                  }
                  className="field mt-3"
                  placeholder="50"
                />
              </label>
              <div className="mt-5 rounded-lg border border-white/10 bg-slate-900/60 p-3 text-sm text-slate-300">
                {geoPoints.length} indexed image pin
                {geoPoints.length === 1 ? "" : "s"}
              </div>
              <div className="mt-6 flex justify-end gap-3">
                <button
                  type="button"
                  onClick={() => setRegionOpen(false)}
                  className="rounded-lg px-4 py-2.5 font-bold text-slate-300 hover:bg-white/10 hover:text-white"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={applyRegion}
                  disabled={
                    num(regionDraft.lat) == null ||
                    num(regionDraft.lng) == null ||
                    num(regionDraft.radius) == null
                  }
                  className="primary-btn px-5 py-2.5 disabled:opacity-50"
                >
                  OK
                </button>
              </div>
            </aside>
          </div>
        </div>
      )}
      {faceSetupOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-4"
          onClick={() => setFaceSetupOpen(false)}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="glass-panel flex max-h-[94vh] w-full max-w-7xl flex-col overflow-hidden rounded-3xl"
          >
            <div className="flex items-center justify-between border-b border-white/10 p-5">
              <div>
                <p className="text-xs font-bold uppercase tracking-[0.3em] text-blue-300">
                  Guided Face Setup
                </p>
                <h2 className="text-2xl font-black">
                  Sequential face recognition
                </h2>
                <p className="text-sm text-slate-400">
                  Processes only the image you are viewing. Name each face
                  individually; similar unnamed faces in other photos may be
                  labeled automatically when similarity is high enough.
                </p>
              </div>
              <button
                onClick={() => setFaceSetupOpen(false)}
                className="text-slate-400 hover:text-white"
              >
                ✕
              </button>
            </div>
            <div className="grid min-h-0 flex-1 gap-4 overflow-hidden p-4 lg:grid-cols-[1fr_360px]">
              <section className="flex min-h-0 flex-col gap-3">
                <div className="flex items-center justify-between gap-3">
                  <button
                    disabled={faceIndex <= 0 || faceBusy}
                    onClick={() => goFace(-1)}
                    className="rounded-xl bg-white/10 px-4 py-2 font-bold disabled:opacity-40"
                  >
                    ← Previous
                  </button>
                  <p className="truncate text-sm text-slate-300">
                    {faceItems.length
                      ? `${faceIndex + 1} / ${faceItems.length} · ${currentFaceItem?.file_name}`
                      : "No images found"}
                  </p>
                  <button
                    disabled={faceIndex >= faceItems.length - 1 || faceBusy}
                    onClick={() => goFace(1)}
                    className="rounded-xl bg-white/10 px-4 py-2 font-bold disabled:opacity-40"
                  >
                    Next →
                  </button>
                </div>
                <div className="flex min-h-0 flex-1 items-center justify-center rounded-2xl bg-black/40 p-2">
                  {currentFaceItem && (
                    <div className="relative max-h-full max-w-full">
                      <img
                        ref={faceImageRef}
                        src={mediaUrl(currentFaceItem)}
                        onLoad={(e) => {
                          const rect = e.currentTarget.getBoundingClientRect();
                          setFaceImageSize({
                            width: e.currentTarget.naturalWidth,
                            height: e.currentTarget.naturalHeight,
                          });
                          setFaceImageRect({
                            width: rect.width,
                            height: rect.height,
                          });
                        }}
                        className="block max-h-full max-w-full rounded-xl object-contain"
                      />
                      {faceImageSize &&
                        faceImageRect &&
                        visibleFaces.map((f) => {
                          const scaleX = faceImageRect.width / faceImageSize.width;
                          const scaleY = faceImageRect.height / faceImageSize.height;
                          const left = Math.max(0, f.x * scaleX);
                          const top = Math.max(0, f.y * scaleY);
                          const width = Math.max(
                            0,
                            Math.min(faceImageRect.width - left, f.width * scaleX),
                          );
                          const height = Math.max(
                            0,
                            Math.min(faceImageRect.height - top, f.height * scaleY),
                          );
                          const named = f.person_id != null;
                          const unnamedIdx = named
                            ? -1
                            : unnamedFacesHere.findIndex((u) => u.id === f.id);
                          return (
                            <div
                              key={f.id}
                              className={`pointer-events-none absolute rounded-md border-2 shadow-[0_0_0_1px_rgba(15,23,42,0.9)] ${
                                named
                                  ? "border-emerald-400 shadow-[0_0_24px_rgba(52,211,153,0.35)]"
                                  : "border-blue-300 shadow-[0_0_24px_rgba(96,165,250,0.45)]"
                              }`}
                              style={{
                                left,
                                top,
                                width,
                                height,
                              }}
                            >
                              <span
                                className={`absolute -left-2 -top-3 flex h-7 min-w-7 items-center justify-center rounded-full border border-slate-950 px-2 text-xs font-black text-slate-950 shadow-lg ${
                                  named ? "bg-emerald-400" : "bg-blue-400"
                                }`}
                              >
                                {named
                                  ? (f.person_name || "?").slice(0, 3)
                                  : unnamedIdx >= 0
                                    ? unnamedIdx + 1
                                    : "?"}
                              </span>
                            </div>
                          );
                        })}
                    </div>
                  )}
                </div>
              </section>
              <aside className="min-h-0 overflow-y-auto rounded-2xl border border-white/10 bg-slate-950/30 p-4">
                <div className="mb-4 flex items-center justify-between">
                  <h3 className="font-black">Faces in this image</h3>
                  {faceBusy && (
                    <span className="text-xs text-blue-300">Processing…</span>
                  )}
                </div>
                {!faceBusy && visibleFaces.length === 0 && (
                  <p className="rounded-xl border border-dashed border-white/10 p-5 text-center text-sm text-slate-500">
                    No faces detected here. Move to the next image.
                  </p>
                )}
                {!faceBusy &&
                  visibleFaces.length > 0 &&
                  unnamedFacesHere.length === 0 && (
                    <p className="mb-4 rounded-xl border border-emerald-500/25 bg-emerald-950/20 p-4 text-sm text-emerald-200/90">
                      All detected faces in this photo are named. Use the
                      preview boxes to see who is where.
                    </p>
                  )}
                {namedFacesHere.length > 0 && (
                  <div className="mb-4 rounded-xl border border-emerald-500/20 bg-slate-900/40 p-3">
                    <p className="mb-2 text-xs font-bold uppercase tracking-wide text-emerald-300/90">
                      Named ({namedFacesHere.length})
                    </p>
                    <ul className="space-y-1.5 text-sm text-slate-300">
                      {namedFacesHere.map((f) => (
                        <li key={f.id} className="flex justify-between gap-2">
                          <span className="truncate text-slate-400">
                            Face #{f.id}
                          </span>
                          <span className="shrink-0 font-semibold text-emerald-200">
                            {f.person_name}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {visibleFaces.length > 0 && (
                  <>
                    <p className="mb-2 text-xs font-bold uppercase tracking-wide text-slate-500">
                      Needs a name ({unnamedFacesHere.length})
                    </p>
                    <div className="space-y-3">
                      {unnamedFacesHere.map((f, index) => (
                        <div
                          key={f.id}
                          className="rounded-xl bg-slate-900/60 p-3"
                        >
                          <div className="mb-2 flex items-center justify-between text-xs text-slate-400">
                            <span className="flex items-center gap-2">
                              <span className="flex h-6 min-w-6 items-center justify-center rounded-full bg-blue-400 px-1.5 text-[11px] font-black text-slate-950">
                                {index + 1}
                              </span>
                              Face #{f.id}
                            </span>
                            <span>Unnamed</span>
                          </div>
                          <div className="flex gap-2">
                            <input
                              className="field h-10 flex-1"
                              value={faceNames[f.id] ?? ""}
                              onChange={(e) =>
                                setFaceNames((n) => ({
                                  ...n,
                                  [f.id]: e.target.value,
                                }))
                              }
                              placeholder="Name this person"
                            />
                            <button
                              onClick={() => saveFaceName(f.id)}
                              disabled={
                                faceBusy || !(faceNames[f.id] || "").trim()
                              }
                              className="primary-btn px-4 disabled:opacity-40"
                            >
                              Save
                            </button>
                          </div>
                          <p className="mt-2 text-xs text-slate-500">
                            Box: {Math.round(f.x)}, {Math.round(f.y)} ·{" "}
                            {Math.round(f.width)}×{Math.round(f.height)}
                          </p>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </aside>
            </div>
          </div>
        </div>
      )}
      {selected && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6"
          onClick={() => setSelected(null)}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="glass-panel max-h-[92vh] w-full max-w-6xl overflow-auto rounded-3xl"
          >
            <div className="flex items-center justify-between border-b border-white/10 p-4">
              <div>
                <h2 className="font-bold">{selected.file_name}</h2>
                <p className="text-xs text-slate-400">{selected.path}</p>
              </div>
              <button
                onClick={() => setSelected(null)}
                className="text-slate-400 hover:text-white"
              >
                ✕
              </button>
            </div>
            <div className="grid gap-4 p-4 lg:grid-cols-[1fr_330px]">
              {selected.media_type === "video" ? (
                <video
                  src={mediaUrl(selected)}
                  controls
                  className="max-h-[70vh] w-full rounded-2xl bg-black"
                />
              ) : (
                <img
                  src={mediaUrl(selected)}
                  className="max-h-[70vh] w-full rounded-2xl bg-black object-contain"
                />
              )}
              <aside className="glass-panel space-y-3 rounded-2xl p-4 text-sm">
                <p>
                  <b>Type:</b> {selected.media_type}
                </p>
                <p>
                  <b>Size:</b> {formatBytes(selected.size_bytes)}
                </p>
                {metaRows(selected).map(([k, v]) => (
                  <p key={k}>
                    <b>{k}:</b> {v}
                  </p>
                ))}
                <p>
                  <b>Status:</b> {selected.missing ? "Missing" : "Available"}
                </p>
                {selected.latitude != null && selected.longitude != null && (
                  <>
                    <iframe
                      title="OpenStreetMap"
                      className="h-48 w-full rounded-xl border-0"
                      src={`https://www.openstreetmap.org/export/embed.html?bbox=${selected.longitude - 0.03}%2C${selected.latitude - 0.03}%2C${selected.longitude + 0.03}%2C${selected.latitude + 0.03}&layer=mapnik&marker=${selected.latitude}%2C${selected.longitude}`}
                    />
                    <a
                      className="text-blue-300 underline"
                      target="_blank"
                      href={`https://www.openstreetmap.org/?mlat=${selected.latitude}&mlon=${selected.longitude}#map=14/${selected.latitude}/${selected.longitude}`}
                    >
                      Open location
                    </a>
                  </>
                )}
              </aside>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

export default App;
