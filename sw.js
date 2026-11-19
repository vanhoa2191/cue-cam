/* CueCam service worker: network first (always the latest version when online), cache as fallback offline.
   Code files are fetched with cache: 'no-cache' so the browser re-checks GitHub Pages every time
   (its 10-minute HTTP cache otherwise kept old versions after an update). */
const CACHE = 'cuecam-v2';
self.addEventListener('install', e => { self.skipWaiting(); e.waitUntil(caches.open(CACHE).then(c => c.addAll(['./', 'index.html', 'manifest.webmanifest', 'icon-192.png', 'icon.svg']).catch(() => {}))); });
self.addEventListener('activate', e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener('fetch', e => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== 'GET' || url.origin !== location.origin) return; // Google, speech, Drive go straight to the network
  const big = /\.(wasm|tflite|task|onnx)$/.test(url.pathname); // large AI files: normal HTTP caching is fine
  e.respondWith((big ? fetch(req) : fetch(req.mode === 'navigate' ? url.href : req, { cache: 'no-cache' })).then(res => {
    if (res.ok) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(req, copy)).catch(() => {}); }
    return res;
  }).catch(() => caches.match(req, { ignoreSearch: true }).then(r => r || caches.match('index.html'))));
});
