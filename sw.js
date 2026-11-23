/* CueCam service worker: network first (always the latest version when online), cache as fallback offline */
const CACHE = 'cuecam-v2';
// everything the app needs offline (voice filter, MP3 recorder, PDF/Word import); fetched once while online
const FILES = ['./', 'index.html', 'manifest.webmanifest', 'icon-192.png', 'icon-512.png', 'icon.svg', 'vietqr-donate.svg',
  'lib/lame.min.js', 'lib/mp3worker.js', 'lib/ns/rnnoise-worklet.js', 'lib/ns/rnnoise.wasm',
  'lib/pdf.min.js', 'lib/pdf.worker.min.js', 'lib/mammoth.browser.min.js'];
self.addEventListener('install', e => { self.skipWaiting(); e.waitUntil(caches.open(CACHE).then(c => Promise.all(FILES.map(f => c.add(f).catch(() => {}))))); });
self.addEventListener('activate', e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener('fetch', e => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== 'GET' || url.origin !== location.origin) return; // Google, speech, Drive go straight to the network
  e.respondWith(fetch(req).then(res => {
    if (res.ok) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(req, copy)).catch(() => {}); }
    return res;
  }).catch(() => caches.match(req, { ignoreSearch: true }).then(r => r || caches.match('index.html'))));
});
