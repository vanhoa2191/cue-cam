/* CueCam beauty worker: runs the person mask (MediaPipe selfie segmenter) and face detector
   off the main thread, so the teleprompter and the recorder stay smooth.
   Main thread sends small ImageBitmaps, one at a time; we answer with a Uint8 mask and face points. */
import { FilesetResolver, ImageSegmenter, FaceDetector } from './mp/vision_bundle.mjs';

let MF = null, seg = null, face = null, base = '', delegate = 'CPU', busy = false, lastTs = 0;

async function make(kind, fileset, dlg) {
  if (kind === 'seg') return ImageSegmenter.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: base + 'mp/models/selfie_segmenter.tflite', delegate: dlg },
    runningMode: 'VIDEO', outputConfidenceMasks: true, outputCategoryMask: false
  });
  return FaceDetector.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: base + 'mp/models/blaze_face_short_range.tflite', delegate: dlg },
    runningMode: 'VIDEO', minDetectionConfidence: 0.5
  });
}

async function ensure(kind) {
  if (kind === 'seg' && seg) return seg;
  if (kind === 'face' && face) return face;
  // module mode clears self.ModuleFactory after each task is created, and the cached import
  // does not set it again: keep our own copy and put it back before every task
  const fileset = await FilesetResolver.forVisionTasks(base + 'mp/wasm', true);
  if (!MF) { await import(fileset.wasmLoaderPath); MF = self.ModuleFactory; }
  self.ModuleFactory = MF;
  let task = null;
  // GPU is much faster on phones; fall back to CPU (XNNPACK) when WebGL in workers is missing
  for (const dlg of (delegate === 'GPU' ? ['GPU', 'CPU'] : ['CPU'])) {
    try { task = await make(kind, fileset, dlg); delegate = dlg; break; } catch (e) { if (dlg === 'CPU') throw e; }
  }
  if (kind === 'seg') seg = task; else face = task;
  return task;
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === 'init') {
    base = new URL('./', import.meta.url).href; delegate = m.gpu === false ? 'CPU' : 'GPU';
    try {
      if (m.seg) await ensure('seg');
      if (m.face) await ensure('face');
      self.postMessage({ type: 'ready', delegate });
    } catch (err) { self.postMessage({ type: 'error', msg: String(err && err.message || err) }); }
    return;
  }
  if (m.type === 'need') { // turn a model on later
    try { if (m.seg) await ensure('seg'); if (m.face) await ensure('face'); self.postMessage({ type: 'ready', delegate }); }
    catch (err) { self.postMessage({ type: 'error', msg: String(err && err.message || err) }); }
    return;
  }
  if (m.type !== 'frame') return;
  const bmp = m.bitmap;
  if (busy) { bmp.close(); return; }
  busy = true;
  const out = { type: 'result', id: m.id };
  const t0 = performance.now();
  try {
    let ts = Math.max(lastTs + 1, Math.round(m.ts)); lastTs = ts;
    if (m.seg && seg) {
      const r = seg.segmentForVideo(bmp, ts);
      const mk = r.confidenceMasks && r.confidenceMasks[0];
      if (mk) {
        const f = mk.getAsFloat32Array(), u = new Uint8Array(f.length);
        for (let i = 0; i < f.length; i++) { const v = f[i] * 255; u[i] = v < 0 ? 0 : v > 255 ? 255 : v; }
        out.mask = u; out.mw = mk.width; out.mh = mk.height;
      }
      if (r.close) r.close();
    }
    if (m.face && face) {
      const r = face.detectForVideo(bmp, ts), W = bmp.width, H = bmp.height;
      let best = null;
      for (const d of r.detections || []) {
        const b = d.boundingBox; if (!b) continue;
        if (!best || b.width * b.height > best.b.width * best.b.height) best = { b, k: d.keypoints || [] };
      }
      if (best) out.face = { x: best.b.originX / W, y: best.b.originY / H, w: best.b.width / W, h: best.b.height / H, k: best.k.map(p => [p.x, p.y]) };
      else out.face = null;
    }
  } catch (err) { out.err = String(err && err.message || err); }
  bmp.close();
  out.ms = performance.now() - t0;
  busy = false;
  self.postMessage(out, out.mask ? [out.mask.buffer] : []);
};
