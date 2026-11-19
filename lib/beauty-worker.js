/* CueCam beauty worker: runs the person mask (MediaPipe selfie segmenter) and face detector
   off the main thread, so the teleprompter and the recorder stay smooth.
   Main thread sends small ImageBitmaps, one at a time; we answer with a Uint8 mask and face points. */
import { FilesetResolver, ImageSegmenter, FaceDetector, FaceLandmarker } from './mp/vision_bundle.mjs';

const LIP_OUT = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146];
const LIP_IN = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];
// eyelids, corner to corner: [upper, lower] for each eye
const EYES = [
  [[33, 246, 161, 160, 159, 158, 157, 173, 133], [33, 7, 163, 144, 145, 153, 154, 155, 133]],
  [[263, 466, 388, 387, 386, 385, 384, 398, 362], [263, 249, 390, 373, 374, 380, 381, 382, 362]]
];
const OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
let MF = null, seg = null, face = null, mesh = null, base = '', delegate = 'CPU', busy = false, lastTs = 0;

async function make(kind, fileset, dlg) {
  if (kind === 'seg') return ImageSegmenter.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: base + 'mp/models/selfie_segmenter.tflite', delegate: dlg },
    runningMode: 'VIDEO', outputConfidenceMasks: true, outputCategoryMask: false
  });
  if (kind === 'mesh') return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: base + 'mp/models/face_landmarker.task', delegate: dlg },
    runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: false, outputFacialTransformationMatrixes: false,
    minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5, minTrackingConfidence: 0.5
  });
  return FaceDetector.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: base + 'mp/models/blaze_face_short_range.tflite', delegate: dlg },
    runningMode: 'VIDEO', minDetectionConfidence: 0.5
  });
}

async function ensure(kind) {
  if (kind === 'seg' && seg) return seg;
  if (kind === 'face' && face) return face;
  if (kind === 'mesh' && mesh) return mesh;
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
  if (kind === 'seg') seg = task; else if (kind === 'mesh') mesh = task; else face = task;
  return task;
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === 'init') {
    base = new URL('./', import.meta.url).href; delegate = m.gpu === false ? 'CPU' : 'GPU';
    try {
      if (m.seg) await ensure('seg');
      if (m.face) await ensure('face');
      if (m.mesh) await ensure('mesh');
      self.postMessage({ type: 'ready', delegate });
    } catch (err) { self.postMessage({ type: 'error', msg: String(err && err.message || err) }); }
    return;
  }
  if (m.type === 'need') { // turn a model on later
    try { if (m.seg) await ensure('seg'); if (m.face) await ensure('face'); if (m.mesh) await ensure('mesh'); self.postMessage({ type: 'ready', delegate }); }
    catch (err) { self.postMessage({ type: 'error', msg: String(err && err.message || err) }); }
    return;
  }
  if (m.type !== 'frame') return;
  const bmp = m.bitmap;
  if (busy) { bmp.close(); if (m.big) m.big.close(); return; }
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
    if (m.mesh && mesh) {
      // 478 face points: lip outline for lipstick, plus the face box/eyes/mouth for skin smoothing
      const r = mesh.detectForVideo(m.big || bmp, ts), L = r.faceLandmarks && r.faceLandmarks[0];
      if (L) {
        const pt = i => [L[i].x, L[i].y];
        out.lips = { o: LIP_OUT.map(pt), i: LIP_IN.map(pt), e: EYES.map(e => [e[0].map(pt), e[1].map(pt)]) };
        let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
        for (const i of OVAL) { const q = L[i]; x0 = Math.min(x0, q.x); y0 = Math.min(y0, q.y); x1 = Math.max(x1, q.x); y1 = Math.max(y1, q.y); }
        const h = y1 - y0; // oval spans forehead..chin; mimic the detector box (brows..chin)
        out.face = { x: x0, y: y0 + h * 0.18, w: x1 - x0, h: h * 0.82, k: [pt(468), pt(473), pt(1), pt(13)] };
      } else { out.lips = null; out.face = null; }
    }
    if (m.face && face && !(m.mesh && mesh)) {
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
  bmp.close(); if (m.big) m.big.close();
  out.ms = performance.now() - t0;
  busy = false;
  self.postMessage(out, out.mask ? [out.mask.buffer] : []);
};
