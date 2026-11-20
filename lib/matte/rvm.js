/* CueCam "Làm nét phông": Robust Video Matting (Lin et al. 2021, GPL-3.0) run over a whole recorded video,
   frame by frame, after recording. Unlike the live segmenter it does not have to keep up with the camera, so it can
   use a much heavier model: a soft alpha with hair detail (deep guided filter at full resolution), a foreground
   with the old background's colour spill removed, and recurrent state so the edge does not flicker between frames.
   Uses onnxruntime-web from lib/ort (shared with lib/beauty-worker.js): WebGPU when the phone has it, else CPU. */

const MODEL = 'matte/rvm_mobilenetv3_fp32.onnx';

export async function createMatter(base, { preferCpu = false } = {}) {
  const ort = await import(base + 'ort/ort.webgpu.min.mjs');
  ort.env.wasm.wasmPaths = base + 'ort/'; ort.env.wasm.numThreads = 1; ort.env.logLevel = 'error';
  const rec = ['r1o', 'r2o', 'r3o', 'r4o'];
  let sess = null, ep = 'cpu';
  if (!preferCpu && navigator.gpu) {
    try {
      sess = await ort.InferenceSession.create(base + MODEL, {
        executionProviders: ['webgpu'], graphOptimizationLevel: 'all',
        // the recurrent state only goes back into the model: keep it on the GPU between frames
        preferredOutputLocation: { r1o: 'gpu-buffer', r2o: 'gpu-buffer', r3o: 'gpu-buffer', r4o: 'gpu-buffer', fgr: 'cpu', pha: 'cpu' }
      });
      ep = 'webgpu';
    } catch (e) { console.warn('RVM WebGPU failed, using CPU', e); sess = null; }
  }
  if (!sess) sess = await ort.InferenceSession.create(base + MODEL, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });

  const zero = () => new ort.Tensor('float32', new Float32Array(1), [1, 1, 1, 1]);
  let state = null, ratio = null;
  const dropState = () => { if (state) for (const t of state) if (t.location === 'gpu-buffer') t.dispose(); state = null; };

  return {
    ep,
    reset() { dropState(); },
    /* rgba: Uint8ClampedArray w*h*4 of the frame. Returns { fgr (CHW float 0..1), pha (float 0..1) }. */
    async run(rgba, w, h) {
      const n = w * h, src = new Float32Array(3 * n);
      for (let i = 0, j = 0; i < n; i++, j += 4) { src[i] = rgba[j] / 255; src[n + i] = rgba[j + 1] / 255; src[2 * n + i] = rgba[j + 2] / 255; }
      // the authors' advice: the coarse network should see ~256-512 px; the refiner then works at full resolution
      const r = Math.min(1, 512 / Math.max(w, h));
      if (!ratio || ratio.data[0] !== r) ratio = new ort.Tensor('float32', new Float32Array([r]), [1]);
      const s = state || [zero(), zero(), zero(), zero()];
      const out = await sess.run({ src: new ort.Tensor('float32', src, [1, 3, h, w]), r1i: s[0], r2i: s[1], r3i: s[2], r4i: s[3], downsample_ratio: ratio });
      dropState(); state = rec.map(k => out[k]);
      return { fgr: out.fgr.data, pha: out.pha.data };
    },
    release() { dropState(); try { sess.release(); } catch (e) {} }
  };
}

/* Puts the person (fgr, pha) over a background. bg: Uint8ClampedArray RGBA (same size). Writes into out (RGBA). */
export function composite(fgr, pha, bg, out, n) {
  const n2 = 2 * n;
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    const a = pha[i], b = 1 - a;
    out[j] = fgr[i] * a * 255 + bg[j] * b;
    out[j + 1] = fgr[n + i] * a * 255 + bg[j + 1] * b;
    out[j + 2] = fgr[n2 + i] * a * 255 + bg[j + 2] * b;
    out[j + 3] = 255;
  }
}

/* Blurred background of this very frame, with the person taken out before blurring so no halo of the person
   is smeared around them. The canvas keeps premultiplied colour, so blurring (colour x (1 - alpha)) and reading it
   back gives the average of background pixels only. Done at quarter size: it is going to be blurry anyway. */
export function makeBlurBg(w, h) {
  const sw = Math.max(16, Math.round(w / 4)), sh = Math.max(16, Math.round(h / 4));
  const mk = (a, b) => { const c = new OffscreenCanvas(a, b); return [c, c.getContext('2d', { willReadFrequently: true })]; };
  const [c0, x0] = mk(sw, sh), [c1, x1] = mk(sw, sh), [c2, x2] = mk(sw, sh), [, xf] = mk(w, h);
  return (frameCanvas, pha, strength) => {
    x0.drawImage(frameCanvas, 0, 0, sw, sh);
    const d = x0.getImageData(0, 0, sw, sh), p = d.data, fx = w / sw, fy = h / sh;
    for (let y = 0, j = 0; y < sh; y++) {
      const row = Math.min(h - 1, Math.floor((y + 0.5) * fy)) * w;
      for (let x = 0; x < sw; x++, j += 4) {
        // erode the person a little (take the max alpha around): their edge must not leak into the blur
        const cx = Math.min(w - 1, Math.floor((x + 0.5) * fx));
        let a = pha[row + cx];
        const l = pha[row + Math.max(0, cx - 3)], r = pha[row + Math.min(w - 1, cx + 3)];
        if (l > a) a = l; if (r > a) a = r;
        p[j + 3] = (1 - Math.min(1, a * 1.3)) * 255;
      }
    }
    x1.clearRect(0, 0, sw, sh); x1.putImageData(d, 0, 0);
    x2.clearRect(0, 0, sw, sh); x2.filter = 'blur(' + (1 + strength * 0.14).toFixed(1) + 'px)'; x2.drawImage(c1, 0, 0); x2.filter = 'none';
    const b = x2.getImageData(0, 0, sw, sh), q = b.data;
    // where everything nearby was person there is nothing to average: fill from the plain blurred frame
    x0.filter = 'blur(' + (1 + strength * 0.14).toFixed(1) + 'px)'; x0.drawImage(frameCanvas, 0, 0, sw, sh); x0.filter = 'none';
    const pl = x0.getImageData(0, 0, sw, sh).data;
    for (let j = 0; j < q.length; j += 4) {
      const k = Math.min(1, q[j + 3] / 40);
      q[j] = q[j] * k + pl[j] * (1 - k); q[j + 1] = q[j + 1] * k + pl[j + 1] * (1 - k); q[j + 2] = q[j + 2] * k + pl[j + 2] * (1 - k); q[j + 3] = 255;
    }
    x1.putImageData(b, 0, 0);
    xf.imageSmoothingQuality = 'high'; xf.drawImage(c1, 0, 0, w, h);
    return xf.getImageData(0, 0, w, h).data;
  };
}

/* A still background (photo or plain colour), cover-fitted once. */
export function makeStillBg(w, h, img, color) {
  const c = new OffscreenCanvas(w, h), x = c.getContext('2d', { willReadFrequently: true });
  x.fillStyle = color || '#1b2433'; x.fillRect(0, 0, w, h);
  if (img) {
    const iw = img.width, ih = img.height, s = Math.max(w / iw, h / ih);
    x.imageSmoothingQuality = 'high'; x.drawImage(img, (w - iw * s) / 2, (h - ih * s) / 2, iw * s, ih * s);
  }
  return x.getImageData(0, 0, w, h).data;
}
