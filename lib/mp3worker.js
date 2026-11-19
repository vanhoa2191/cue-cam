/* CueCam: encode mono PCM (Float32Array) to MP3 with lamejs (LGPL, lib/lame.min.js) */
importScripts('lame.min.js');
onmessage = e => {
  const { pcm, rate, kbps } = e.data;
  try {
    const enc = new lamejs.Mp3Encoder(1, rate, kbps || 128);
    const BLOCK = 1152 * 20, out = [], n = pcm.length;
    const buf = new Int16Array(BLOCK);
    let last = 0;
    for (let i = 0; i < n; i += BLOCK) {
      const len = Math.min(BLOCK, n - i);
      for (let k = 0; k < len; k++) { const s = Math.max(-1, Math.min(1, pcm[i + k])); buf[k] = s < 0 ? s * 0x8000 : s * 0x7fff; }
      const mp3 = enc.encodeBuffer(len === BLOCK ? buf : buf.subarray(0, len));
      if (mp3.length) out.push(new Uint8Array(mp3));
      const p = Math.floor(i / n * 100);
      if (p >= last + 5) { last = p; postMessage({ progress: p }); }
    }
    const end = enc.flush(); if (end.length) out.push(new Uint8Array(end));
    postMessage({ blob: new Blob(out, { type: 'audio/mpeg' }) });
  } catch (err) { postMessage({ error: String(err && err.message || err) }); }
};
