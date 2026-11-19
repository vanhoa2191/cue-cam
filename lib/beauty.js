/* CueCam beauty + background pipeline (runs on the phone, nothing is uploaded).
   camera <video> -> WebGL canvas (skin smoothing, brightening, tone, studio light, colour filter,
   background blur/replace) -> canvas.captureStream() -> MediaRecorder, so the saved video has the look.
   The person mask and face box come from lib/beauty-worker.js (MediaPipe, in a Web Worker). */
(function () {
  'use strict';

  const FILTERS = [
    { id: 'none', name: 'Gốc', sw: 'linear-gradient(135deg,#777,#bbb)' },
    { id: 'clear', name: 'Trong trẻo', sw: 'linear-gradient(135deg,#bfe3ff,#fff)', exp: 1.05, con: 1.06, sat: 1.06, temp: -0.15, hi: [0, 0.02, 0.04] },
    { id: 'warm', name: 'Ấm áp', sw: 'linear-gradient(135deg,#f3b27a,#ffe2b8)', exp: 1.03, sat: 1.05, temp: 0.5, tint: 0.08, sh: [0.04, 0.01, -0.03] },
    { id: 'pink', name: 'Hồng phấn', sw: 'linear-gradient(135deg,#f6b6c8,#fde6ee)', exp: 1.06, con: 0.95, sat: 0.95, tint: 0.25, hi: [0.06, 0, 0.04], fade: 0.03 },
    { id: 'cream', name: 'Kem sữa', sw: 'linear-gradient(135deg,#e9d6bb,#fff7ea)', exp: 1.07, con: 0.9, sat: 0.85, temp: 0.2, fade: 0.06 },
    { id: 'golden', name: 'Nắng chiều', sw: 'linear-gradient(135deg,#f0a040,#ffd27a)', exp: 1.04, sat: 1.08, temp: 0.75, hi: [0.08, 0.04, -0.04], vig: 0.15 },
    { id: 'cinema', name: 'Điện ảnh', sw: 'linear-gradient(135deg,#1f6f7a,#f0a060)', con: 1.12, sat: 0.9, sh: [-0.07, 0.02, 0.09], hi: [0.09, 0.04, -0.05], vig: 0.3 },
    { id: 'cool', name: 'Lạnh xanh', sw: 'linear-gradient(135deg,#6aa6d8,#d6ecff)', con: 1.05, sat: 0.95, temp: -0.6, sh: [0, 0.02, 0.06] },
    { id: 'vintage', name: 'Hoài cổ', sw: 'linear-gradient(135deg,#a08060,#e0cfa8)', con: 0.92, sat: 0.72, temp: 0.35, fade: 0.1, vig: 0.25 },
    { id: 'vivid', name: 'Rực rỡ', sw: 'linear-gradient(135deg,#e2306c,#ffcd38)', con: 1.1, sat: 1.35 },
    { id: 'mono', name: 'Đen trắng', sw: 'linear-gradient(135deg,#222,#eee)', con: 1.15, sat: 0, vig: 0.2 }
  ];
  const BGS = [
    { id: 'studio', name: 'Studio xám' }, { id: 'navy', name: 'Xanh đêm' }, { id: 'sunset', name: 'Hoàng hôn' },
    { id: 'beige', name: 'Kem ấm' }, { id: 'bokeh', name: 'Đèn mờ' }, { id: 'office', name: 'Phòng làm việc' }
  ];
  const LIPS = [
    { id: 'red', name: 'Đỏ cổ điển', c: '#b3122e' }, { id: 'rose', name: 'Hồng đất', c: '#b2565f' }, { id: 'coral', name: 'Cam đào', c: '#e0614a' },
    { id: 'nude', name: 'Nude', c: '#c4806f' }, { id: 'pink', name: 'Hồng tươi', c: '#e0457b' }, { id: 'wine', name: 'Đỏ rượu', c: '#74132c' }
  ];
  const DEF = { lip: 0, lipColor: 'rose', smooth: 0, bright: 0, tone: 0, light: 0, faceOnly: true, filter: 'none', fstr: 80, bg: 'none', blur: 60, edge: 50 };

  /* ---------------- shaders ---------------- */
  const HP = '#ifdef GL_FRAGMENT_PRECISION_HIGH\nprecision highp float;\n#else\nprecision mediump float;\n#endif\nvarying vec2 v;\n';
  const VS = 'attribute vec2 p; varying vec2 v; void main(){ v = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }';
  const FS = {
    // 4-tap box downsample (each bilinear tap averages 2x2)
    down: HP + 'uniform sampler2D t; uniform vec2 px; void main(){ gl_FragColor = 0.25 * (texture2D(t, v + px * vec2(-1.,-1.)) + texture2D(t, v + px * vec2(1.,-1.)) + texture2D(t, v + px * vec2(-1.,1.)) + texture2D(t, v + px * vec2(1.,1.))); }',
    // separable edge-preserving (bilateral) blur for skin
    bil: HP + `uniform sampler2D t; uniform vec2 d; uniform float k;
      void main(){ vec3 c0 = texture2D(t, v).rgb; vec3 s = c0; float ws = 1.0;
        for (int i = 1; i <= 6; i++) { float fi = float(i); float sw = exp(-fi * fi / 20.0);
          vec3 a = texture2D(t, v + d * fi).rgb, b = texture2D(t, v - d * fi).rgb;
          vec3 da = a - c0, db = b - c0;
          float wa = sw * exp(-dot(da, da) * k), wb = sw * exp(-dot(db, db) * k);
          s += a * wa + b * wb; ws += wa + wb; }
        gl_FragColor = vec4(s / ws, 1.0); }`,
    // mask refine: joint bilateral against the image (edges follow hair/shoulders) + adaptive temporal smoothing (no flicker)
    mask: HP + `uniform sampler2D m, g, pv; uniform vec2 px; uniform float a0;
      void main(){ vec3 c0 = texture2D(g, v).rgb; float s = 0.0, ws = 0.0;
        for (int y = -3; y <= 3; y++) for (int x = -3; x <= 3; x++) {
          vec2 o = vec2(float(x), float(y)) * px * 1.4; vec3 dc = texture2D(g, v + o).rgb - c0;
          float w = exp(-dot(dc, dc) * 40.0) * exp(-float(x * x + y * y) / 10.0);
          s += texture2D(m, v + o).r * w; ws += w; }
        float cur = s / ws, prev = texture2D(pv, v).r;
        float k = clamp(0.3 + abs(cur - prev) * 2.2, 0.0, 1.0); k = max(k, a0);
        gl_FragColor = vec4(vec3(mix(prev, cur, k)), 1.0); }`,
    // background blur, first pass: weights by "not person" so the person never bleeds into the blur (no halo)
    blur0: HP + `uniform sampler2D t, m; uniform vec2 d;
      void main(){ vec4 s = vec4(0.0); float ws = 0.0;
        for (int i = -6; i <= 6; i++) { float fi = float(i); float w = exp(-fi * fi / 18.0); vec2 o = v + d * fi;
          float b = 1.0 - smoothstep(0.2, 0.6, texture2D(m, o).r); b = b * b + 0.001;
          s += vec4(texture2D(t, o).rgb * b, b) * w; ws += w; }
        gl_FragColor = s / ws; }`,
    blurp: HP + `uniform sampler2D t; uniform vec2 d;
      void main(){ vec4 s = vec4(0.0); float ws = 0.0;
        for (int i = -6; i <= 6; i++) { float fi = float(i); float w = exp(-fi * fi / 18.0); s += texture2D(t, v + d * fi) * w; ws += w; }
        gl_FragColor = s / ws; }`,
    comp: HP + `uniform sampler2D vid, smo, msk, bgb, bgi, lipm; uniform float uLip; uniform vec3 uLipCol;
      uniform vec2 res; uniform float uSmooth, uBright, uTone, uLight, uBg, uSeg, uFaceOn, uEdge;
      uniform vec4 uFace, uEyeA, uEyeB, uMouth; uniform vec4 uFit;
      uniform vec3 uF1, uF2, uSh, uHi; uniform vec2 uF3;
      float ell(vec2 p, vec2 c, vec2 r){ return length((p - c) / r); }
      float luma(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }
      void main(){
        vec3 c = texture2D(vid, v).rgb; vec2 p = v * res;
        float pm = uSeg > 0.5 ? texture2D(msk, v).r : 1.0;
        // skin likelihood in YCbCr
        float y = luma(c), cb = 0.5 + (-0.169 * c.r - 0.331 * c.g + 0.5 * c.b), cr = 0.5 + (0.5 * c.r - 0.419 * c.g - 0.081 * c.b);
        float skin = 1.0 - smoothstep(0.75, 1.25, length(vec2((cb - 0.41) / 0.085, (cr - 0.6) / 0.075)));
        skin *= smoothstep(0.08, 0.2, y);
        float reg = 1.0;
        if (uFaceOn > 0.5) {
          float f = 1.0 - smoothstep(0.85, 1.08, ell(p, uFace.xy, uFace.zw));
          float n = 1.0 - smoothstep(0.7, 1.05, ell(p, uFace.xy - vec2(0.0, uFace.w * 1.1), uFace.zw * vec2(0.62, 0.55)));
          float keep = smoothstep(0.7, 1.25, ell(p, uEyeA.xy, uEyeA.zw)) * smoothstep(0.7, 1.25, ell(p, uEyeB.xy, uEyeB.zw)) * smoothstep(0.75, 1.25, ell(p, uMouth.xy, uMouth.zw));
          reg = max(f, n * 0.85) * mix(0.15, 1.0, keep);
        }
        if (uSmooth > 0.0) {
          vec3 s = texture2D(smo, v).rgb;
          float k = uSmooth * skin * reg * min(1.0, pm * 1.5);
          c = mix(c, s, k * 0.88);
        }
        if (uBright > 0.0) {
          float bw = uBright * (0.45 + 0.55 * skin * reg);
          c = c + (1.0 - c) * c * 0.75 * bw;
          c = mix(c, vec3(luma(c)), 0.12 * bw * skin);
        }
        if (uLip > 0.0) {
          // lipstick: overlay blend keeps the lips' own shading and shine
          float lm = texture2D(lipm, v).r * uLip, l = luma(c);
          vec3 ov = l < 0.5 ? 2.0 * l * uLipCol : 1.0 - 2.0 * (1.0 - l) * (1.0 - uLipCol);
          c = mix(c, mix(ov, uLipCol * (0.6 + 0.8 * l), 0.2), lm * 0.7);
        }
        c *= vec3(1.0 + 0.08 * uTone, 1.0 + 0.012 * uTone, 1.0 - 0.08 * uTone);
        // background
        float a = 1.0;
        if (uBg > 0.5 && uSeg > 0.5) {
          float lo = mix(0.42, 0.22, uEdge), hi = mix(0.58, 0.78, uEdge);
          a = smoothstep(lo, hi, pm);
          vec3 bg;
          if (uBg < 1.5) { vec4 b = texture2D(bgb, v); bg = b.a > 0.004 ? b.rgb / b.a : c; }
          else bg = texture2D(bgi, (v - 0.5) * uFit.xy + 0.5).rgb;
          // light wrap (as in Google Meet): let a little of the new background spill over the edge so the person sits in it
          float e = a * (1.0 - a) * 4.0; c = mix(c, bg, e * 0.3);
          c = mix(bg, c, a);
        }
        // studio light: soft key light on the face from the upper left, darker surroundings
        if (uLight > 0.0) {
          vec2 lc = uFace.xy + vec2(-uFace.z * 0.35, uFace.w * 0.25);
          float d1 = ell(p, lc, uFace.zw * 2.2), d2 = ell(p, uFace.xy, uFace.zw * 3.4);
          float key = exp(-d1 * d1 * 1.6), amb = exp(-d2 * d2);
          float subj = uSeg > 0.5 ? a : 1.0;
          c = c * (1.0 + 0.42 * uLight * key * (0.4 + 0.6 * subj)) + vec3(0.025, 0.018, 0.0) * uLight * key;
          c *= 1.0 - uLight * 0.38 * (1.0 - amb) * (1.0 - 0.6 * subj);
        }
        // colour filter
        if (uF3.y > 0.0) {
          vec3 f = c * uF1.x; f = (f - 0.5) * uF1.y + 0.5;
          f = mix(vec3(luma(f)), f, uF1.z);
          f *= vec3(1.0 + uF2.x * 0.1, 1.0 + uF2.y * 0.04, 1.0 - uF2.x * 0.1);
          float l = clamp(luma(f), 0.0, 1.0); f += uSh * (1.0 - l) + uHi * l;
          f = mix(vec3(uF2.z), vec3(1.0), clamp(f, 0.0, 1.0));
          float dv = length(v - 0.5) * 1.414; f *= 1.0 - uF3.x * smoothstep(0.45, 1.05, dv);
          c = mix(c, f, uF3.y);
        }
        gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
      }`
  };

  /* ---------------- background presets ---------------- */
  function drawPreset(id, w, h) {
    const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
    const x = cv.getContext('2d'), m = Math.max(w, h);
    const rad = (stops, cx, cy, r) => { const g = x.createRadialGradient(cx, cy, 0, cx, cy, r); stops.forEach((s, i) => g.addColorStop(i / (stops.length - 1), s)); x.fillStyle = g; x.fillRect(0, 0, w, h); };
    const lin = (stops) => { const g = x.createLinearGradient(0, 0, w * 0.3, h); stops.forEach((s, i) => g.addColorStop(i / (stops.length - 1), s)); x.fillStyle = g; x.fillRect(0, 0, w, h); };
    let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const dots = (n, cols, rmin, rmax, alpha) => {
      x.save(); x.filter = 'blur(' + Math.round(m * 0.012) + 'px)';
      for (let i = 0; i < n; i++) { x.globalAlpha = alpha * (0.5 + rnd() * 0.5); x.fillStyle = cols[i % cols.length]; x.beginPath(); x.arc(rnd() * w, rnd() * h * 0.85, m * (rmin + rnd() * (rmax - rmin)), 0, 7); x.fill(); }
      x.restore(); x.globalAlpha = 1;
    };
    if (id === 'studio') rad(['#8a8f97', '#5a5e65', '#25282c'], w * 0.5, h * 0.38, m * 0.75);
    else if (id === 'navy') rad(['#2d4f7c', '#17263f', '#0a111d'], w * 0.5, h * 0.35, m * 0.8);
    else if (id === 'sunset') lin(['#ffb36b', '#f27a7d', '#8a5aa8', '#3a3170']);
    else if (id === 'beige') rad(['#f4e7d3', '#e2cfb2', '#b89f80'], w * 0.5, h * 0.35, m * 0.85);
    else if (id === 'bokeh') { rad(['#2a2233', '#140f1a'], w * 0.5, h * 0.4, m * 0.8); dots(46, ['#ffcf7a', '#ff9a6a', '#7ab8ff', '#ffe6b0'], 0.02, 0.07, 0.55); }
    else if (id === 'office') {
      lin(['#d8d2c8', '#b9b1a4', '#8c8476']);
      x.save(); x.filter = 'blur(' + Math.round(m * 0.01) + 'px)';
      x.fillStyle = 'rgba(255,255,245,.55)'; x.fillRect(w * 0.06, h * 0.08, w * 0.38, h * 0.42);
      x.fillStyle = 'rgba(90,70,50,.55)'; for (let i = 0; i < 4; i++) x.fillRect(w * 0.58, h * (0.12 + i * 0.12), w * 0.36, h * 0.012);
      const cols = ['#6b4f3a', '#3f5d78', '#8a3c3c', '#4f6b3f', '#c9a25a'];
      for (let r = 0; r < 4; r++) for (let i = 0; i < 9; i++) { x.fillStyle = cols[(i + r) % 5]; x.fillRect(w * (0.6 + i * 0.037), h * (0.075 + r * 0.12), w * 0.028, h * 0.043); }
      x.fillStyle = 'rgba(60,90,50,.8)'; x.beginPath(); x.ellipse(w * 0.16, h * 0.7, w * 0.12, h * 0.1, 0, 0, 7); x.fill();
      x.restore();
    }
    // a little grain so the gradients do not band in the video
    const img = x.getImageData(0, 0, w, h), d = img.data;
    for (let i = 0; i < d.length; i += 4) { const n = (rnd() - 0.5) * 6; d[i] += n; d[i + 1] += n; d[i + 2] += n; }
    x.putImageData(img, 0, 0);
    return cv;
  }

  /* ---------------- engine ---------------- */
  const E = {
    video: null, canvas: null, gl: null, fx: Object.assign({}, DEF), running: false, locked: false, track: null,
    W: 0, H: 0, maxLong: 1920, prog: {}, tex: {}, fbo: {}, quad: null, ping: 0, firstMask: true,
    worker: null, seq: 0, held: null, lips: null, lipsT: 0, wantMesh: false, wState: 'off', inflight: false, lastSend: 0, segOK: false, faceOK: false, wantSeg: false, wantFace: false,
    face: null, faceT: 0, aiMs: 0, fps: 0, frames: 0, fpsT: 0, bgKey: '', userImg: null, onStat: null, raf: 0, vfc: 0, lost: false
  };

  function compile(type, src) { const gl = E.gl, s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; }
  function program(fs) {
    const gl = E.gl, p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, VS)); gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.bindAttribLocation(p, 0, 'p'); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {}; const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const a = gl.getActiveUniform(p, i); u[a.name.replace(/\[0\]$/, '')] = gl.getUniformLocation(p, a.name); }
    return { p, u };
  }
  function mkTex(w, h) {
    const gl = E.gl, t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (w) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    return { t, w, h };
  }
  function mkTarget(w, h) {
    const gl = E.gl, tx = mkTex(w, h), f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tx.t, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { t: tx.t, w, h, f };
  }
  function freeTarget(x) { if (!x) return; const gl = E.gl; gl.deleteTexture(x.t); if (x.f) gl.deleteFramebuffer(x.f); }

  function initGL() {
    if (E.gl) return true;
    const cv = E.canvas;
    const gl = cv.getContext('webgl', { alpha: false, antialias: false, depth: false, stencil: false, premultipliedAlpha: false, preserveDrawingBuffer: false, powerPreference: 'high-performance' });
    if (!gl) return false;
    E.gl = gl;
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true); gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    for (const k in FS) E.prog[k] = program(FS[k]);
    E.quad = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, E.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    E.tex.vid = mkTex(0, 0);
    E.tex.mraw = mkTex(0, 0); gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, 1, 1, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, new Uint8Array([255]));
    E.tex.lip = mkTex(0, 0); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
    E.tex.img = mkTex(0, 0); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([40, 40, 40, 255]));
    cv.addEventListener('webglcontextlost', e => { e.preventDefault(); E.lost = true; stopLoop(); stat('Máy tạm dừng hiệu ứng (hết bộ nhớ đồ hoạ). Tắt camera rồi bật lại.'); }, false);
    return true;
  }
  function resize(w, h) {
    if (w === E.W && h === E.H) return;
    E.W = w; E.H = h; E.canvas.width = w; E.canvas.height = h;
    const w2 = Math.ceil(w / 2), h2 = Math.ceil(h / 2), w4 = Math.ceil(w / 4), h4 = Math.ceil(h / 4);
    ['s1', 's2', 'q', 'm0', 'm1', 'b1', 'b2'].forEach(k => { freeTarget(E.fbo[k]); });
    E.fbo.s1 = mkTarget(w2, h2); E.fbo.s2 = mkTarget(w2, h2);
    E.fbo.q = mkTarget(w4, h4); E.fbo.m0 = mkTarget(w4, h4); E.fbo.m1 = mkTarget(w4, h4);
    E.fbo.b1 = mkTarget(w4, h4); E.fbo.b2 = mkTarget(w4, h4);
    E.firstMask = true; E.bgKey = '';
  }
  function bind(unit, t) { const gl = E.gl; gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t); }
  function draw(pr, target, setU) {
    const gl = E.gl; gl.useProgram(pr.p);
    if (target) { gl.bindFramebuffer(gl.FRAMEBUFFER, target.f); gl.viewport(0, 0, target.w, target.h); }
    else { gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, E.W, E.H); }
    setU(pr.u, gl); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  function needPipe() { const f = E.fx; return f.lip > 0 || f.smooth > 0 || f.bright > 0 || f.tone !== 0 || f.light > 0 || f.filter !== 'none' || f.bg !== 'none'; }
  function needSeg() { return E.fx.bg !== 'none' || E.fx.light > 0; }
  function needFace() { return (E.fx.smooth > 0 && E.fx.faceOnly) || E.fx.light > 0; }
  function needMesh() { return E.fx.lip > 0; }
  // wait for each frame's own mask before showing it, so edges and lips never trail behind movement
  function syncMode() { return E.wState === 'ready' && ((E.fx.bg !== 'none' && E.wantSeg) || (needMesh() && E.wantMesh)); }

  /* ---------------- AI worker ---------------- */
  function workerUp() {
    const seg = needSeg(), face = needFace(), mesh = needMesh();
    if (!seg && !face && !mesh) return;
    if (E.wState === 'failed') return;
    if (!E.worker) {
      try { E.worker = new Worker(new URL('beauty-worker.js', SRC), { type: 'module' }); }
      catch (e) { E.wState = 'failed'; stat('Trình duyệt này chưa chạy được AI nhận diện. Hãy cập nhật Chrome.'); return; }
      E.wState = 'loading'; E.wantSeg = seg; E.wantFace = face; E.wantMesh = mesh;
      stat('Đang tải AI nhận diện người (lần đầu khoảng 4 MB)…');
      E.worker.onmessage = onWorker;
      E.worker.onerror = () => { E.wState = 'failed'; stat('Không tải được AI nhận diện. Kiểm tra mạng rồi thử lại.'); };
      E.worker.postMessage({ type: 'init', seg, face, mesh, gpu: false });
      return;
    }
    if ((seg && !E.wantSeg) || (face && !E.wantFace) || (mesh && !E.wantMesh)) {
      E.wantSeg = E.wantSeg || seg; E.wantFace = E.wantFace || face; E.wantMesh = E.wantMesh || mesh;
      if (mesh) stat('Đang tải AI nhận diện môi (lần đầu khoảng 4 MB)…');
      if (E.wState === 'ready') { E.wState = 'loading'; E.worker.postMessage({ type: 'need', seg: E.wantSeg, face: E.wantFace, mesh: E.wantMesh }); }
    }
  }
  function onWorker(e) {
    const m = e.data;
    if (m.type === 'ready') { E.wState = 'ready'; stat(''); return; }
    if (m.type === 'error') { E.wState = 'failed'; stat('Máy này chưa chạy được AI nhận diện (' + m.msg + ').'); return; }
    if (m.type !== 'result') return;
    E.inflight = false;
    const held = E.held && E.held.id === m.id ? E.held : null; if (held) E.held = null;
    if (m.err) { if (held) held.bmp.close(); return; }
    if ('lips' in m) { E.lips = m.lips; E.lipsT = performance.now(); drawLips(); }
    E.aiMs = E.aiMs ? E.aiMs * 0.8 + m.ms * 0.2 : m.ms;
    if (m.mask && E.gl) {
      const gl = E.gl; gl.bindTexture(gl.TEXTURE_2D, E.tex.mraw.t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, m.mw, m.mh, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, m.mask);
      if (!E.segOK) E.firstMask = true;
      E.segOK = true;
    }
    if ('face' in m) {
      if (m.face) {
        const f = m.face, k = f.k || [];
        const nf = { cx: f.x + f.w / 2, cy: f.y + f.h * 0.52, w: f.w, h: f.h, eyes: [k[0], k[1]], mouth: k[3] };
        const o = E.face, a = 0.55;
        if (o && Math.abs(o.cx - nf.cx) < 0.15) {
          const mix = (p, q) => p + (q - p) * a, mp = (p, q) => p && q ? [mix(p[0], q[0]), mix(p[1], q[1])] : q || p;
          E.face = { cx: mix(o.cx, nf.cx), cy: mix(o.cy, nf.cy), w: mix(o.w, nf.w), h: mix(o.h, nf.h), eyes: [mp(o.eyes[0], nf.eyes[0]), mp(o.eyes[1], nf.eyes[1])], mouth: mp(o.mouth, nf.mouth) };
        } else E.face = nf;
        E.faceT = performance.now(); E.faceOK = true;
      }
    }
    if (held) {
      if (E.running) { try { render(performance.now(), held.bmp); tick(performance.now()); } catch (err) { console.warn('[beauty]', err); } }
      held.bmp.close();
    }
  }
  function drawLips() {
    if (!E.gl || !E.W) return;
    const w = Math.ceil(E.W / 2), h = Math.ceil(E.H / 2);
    let cv = E.lipCv; if (!cv || cv.width !== w || cv.height !== h) { cv = E.lipCv = document.createElement('canvas'); cv.width = w; cv.height = h; }
    const x = cv.getContext('2d'); x.clearRect(0, 0, w, h);
    const L = E.lips;
    if (L) {
      const path = pts => { x.beginPath(); pts.forEach((p, i) => { const X = p[0] * w, Y = p[1] * h; if (i) x.lineTo(X, Y); else x.moveTo(X, Y); }); x.closePath(); };
      x.filter = 'blur(' + Math.max(1, w / 360).toFixed(1) + 'px)';
      x.fillStyle = '#fff'; path(L.o); x.fill();
      x.globalCompositeOperation = 'destination-out'; path(L.i); x.fill();
      x.globalCompositeOperation = 'source-over'; x.filter = 'none';
    }
    const gl = E.gl; gl.bindTexture(gl.TEXTURE_2D, E.tex.lip.t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, cv);
  }
  function sendFrame(now, sync) {
    if (E.wState !== 'ready' || E.inflight || !E.video.videoWidth) return false;
    const seg = needSeg() && E.wantSeg, mesh = needMesh() && E.wantMesh, face = needFace() && E.wantFace && !mesh;
    if (!seg && !face && !mesh) return false;
    if (!sync && now - E.lastSend < 30) return false;
    E.lastSend = now; E.inflight = true;
    const id = ++E.seq, vw = E.video.videoWidth, vh = E.video.videoHeight, sw = vw >= vh ? 320 : 256, sh = Math.round(sw * vh / vw);
    const small = full => createImageBitmap(full || E.video, { resizeWidth: sw, resizeHeight: sh, resizeQuality: 'medium' });
    const go = b => { if (!E.worker) { b.close(); E.inflight = false; return; } E.worker.postMessage({ type: 'frame', bitmap: b, ts: now, seg, face, mesh: !!mesh, id }, [b]); };
    if (sync) {
      // keep this exact camera frame; it is shown when its own mask comes back
      createImageBitmap(E.video, { imageOrientation: 'flipY' })
        .then(full => { if (E.held) E.held.bmp.close(); E.held = { id, bmp: full, t: now }; return createImageBitmap(full, { imageOrientation: 'flipY', resizeWidth: sw, resizeHeight: sh, resizeQuality: 'medium' }); })
        .then(go).catch(() => { E.inflight = false; });
    } else small().then(go).catch(() => { E.inflight = false; });
    setTimeout(() => { if (E.inflight && performance.now() - E.lastSend > 3000) E.inflight = false; }, 3100);
    return true;
  }

  /* ---------------- background image ---------------- */
  function bgSource() {
    const f = E.fx;
    if (f.bg === 'user') return E.userImg;
    const pr = BGS.find(b => b.id === f.bg); if (!pr) return null;
    const portrait = E.H >= E.W, w = portrait ? 720 : 1280, h = portrait ? 1280 : 720;
    return drawPreset(pr.id, w, h);
  }
  function ensureBg() {
    const f = E.fx; if (f.bg === 'none' || f.bg === 'blur') return;
    const key = f.bg + ':' + (E.H >= E.W ? 'p' : 'l') + ':' + (f.bg === 'user' ? (E.userImg ? E.userImg.width + 'x' + E.userImg.height + E.userStamp : 'none') : '');
    if (key === E.bgKey) return;
    E.bgKey = key;
    const src = bgSource(), gl = E.gl;
    gl.bindTexture(gl.TEXTURE_2D, E.tex.img.t);
    if (src) { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src); E.imgW = src.width; E.imgH = src.height; }
    else { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([40, 40, 40, 255])); E.imgW = E.imgH = 1; }
  }

  /* ---------------- render ---------------- */
  function frame() {
    if (!E.running) return;
    const vid = E.video, now = performance.now();
    if (vid.readyState >= 2 && vid.videoWidth) {
      if (syncMode()) {
        // the frame is drawn when its mask returns (onWorker); if the AI stalls, show live video meanwhile
        if (!sendFrame(now, true) && now - E.lastSend > 400) { try { render(now); tick(now); } catch (e) { console.warn('[beauty]', e); } }
      } else {
        try { render(now); tick(now); } catch (e) { console.warn('[beauty]', e); }
        sendFrame(now, false);
      }
    }
    schedule();
  }
  function tick(now) {
    if (E.snapCb) { const cb = E.snapCb; E.snapCb = null; cb(E.canvas.toDataURL('image/jpeg', 0.9)); }
    E.frames++;
    if (now - E.fpsT > 1000) { E.fps = E.frames * 1000 / (now - E.fpsT); E.frames = 0; E.fpsT = now; stat(); }
  }
  function schedule() {
    if (!E.running) return;
    if (E.video.requestVideoFrameCallback) E.vfc = E.video.requestVideoFrameCallback(frame);
    else E.raf = requestAnimationFrame(frame);
  }
  function render(now, src) {
    const gl = E.gl, vid = src || E.video, f = E.fx;
    let vw = src ? src.width : vid.videoWidth, vh = src ? src.height : vid.videoHeight;
    const sc = Math.min(1, E.maxLong / Math.max(vw, vh));
    if (!E.locked) { const pw = E.W; resize(Math.round(vw * sc / 2) * 2, Math.round(vh * sc / 2) * 2); if (pw !== E.W) drawLips(); }
    const W = E.W, H = E.H;
    bind(0, E.tex.vid.t); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, vid);
    const seg = E.segOK && needSeg();
    // face geometry, in output pixels, y up
    let faceOn = 0, fu = [W / 2, H * 0.62, W * 0.2, H * 0.12], ea = [-1e4, -1e4, 1, 1], eb = ea, mo = ea;
    if (E.face && now - E.faceT < 800) {
      const F = E.face, X = x => x * W, Y = y => (1 - y) * H, fw = F.w * W, fh = F.h * H;
      fu = [X(F.cx), Y(F.cy), fw * 0.62, fh * 0.78];
      const er = [fw * 0.15, fw * 0.085];
      if (F.eyes[0]) ea = [X(F.eyes[0][0]), Y(F.eyes[0][1]) + fw * 0.02, er[0], er[1]];
      if (F.eyes[1]) eb = [X(F.eyes[1][0]), Y(F.eyes[1][1]) + fw * 0.02, er[0], er[1]];
      if (F.mouth) mo = [X(F.mouth[0]), Y(F.mouth[1]) - fw * 0.03, fw * 0.22, fw * 0.11];
      faceOn = f.faceOnly ? 1 : 0;
    }
    const P = E.prog, s1 = E.fbo.s1, s2 = E.fbo.s2, q = E.fbo.q;
    if (f.smooth > 0) {
      const st = Math.max(1, Math.min(s1.w, s1.h) / 400), k = 140 - 100 * f.smooth / 100;
      bind(0, E.tex.vid.t);
      draw(P.bil, s1, (u, g) => { g.uniform1i(u.t, 0); g.uniform2f(u.d, st / s1.w, 0); g.uniform1f(u.k, k); });
      bind(0, s1.t);
      draw(P.bil, s2, (u, g) => { g.uniform1i(u.t, 0); g.uniform2f(u.d, 0, st / s2.h); g.uniform1f(u.k, k); });
    }
    if (seg) {
      bind(0, E.tex.vid.t);
      draw(P.down, q, (u, g) => { g.uniform1i(u.t, 0); g.uniform2f(u.px, 1 / vw, 1 / vh); });
      const src = E.ping ? E.fbo.m1 : E.fbo.m0, dst = E.ping ? E.fbo.m0 : E.fbo.m1;
      bind(0, E.tex.mraw.t); bind(1, q.t); bind(2, src.t);
      draw(P.mask, dst, (u, g) => { g.uniform1i(u.m, 0); g.uniform1i(u.g, 1); g.uniform1i(u.pv, 2); g.uniform2f(u.px, 1 / q.w, 1 / q.h); g.uniform1f(u.a0, E.firstMask ? 1 : 0); });
      E.firstMask = false; E.ping ^= 1; E.mask = dst;
      if (f.bg === 'blur') {
        const st = 0.5 + 2.4 * f.blur / 100, b1 = E.fbo.b1, b2 = E.fbo.b2;
        bind(0, q.t); bind(1, dst.t);
        draw(P.blur0, b1, (u, g) => { g.uniform1i(u.t, 0); g.uniform1i(u.m, 1); g.uniform2f(u.d, st / q.w, 0); });
        bind(0, b1.t); draw(P.blurp, b2, (u, g) => { g.uniform1i(u.t, 0); g.uniform2f(u.d, 0, st / q.h); });
        bind(0, b2.t); draw(P.blurp, b1, (u, g) => { g.uniform1i(u.t, 0); g.uniform2f(u.d, st * 1.4 / q.w, 0); });
        bind(0, b1.t); draw(P.blurp, b2, (u, g) => { g.uniform1i(u.t, 0); g.uniform2f(u.d, 0, st * 1.4 / q.h); });
      } else ensureBg();
    }
    const fl = FILTERS.find(x => x.id === f.filter) || FILTERS[0];
    let fit = [1, 1];
    if (f.bg !== 'none' && f.bg !== 'blur' && E.imgW) { const ia = E.imgW / E.imgH, oa = W / H; fit = ia > oa ? [oa / ia, 1] : [1, ia / oa]; }
    bind(0, E.tex.vid.t); bind(1, s2.t); bind(2, E.mask ? E.mask.t : E.tex.mraw.t); bind(3, E.fbo.b2.t); bind(4, E.tex.img.t); bind(5, E.tex.lip.t);
    draw(P.comp, null, (u, g) => {
      g.uniform1i(u.vid, 0); g.uniform1i(u.smo, 1); g.uniform1i(u.msk, 2); g.uniform1i(u.bgb, 3); g.uniform1i(u.bgi, 4); g.uniform1i(u.lipm, 5);
      const lc = (LIPS.find(x => x.id === f.lipColor) || LIPS[0]).c, hx = i => parseInt(lc.slice(i, i + 2), 16) / 255;
      g.uniform1f(u.uLip, f.lip > 0 && E.lips && now - E.lipsT < 600 ? f.lip / 100 : 0); g.uniform3f(u.uLipCol, hx(1), hx(3), hx(5));
      g.uniform2f(u.res, W, H);
      g.uniform1f(u.uSmooth, f.smooth / 100); g.uniform1f(u.uBright, f.bright / 100); g.uniform1f(u.uTone, f.tone / 50); g.uniform1f(u.uLight, f.light / 100);
      g.uniform1f(u.uBg, f.bg === 'none' ? 0 : f.bg === 'blur' ? 1 : 2); g.uniform1f(u.uSeg, seg ? 1 : 0); g.uniform1f(u.uFaceOn, faceOn); g.uniform1f(u.uEdge, f.edge / 100);
      g.uniform4fv(u.uFace, fu); g.uniform4fv(u.uEyeA, ea); g.uniform4fv(u.uEyeB, eb); g.uniform4fv(u.uMouth, mo); g.uniform4f(u.uFit, fit[0], fit[1], 0, 0);
      g.uniform3f(u.uF1, fl.exp || 1, fl.con || 1, fl.sat == null ? 1 : fl.sat);
      g.uniform3f(u.uF2, fl.temp || 0, fl.tint || 0, fl.fade || 0);
      g.uniform3fv(u.uSh, fl.sh || [0, 0, 0]); g.uniform3fv(u.uHi, fl.hi || [0, 0, 0]);
      g.uniform2f(u.uF3, fl.vig || 0, fl.id === 'none' ? 0 : f.fstr / 100);
    });
  }

  function startLoop() {
    if (E.running || E.lost) return;
    if (!initGL()) { stat('Máy này không hỗ trợ WebGL, chưa dùng được hiệu ứng.'); return; }
    E.running = true; E.fpsT = performance.now(); E.frames = 0;
    E.canvas.classList.add('on');
    schedule();
  }
  function stopLoop() {
    E.running = false;
    if (E.vfc && E.video.cancelVideoFrameCallback) E.video.cancelVideoFrameCallback(E.vfc);
    cancelAnimationFrame(E.raf); E.vfc = E.raf = 0;
    if (E.canvas) E.canvas.classList.remove('on');
  }
  function update() {
    if (!E.video || !E.video.srcObject) { stopLoop(); return; }
    if (needPipe() || E.locked) { workerUp(); startLoop(); } else stopLoop();
    stat();
  }
  function stat(msg) {
    if (msg !== undefined) E.msg = msg;
    if (!E.onStat) return;
    let s = E.msg || '';
    if (!s && E.running) {
      s = Math.round(E.fps) + ' khung/giây';
      if (E.wState === 'ready' && (needSeg() || needFace() || needMesh())) s += ' · AI ' + Math.round(E.aiMs) + ' ms';
      if (E.fps && E.fps < 20) s += ' · máy hơi chậm, thử giảm hiệu ứng hoặc chọn Nhẹ';
      if (needSeg() && E.wState === 'ready' && !E.segOK) s = 'Đang nhận diện người…';
    }
    E.onStat(s, needPipe());
  }

  const SRC = document.currentScript ? document.currentScript.src : location.href;
  window.CueBeauty = {
    FILTERS, BGS, LIPS, DEF, drawPreset,
    attach(video, canvas, onStat) { E.video = video; E.canvas = canvas; E.onStat = onStat; },
    set(fx) { E.fx = Object.assign({}, DEF, fx); update(); },
    setQuality(maxLong) { if (E.locked) return; E.maxLong = maxLong; },
    setUserImage(img) { E.userImg = img; E.userStamp = Date.now(); E.bgKey = ''; },
    refresh: update, // camera (re)opened or closed
    active: () => E.running,
    // the stream to record: the canvas when a look is on, so the saved video has it
    recordTrack() {
      if (!E.running) return null;
      if (!E.track || E.track.readyState === 'ended') E.track = E.canvas.captureStream(30).getVideoTracks()[0];
      E.locked = true; return E.track;
    },
    unlock() { E.locked = false; update(); },
    stop() { E.locked = false; stopLoop(); },
    snap: () => new Promise(ok => { E.snapCb = ok; }), // test helper: next rendered frame as JPEG
    info: () => ({ running: E.running, fps: E.fps, aiMs: E.aiMs, w: E.W, h: E.H, worker: E.wState, seg: E.segOK, face: E.face })
  };
})();
