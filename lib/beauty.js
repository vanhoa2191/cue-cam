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
    { id: 'nude', name: 'Nude', c: '#b9776b' }, { id: 'rose', name: 'Hồng đất', c: '#b0505e' }, { id: 'coral', name: 'Cam đào', c: '#d0604f' },
    { id: 'pink', name: 'Hồng tươi', c: '#cf4a72' }, { id: 'red', name: 'Đỏ cổ điển', c: '#a8202f' }, { id: 'berry', name: 'Đỏ mận', c: '#8a2b48' }
  ];
  const DEF = { dark: 0, eyes: 0, lip: 0, lipColor: 'rose', smooth: 0, bright: 0, tone: 0, light: 0, faceOnly: true, filter: 'none', fstr: 80, bg: 'none', blur: 60, edge: 50, hq: false };

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
    // local person colour (mode 0) / background colour (mode 1) around each point, for colour matting at the edge
    fb: HP + `uniform sampler2D t, m; uniform vec2 px; uniform float mode;
      void main(){ vec4 s = vec4(0.0); float ws = 0.0;
        for (int y = -2; y <= 2; y++) for (int x = -2; x <= 2; x++) {
          vec2 o = v + vec2(float(x), float(y)) * px * 1.8; float mm = texture2D(m, o).r;
          float w = mode < 0.5 ? smoothstep(0.75, 0.95, mm) : 1.0 - smoothstep(0.05, 0.25, mm);
          s += vec4(texture2D(t, o).rgb * w, w); ws += 1.0; }
        gl_FragColor = s / ws; }`,
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
    comp: HP + `uniform sampler2D vid, smo, msk, bgb, bgi, lipm, fgt, bgt; uniform float uLip; uniform vec3 uLipCol;
      uniform vec2 res; uniform float uDark, uEyes, uFaceOk, uSmooth, uBright, uTone, uLight, uBg, uSeg, uFaceOn, uEdge, uHq;
      uniform vec4 uFace, uEyeA, uEyeB, uMouth; uniform vec4 uFit; uniform vec2 uMpx, uCrop, uCropC;
      uniform vec3 uF1, uF2, uSh, uHi; uniform vec2 uF3;
      float ell(vec2 p, vec2 c, vec2 r){ return length((p - c) / r); }
      float luma(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }
      vec3 setLum(vec3 c, float l){ c += l - luma(c); float L = luma(c), n = min(min(c.r, c.g), c.b), x = max(max(c.r, c.g), c.b);
        if (n < 0.0) c = L + (c - L) * L / max(L - n, 1e-4); if (x > 1.0) c = L + (c - L) * (1.0 - L) / max(x - L, 1e-4); return c; }
      void main(){
        vec2 sv = (v - 0.5) * uCrop + uCropC; // output -> camera frame (centre crop to the chosen 9:16 / 16:9)
        vec3 c0 = texture2D(vid, sv).rgb;
        vec3 c = texture2D(vid, sv).rgb; vec2 p = v * res;
        float pm = 1.0;
        if (uSeg > 0.5) { // 5-tap tent on the half-res mask: no staircase on the edge after upscaling
          vec2 o = uMpx * 0.75;
          pm = 0.4 * texture2D(msk, sv).r + 0.15 * (texture2D(msk, sv + o).r + texture2D(msk, sv - o).r + texture2D(msk, sv + vec2(o.x, -o.y)).r + texture2D(msk, sv + vec2(-o.x, o.y)).r);
        }
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
          vec3 s = texture2D(smo, sv).rgb;
          float k = uSmooth * skin * reg * min(1.0, pm * 1.5);
          c = mix(c, s, k * 0.88);
        }
        vec4 fm = texture2D(lipm, sv); // face-part masks from the 478 landmarks: r lips, g under-eye, b eyes
        if (uDark > 0.0) {
          // dark circles: lift and de-blue the crescent drawn under each lower eyelid
          // keep the skin texture: brighten the pixel itself, warm it a little, and only partly smooth
          float kd = fm.g * uDark;
          vec3 s2 = texture2D(smo, sv).rgb; vec3 base = mix(c, s2, 0.35 * kd);
          vec3 lift = base * (1.0 + 0.32 * kd) * vec3(1.0 + 0.03 * kd, 1.0, 1.0 - 0.05 * kd);
          c = clamp(lift, 0.0, 1.0);
        }
        if (uEyes > 0.0) {
          // brighter eyes: a little more contrast and light inside the eye opening
          vec3 ec = (c - 0.5) * 1.2 + 0.5 + 0.045;
          c = mix(c, clamp(ec, 0.0, 1.0), fm.b * uEyes * 0.75);
        }
        if (uBright > 0.0) {
          float bw = uBright * (0.45 + 0.55 * skin * reg);
          c = c + (1.0 - c) * c * 0.75 * bw;
          c = mix(c, vec3(luma(c)), 0.12 * bw * skin);
        }
        if (uLip > 0.0) {
          // lipstick as a tint, not paint: take the colour's hue/saturation but keep the lip's own brightness
          // (texture, creases, shading), slightly deepened, and let shine/highlights show through
          float lm = fm.r * uLip, l = luma(c);
          vec3 t = setLum(uLipCol, l * (0.8 + 0.3 * luma(uLipCol)));
          float hw = 1.0 - 0.75 * smoothstep(0.62, 0.9, l);
          c = mix(c, t, lm * hw * 0.8);
        }
        c *= vec3(1.0 + 0.08 * uTone, 1.0 + 0.012 * uTone, 1.0 - 0.08 * uTone);
        // background
        float a = 1.0;
        if (uBg > 0.5 && uSeg > 0.5) {
          // the HQ matte is a real alpha (hair strands are partly see-through): use it almost as is
          float lo = mix(mix(0.36, 0.16, uEdge), mix(0.2, 0.03, uEdge), uHq), hi = mix(mix(0.64, 0.84, uEdge), mix(0.8, 0.97, uEdge), uHq);
          a = smoothstep(lo, hi, pm);
          // colour matting at full resolution: in the uncertain band, decide by how close this pixel's colour is to
          // the nearby person colour vs background colour. Edges then follow the real outline (sharp, no stairs).
          vec4 F = texture2D(fgt, sv), B = texture2D(bgt, sv);
          if (F.a > 0.04 && B.a > 0.04) {
            vec3 fc = F.rgb / F.a, bc = B.rgb / B.a, dfb = fc - bc; float dd = dot(dfb, dfb);
            float ac = clamp(dot(c0 - bc, dfb) / max(dd, 1e-4), 0.0, 1.0);
            float band = smoothstep(0.08, 0.3, pm) * (1.0 - smoothstep(0.7, 0.92, pm));
            float conf = smoothstep(0.02, 0.09, dd);
            a = mix(a, smoothstep(0.15, 0.85, ac), band * conf * mix(0.8, 0.45, uEdge) * (1.0 - 0.7 * uHq));
            // HQ: semi-transparent edge pixels still carry the old background's light (a pale halo);
            // pull their colour towards the nearby person colour before laying them on the new background
            c = mix(c, fc, uHq * (1.0 - a) * smoothstep(0.02, 0.5, a) * 0.85);
          }
          vec3 bg;
          if (uBg < 1.5) { vec4 b = texture2D(bgb, sv); bg = b.a > 0.004 ? b.rgb / b.a : c; }
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
    face: null, faceT: 0, aiMs: 0, hqState: 'off', hqMs: 0, hqSlow: 0, fps: 0, frames: 0, fpsT: 0, bgKey: '', userImg: null, onStat: null, raf: 0, vfc: 0, lost: false
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
  function resize(w, h, sw, sh) { // output size, and the (scaled) camera frame size the work buffers follow
    sw = Math.round(sw || w); sh = Math.round(sh || h);
    if (w === E.W && h === E.H && sw === E.SW && sh === E.SH) return;
    E.W = w; E.H = h; E.SW = sw; E.SH = sh; E.canvas.width = w; E.canvas.height = h;
    const w2 = Math.ceil(sw / 2), h2 = Math.ceil(sh / 2), w4 = Math.ceil(sw / 4), h4 = Math.ceil(sh / 4);
    ['s1', 's2', 'h', 'q', 'm0', 'm1', 'b1', 'b2', 'f1', 'f2'].forEach(k => { freeTarget(E.fbo[k]); });
    E.fbo.s1 = mkTarget(w2, h2); E.fbo.s2 = mkTarget(w2, h2);
    E.fbo.q = mkTarget(w4, h4); E.fbo.h = mkTarget(w2, h2); E.fbo.m0 = mkTarget(w2, h2); E.fbo.m1 = mkTarget(w2, h2);
    E.fbo.b1 = mkTarget(w4, h4); E.fbo.b2 = mkTarget(w4, h4); E.fbo.f1 = mkTarget(w4, h4); E.fbo.f2 = mkTarget(w4, h4);
    E.firstMask = true; E.bgKey = '';
  }
  function bind(unit, t) { const gl = E.gl; gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t); }
  function draw(pr, target, setU) {
    const gl = E.gl; gl.useProgram(pr.p);
    if (target) { gl.bindFramebuffer(gl.FRAMEBUFFER, target.f); gl.viewport(0, 0, target.w, target.h); }
    else { gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, E.W, E.H); }
    setU(pr.u, gl); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  function needPipe() { const f = E.fx; return needCrop() || f.dark > 0 || f.eyes > 0 || f.lip > 0 || f.smooth > 0 || f.bright > 0 || f.tone !== 0 || f.light > 0 || f.filter !== 'none' || f.bg !== 'none'; }
  // Android hands a portrait stream to a phone held upright, whatever we ask: crop it to the chosen frame
  function needCrop() { const v = E.video; if (!E.aspect || !v || !v.videoWidth) return false; return Math.abs(v.videoWidth / v.videoHeight - E.aspect) > 0.03; }
  function needSeg() { return E.fx.bg !== 'none' || E.fx.light > 0; }
  function needFace() { return (E.fx.smooth > 0 && E.fx.faceOnly) || E.fx.light > 0; }
  function needMesh() { return E.fx.lip > 0 || E.fx.dark > 0 || E.fx.eyes > 0; }
  // wait for each frame's own mask before showing it, so edges and lips never trail behind movement
  function syncMode() { return E.wState === 'ready' && ((E.fx.bg !== 'none' && E.wantSeg) || (needMesh() && E.wantMesh)); }

  /* ---------------- AI worker ---------------- */
  function workerUp() {
    const seg = needSeg(), face = needFace(), mesh = needMesh();
    if (!seg && !face && !mesh) return;
    if (E.wState === 'failed') return;
    if (!E.worker) {
      try { E.worker = new Worker(new URL('beauty-worker.js?v=15', SRC), { type: 'module' }); }
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
      if (mesh) stat('Đang tải AI nhận diện khuôn mặt chi tiết (lần đầu khoảng 4 MB)…');
      if (E.wState === 'ready') { E.wState = 'loading'; E.worker.postMessage({ type: 'need', seg: E.wantSeg, face: E.wantFace, mesh: E.wantMesh }); }
    }
    hqUp();
  }
  function wantHQ() { return E.fx.hq && E.fx.bg !== 'none'; }
  function hqOn() { return wantHQ() && E.hqState === 'ready'; }
  function hqUp() {
    if (!wantHQ() || E.hqState !== 'off' || !E.worker || E.wState !== 'ready') return;
    E.hqState = 'loading'; E.hqSlow = 0; E.hqMs = 0;
    stat('Đang tải AI Chất lượng cao (lần đầu khoảng 30 MB), trong lúc chờ vẫn dùng chế độ thường…');
    E.worker.postMessage({ type: 'hq', maxMs: E.hqTest ? 1e9 : 150 });
  }
  function hqFail(msg) {
    E.hqState = 'failed'; E.firstMask = true; stat(msg);
    setTimeout(() => { if (E.msg === msg) stat(''); }, 8000);
    if (E.onHq) E.onHq(false, msg);
  }
  function onWorker(e) {
    const m = e.data;
    if (m.type === 'ready') { E.wState = 'ready'; stat(''); hqUp(); return; }
    if (m.type === 'hq') {
      if (m.ok) { E.hqState = 'ready'; E.firstMask = true; stat(''); if (E.onHq) E.onHq(true); }
      else hqFail(/nogpu/.test(m.msg) ? 'Máy hoặc trình duyệt này chưa có WebGPU nên chưa chạy được Chất lượng cao. Đang dùng chế độ thường.'
        : /slow/.test(m.msg) ? 'Máy chạy Chất lượng cao hơi chậm (' + m.msg.split(':')[1] + ' ms mỗi khung), đã về chế độ thường cho video mượt.'
        : 'Chưa chạy được Chất lượng cao (' + m.msg + '). Đang dùng chế độ thường.');
      return;
    }
    if (m.type === 'error') { E.wState = 'failed'; stat('Máy này chưa chạy được AI nhận diện (' + m.msg + ').'); return; }
    if (m.type !== 'result') return;
    E.inflight = false;
    const held = E.held && E.held.id === m.id ? E.held : null; if (held) E.held = null;
    if (m.err) { if (held) held.bmp.close(); return; }
    if ('lips' in m) {
      const o = E.lips, n = m.lips, k = 0.6; // a touch of smoothing against jitter, small enough to follow speech
      E.lips = o && n && o.o.length === n.o.length && Math.abs(o.o[0][0] - n.o[0][0]) < 0.03 ? { e: n.e, o: n.o.map((p, i) => [o.o[i][0] + (p[0] - o.o[i][0]) * k, o.o[i][1] + (p[1] - o.o[i][1]) * k]), i: n.i.map((p, i) => [o.i[i][0] + (p[0] - o.i[i][0]) * k, o.i[i][1] + (p[1] - o.i[i][1]) * k]) } : n;
      E.lipsT = performance.now(); drawLips();
    }
    E.aiMs = E.aiMs ? E.aiMs * 0.8 + m.ms * 0.2 : m.ms;
    if (m.hq) { // keep watching: if the phone heats up and slows down, go back to the light mode
      E.hqMs = E.hqMs ? E.hqMs * 0.9 + m.ms * 0.1 : m.ms;
      E.hqSlow = E.hqMs > 110 ? E.hqSlow + 1 : 0;
      if (E.hqSlow > 45 && !E.locked && !E.hqTest) hqFail('Máy chạy Chất lượng cao chậm dần (' + Math.round(E.hqMs) + ' ms), đã về chế độ thường cho video mượt.');
    }
    if (m.mask && E.gl && !!m.hq !== hqOn() && E.segOK) m.mask = null; // a mask from the other mode, right after switching
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
    const w = Math.ceil((E.SW || E.W) / 1.5), h = Math.ceil((E.SH || E.H) / 1.5);
    let cv = E.lipCv; if (!cv || cv.width !== w || cv.height !== h) { cv = E.lipCv = document.createElement('canvas'); cv.width = w; cv.height = h; }
    const x = cv.getContext('2d'); x.clearRect(0, 0, w, h);
    const L = E.lips;
    if (L) {
      // smooth closed curve through the landmarks (midpoint quadratics), so the lip line is not angular
      const path = pts => {
        const P = pts.map(p => [p[0] * w, p[1] * h]), n = P.length, mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
        x.beginPath(); let m0 = mid(P[n - 1], P[0]); x.moveTo(m0[0], m0[1]);
        for (let i = 0; i < n; i++) { const m = mid(P[i], P[(i + 1) % n]); x.quadraticCurveTo(P[i][0], P[i][1], m[0], m[1]); }
        x.closePath();
      };
      x.filter = 'blur(' + Math.max(0.8, w / 600).toFixed(2) + 'px)';
      // fade out towards the mouth corners, where real lipstick thins out
      const a = L.o[0], b = L.o[10], g = x.createLinearGradient(a[0] * w, a[1] * h, b[0] * w, b[1] * h);
      g.addColorStop(0, 'rgba(255,0,0,0)'); g.addColorStop(0.16, '#f00'); g.addColorStop(0.84, '#f00'); g.addColorStop(1, 'rgba(255,0,0,0)');
      x.fillStyle = g; path(L.o); x.fill();
      x.globalCompositeOperation = 'destination-out'; path(L.i); x.fill();
      x.globalCompositeOperation = 'lighter';
      for (const [up, lo] of L.e || []) {
        const P = q => [q[0] * w, q[1] * h], a0 = P(lo[0]), a1 = P(lo[lo.length - 1]);
        const ew = Math.hypot(a1[0] - a0[0], a1[1] - a0[1]); if (ew < 2) continue;
        let nx = -(a1[1] - a0[1]) / ew, ny = (a1[0] - a0[0]) / ew; if (ny < 0) { nx = -nx; ny = -ny; } // perpendicular, pointing down the face
        // under-eye crescent: from just below the lower lid down to ~0.45 eye-width at its middle (green channel)
        const top = lo.map(q => { const p = P(q); return [p[0] + nx * ew * 0.07, p[1] + ny * ew * 0.07]; });
        const bot = lo.map((q, i) => { const p = P(q), t = i / (lo.length - 1), d = ew * (0.09 + 0.24 * Math.sin(Math.PI * Math.min(1, t * 1.1))); return [p[0] + nx * d, p[1] + ny * d]; });
        x.filter = 'blur(' + (ew * 0.09).toFixed(1) + 'px)'; x.fillStyle = 'rgb(0,200,0)';
        x.beginPath(); top.forEach((p, i) => i ? x.lineTo(p[0], p[1]) : x.moveTo(p[0], p[1])); bot.reverse().forEach(p => x.lineTo(p[0], p[1])); x.closePath(); x.fill();
        // eye opening (blue channel)
        x.filter = 'blur(' + Math.max(0.6, ew * 0.03).toFixed(1) + 'px)'; x.fillStyle = 'rgb(0,0,255)';
        x.beginPath(); up.map(P).forEach((p, i) => i ? x.lineTo(p[0], p[1]) : x.moveTo(p[0], p[1])); lo.slice().reverse().map(P).forEach(p => x.lineTo(p[0], p[1])); x.closePath(); x.fill();
      }
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
        // the lip tracker needs a sharper picture than the person mask: the mouth is only a few dozen pixels at 256 wide
    const bw = Math.min(vw, vw >= vh ? 960 : 720), bh = Math.round(bw * vh / vw);
    const go = (b, big, hqb) => {
      if (!E.worker) { [b, big, hqb].forEach(x => x && x.close()); E.inflight = false; return; }
      E.worker.postMessage({ type: 'frame', bitmap: b, big, hqb, ts: now, seg, face, mesh: !!mesh, id }, [b, big, hqb].filter(Boolean));
    };
    // MODNet input: long side 448, both sides a multiple of 32
    const hq = seg && hqOn(), r32 = x => Math.max(32, Math.round(x / 32) * 32);
    const hw = vw >= vh ? 448 : r32(448 * vw / vh), hh = vw >= vh ? r32(448 * vh / vw) : 448;
    const pair = (src, flip) => {
      const o = { resizeQuality: 'medium' }; if (flip) o.imageOrientation = 'flipY';
      const hi = Object.assign({ resizeQuality: 'high' }, flip ? { imageOrientation: 'flipY' } : {});
      return Promise.all([createImageBitmap(src, Object.assign({ resizeWidth: sw, resizeHeight: sh }, o)),
        mesh ? createImageBitmap(src, Object.assign({ resizeWidth: bw, resizeHeight: bh }, hi)) : null,
        hq ? createImageBitmap(src, Object.assign({ resizeWidth: hw, resizeHeight: hh }, hi)) : null]);
    };
    if (sync) {
      // keep this exact camera frame; it is shown when its own mask comes back
      createImageBitmap(E.video, { imageOrientation: 'flipY' })
        .then(full => { if (E.held) E.held.bmp.close(); E.held = { id, bmp: full, t: now }; return pair(full, true); })
        .then(([b, big, hqb]) => go(b, big, hqb)).catch(() => { E.inflight = false; });
    } else pair(E.video, false).then(([b, big, hqb]) => go(b, big, hqb)).catch(() => { E.inflight = false; });
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
    let ow = vw, oh = vh; const ta = E.aspect || vw / vh;
    if (vw / vh > ta) ow = vh * ta; else oh = vw / ta;
    const sc = Math.min(1, E.maxLong / Math.max(ow, oh));
    if (!E.locked) { const pw = E.W; resize(Math.round(ow * sc / 2) * 2, Math.round(oh * sc / 2) * 2, vw * sc, vh * sc); if (pw !== E.W) drawLips(); }
    const W = E.W, H = E.H, oa = W / H, va = vw / vh;
    const cx = va > oa ? oa / va : 1, cy = va > oa ? 1 : va / oa;
    // cutting a landscape band out of an upright picture: keep it high, where the face is (uv y is up)
    const ccx = 0.5, ccy = cy < 1 ? Math.min(1 - cy / 2, Math.max(cy / 2, 0.74)) : 0.5;
    bind(0, E.tex.vid.t); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, vid);
    const seg = E.segOK && needSeg();
    // face geometry, in output pixels, y up
    let faceOk = 0, faceOn = 0, fu = [W / 2, H * 0.62, W * 0.2, H * 0.12], ea = [-1e4, -1e4, 1, 1], eb = ea, mo = ea;
    if (E.face && now - E.faceT < 800) {
      const F = E.face, X = x => ((x - ccx) / cx + 0.5) * W, Y = y => ((1 - y - ccy) / cy + 0.5) * H, fw = F.w * W / cx, fh = F.h * H / cy;
      fu = [X(F.cx), Y(F.cy), fw * 0.62, fh * 0.78];
      const er = [fw * 0.15, fw * 0.085];
      if (F.eyes[0]) ea = [X(F.eyes[0][0]), Y(F.eyes[0][1]) + fw * 0.02, er[0], er[1]];
      if (F.eyes[1]) eb = [X(F.eyes[1][0]), Y(F.eyes[1][1]) + fw * 0.02, er[0], er[1]];
      if (F.mouth) mo = [X(F.mouth[0]), Y(F.mouth[1]) - fw * 0.03, fw * 0.22, fw * 0.11];
      faceOn = f.faceOnly ? 1 : 0; faceOk = 1;
    }
    const P = E.prog, s1 = E.fbo.s1, s2 = E.fbo.s2, q = E.fbo.q;
    if (f.smooth > 0 || f.dark > 0) {
      const st = Math.max(1, Math.min(s1.w, s1.h) / 400), k = 140 - 100 * Math.max(f.smooth, 50) / 100;
      bind(0, E.tex.vid.t);
      draw(P.bil, s1, (u, g) => { g.uniform1i(u.t, 0); g.uniform2f(u.d, st / s1.w, 0); g.uniform1f(u.k, k); });
      bind(0, s1.t);
      draw(P.bil, s2, (u, g) => { g.uniform1i(u.t, 0); g.uniform2f(u.d, 0, st / s2.h); g.uniform1f(u.k, k); });
    }
    if (seg) {
      bind(0, E.tex.vid.t);
      draw(P.down, q, (u, g) => { g.uniform1i(u.t, 0); g.uniform2f(u.px, 1 / vw, 1 / vh); });
      const src = E.ping ? E.fbo.m1 : E.fbo.m0, dst = E.ping ? E.fbo.m0 : E.fbo.m1;
      const hg = E.fbo.h; bind(0, E.tex.vid.t);
      draw(P.down, hg, (u, g) => { g.uniform1i(u.t, 0); g.uniform2f(u.px, 0.5 / vw, 0.5 / vh); });
      bind(0, E.tex.mraw.t); bind(1, hg.t); bind(2, src.t);
      draw(P.mask, dst, (u, g) => { g.uniform1i(u.m, 0); g.uniform1i(u.g, 1); g.uniform1i(u.pv, 2); const r = hqOn() ? 0.45 : 1.6; g.uniform2f(u.px, r / hg.w, r / hg.h); g.uniform1f(u.a0, E.firstMask ? 1 : 0); });
      E.firstMask = false; E.ping ^= 1; E.mask = dst;
      if (f.bg !== 'none') for (const [md, tg] of [[0, E.fbo.f1], [1, E.fbo.f2]]) {
        bind(0, q.t); bind(1, dst.t);
        draw(P.fb, tg, (u, g) => { g.uniform1i(u.t, 0); g.uniform1i(u.m, 1); g.uniform2f(u.px, 1 / q.w, 1 / q.h); g.uniform1f(u.mode, md); });
      }
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
    bind(0, E.tex.vid.t); bind(1, s2.t); bind(2, E.mask ? E.mask.t : E.tex.mraw.t); bind(3, E.fbo.b2.t); bind(4, E.tex.img.t); bind(5, E.tex.lip.t); bind(6, E.fbo.f1.t); bind(7, E.fbo.f2.t);
    draw(P.comp, null, (u, g) => {
      g.uniform1i(u.vid, 0); g.uniform1i(u.smo, 1); g.uniform1i(u.msk, 2); g.uniform1i(u.bgb, 3); g.uniform1i(u.bgi, 4); g.uniform1i(u.lipm, 5); g.uniform1i(u.fgt, 6); g.uniform1i(u.bgt, 7);
      const lc = (LIPS.find(x => x.id === f.lipColor) || LIPS[0]).c, hx = i => parseInt(lc.slice(i, i + 2), 16) / 255;
      g.uniform1f(u.uLip, f.lip > 0 && E.lips && now - E.lipsT < 600 ? f.lip / 100 : 0); g.uniform3f(u.uLipCol, hx(1), hx(3), hx(5));
      g.uniform2f(u.res, W, H);
      g.uniform1f(u.uSmooth, f.smooth / 100); const meshOk = E.lips && now - E.lipsT < 600; g.uniform1f(u.uDark, meshOk ? f.dark / 100 : 0); g.uniform1f(u.uEyes, meshOk ? f.eyes / 100 : 0); g.uniform1f(u.uFaceOk, faceOk); g.uniform1f(u.uBright, f.bright / 100); g.uniform1f(u.uTone, f.tone / 50); g.uniform1f(u.uLight, f.light / 100);
      g.uniform1f(u.uBg, f.bg === 'none' ? 0 : f.bg === 'blur' ? 1 : 2); g.uniform1f(u.uSeg, seg ? 1 : 0); g.uniform1f(u.uFaceOn, faceOn); g.uniform1f(u.uEdge, f.edge / 100); g.uniform1f(u.uHq, seg && hqOn() ? 1 : 0);
      g.uniform4fv(u.uFace, fu); g.uniform4fv(u.uEyeA, ea); g.uniform4fv(u.uEyeB, eb); g.uniform4fv(u.uMouth, mo); g.uniform4f(u.uFit, fit[0], fit[1], 0, 0); g.uniform2f(u.uMpx, 1 / E.fbo.m0.w, 1 / E.fbo.m0.h); g.uniform2f(u.uCrop, cx, cy); g.uniform2f(u.uCropC, ccx, ccy);
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
      if (hqOn()) s += ' · Chất lượng cao';
      if (E.fps && E.fps < 20) s += ' · máy hơi chậm, thử giảm hiệu ứng hoặc chọn Nhẹ';
      if (needSeg() && E.wState === 'ready' && !E.segOK) s = 'Đang nhận diện người…';
    }
    E.onStat(s, needPipe());
  }

  const SRC = document.currentScript ? document.currentScript.src : location.href;
  window.CueBeauty = {
    FILTERS, BGS, LIPS, DEF, drawPreset,
    attach(video, canvas, onStat) { E.video = video; E.canvas = canvas; E.onStat = onStat; },
    set(fx) { const was = E.fx.hq; E.fx = Object.assign({}, DEF, fx); if (E.fx.hq && !was && E.hqState === 'failed') E.hqState = 'off'; update(); },
    setAspect(a) { if (!E.locked) E.aspect = a; },
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
    // look at the current picture to suggest natural beauty values (Auto)
    analyze() {
      const v = E.video; if (!v || !v.videoWidth) return null;
      const w = 160, h = Math.round(160 * v.videoHeight / v.videoWidth), cv = document.createElement('canvas'); cv.width = w; cv.height = h;
      const x = cv.getContext('2d', { willReadFrequently: true }); x.drawImage(v, 0, 0, w, h);
      const d = x.getImageData(0, 0, w, h).data;
      const box = (cx, cy, rw, rh) => { // mean colour of a box (normalised coords, y down)
        let r = 0, g = 0, b = 0, n = 0;
        for (let yy = Math.max(0, Math.round((cy - rh) * h)); yy < Math.min(h, Math.round((cy + rh) * h)); yy++)
          for (let xx = Math.max(0, Math.round((cx - rw) * w)); xx < Math.min(w, Math.round((cx + rw) * w)); xx++) { const i = (yy * w + xx) * 4; r += d[i]; g += d[i + 1]; b += d[i + 2]; n++; }
        return n ? [r / n / 255, g / n / 255, b / n / 255] : null;
      };
      const L = c => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
      const F = E.face && performance.now() - E.faceT < 1500 ? E.face : null;
      const faceC = F ? box(F.cx, F.cy, F.w * 0.25, F.h * 0.25) : box(0.5, 0.35, 0.15, 0.1), all = box(0.5, 0.5, 0.5, 0.5);
      let under = null, cheek = null;
      if (F && F.eyes[0] && F.eyes[1]) {
        const ex = (F.eyes[0][0] + F.eyes[1][0]) / 2, ey = (F.eyes[0][1] + F.eyes[1][1]) / 2, ew = Math.abs(F.eyes[1][0] - F.eyes[0][0]);
        const u1 = box(F.eyes[0][0], ey + ew * 0.32, ew * 0.18, ew * 0.07), u2 = box(F.eyes[1][0], ey + ew * 0.32, ew * 0.18, ew * 0.07);
        const c1 = box(F.eyes[0][0], ey + ew * 0.75, ew * 0.18, ew * 0.1), c2 = box(F.eyes[1][0], ey + ew * 0.75, ew * 0.18, ew * 0.1);
        if (u1 && u2 && c1 && c2) { under = (L(u1) + L(u2)) / 2; cheek = (L(c1) + L(c2)) / 2; }
        void ex;
      }
      return { face: !!F, faceLuma: faceC ? L(faceC) : null, warmth: faceC ? faceC[0] - faceC[2] : null, sceneLuma: all ? L(all) : null, under, cheek };
    },
    maskCanvas: () => E.lipCv, // debug: the lips/under-eye/eye masks
    snap: () => new Promise(ok => { E.snapCb = ok; }), // test helper: next rendered frame as JPEG
    onHq(cb) { E.onHq = cb; },
    hqTest() { E.hqTest = true; }, // test helper: slow emulators may run the HQ mode anyway
    info: () => ({ hq: E.hqState, hqMs: E.hqMs, running: E.running, fps: E.fps, aiMs: E.aiMs, w: E.W, h: E.H, worker: E.wState, seg: E.segOK, face: E.face })
  };
})();
