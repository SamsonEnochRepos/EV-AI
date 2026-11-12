"use strict";
// Holographic face for the Live page: a procedural triangulated 3D head (about 2,200 points) over a glowing AI chip.
// It speaks with the mouth: the jaw hinges open and the lips narrow in step with EV's output level.
// Only draws on a <canvas>: it never touches page markup. Colours come from the theme's CSS variables.
// API (same shape as EVOrb): mount(canvas), unmount(), setState(s), setLevel(v), refreshColors(), plus stats() for tuning.

const EVFace = (() => {
  const TAU = Math.PI * 2;
  const rmq = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
  const reduced = () => !!(rmq && rmq.matches);

  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const sstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
  const g2 = (x, y, cx, cy, sx, sy) => { const a = (x - cx) / sx, b = (y - cy) / sy; return Math.exp(-(a * a + b * b)); };
  const ease = (cur, tgt, dt, rate) => tgt + (cur - tgt) * Math.exp(-rate * dt);
  const mix3 = (p, q, t) => [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t];
  const rgba = (c, a) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${clamp(a, 0, 1).toFixed(3)})`;

  // ---------------------------------------------------------------------------------------------- head shape
  // Cross-section anchors, crown to neck: [y, half-width, half-depth, centre z]. Units are about 11 cm; +z faces the viewer.
  const PROFILE = [
    [1.10, 0.00, 0.00, -0.12], [1.06, 0.26, 0.30, -0.12], [0.95, 0.46, 0.56, -0.12], [0.75, 0.58, 0.76, -0.11],
    [0.45, 0.63, 0.84, -0.10], [0.15, 0.64, 0.84, -0.10], [-0.15, 0.61, 0.80, -0.09], [-0.45, 0.55, 0.72, -0.06],
    [-0.70, 0.45, 0.60, 0.01], [-0.90, 0.29, 0.46, 0.14], [-1.02, 0.19, 0.36, 0.24], [-1.14, 0.28, 0.31, -0.10],
    [-1.35, 0.31, 0.32, -0.14], [-1.52, 0.36, 0.37, -0.14]
  ];
  const PT = PROFILE.map((_, i) => {
    const a = PROFILE[Math.max(0, i - 1)], b = PROFILE[Math.min(PROFILE.length - 1, i + 1)], dy = b[0] - a[0];
    return [(b[1] - a[1]) / dy, (b[2] - a[2]) / dy, (b[3] - a[3]) / dy];
  });
  const PO = [0, 0, 0];                        // shared output of prof()
  function prof(y) {
    let i = 0;
    while (i < PROFILE.length - 2 && y < PROFILE[i + 1][0]) i += 1;
    const p0 = PROFILE[i], p1 = PROFILE[i + 1], h = p1[0] - p0[0];
    const t = clamp((y - p0[0]) / h, 0, 1), t2 = t * t, t3 = t2 * t;
    const h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
    for (let k = 0; k < 3; k += 1) PO[k] = h00 * p0[k + 1] + h10 * h * PT[i][k] + h01 * p1[k + 1] + h11 * h * PT[i + 1][k];
    return PO;
  }

  // Facial relief: forward displacement of the front of the face at (x, y).
  function feat(x, y) {
    const ax = Math.abs(x);
    let d = 0;
    d += 0.05 * g2(x, y, 0, 0.62, 0.55, 0.30);              // forehead
    d += 0.08 * g2(ax, y, 0.27, 0.27, 0.17, 0.045);         // brow ridges
    d += 0.04 * g2(x, y, 0, 0.26, 0.10, 0.05);              // glabella
    d -= 0.12 * g2(ax, y, 0.29, 0.15, 0.14, 0.065);         // eye sockets
    d += 0.10 * g2(x, y, 0, 0.00, 0.05, 0.22);              // nose bridge
    d += 0.17 * g2(x, y, 0, -0.22, 0.085, 0.075);           // nose tip
    d += 0.07 * g2(ax, y, 0.15, -0.28, 0.05, 0.05);         // nostril wings
    d -= 0.05 * g2(x, y, 0, -0.37, 0.13, 0.03);             // under the nose
    d += 0.07 * g2(ax, y, 0.43, -0.08, 0.14, 0.10);         // cheekbones
    d -= 0.05 * g2(ax, y, 0.40, -0.36, 0.12, 0.12);         // cheek hollows
    d += 0.07 * g2(x, y, 0, -0.51, 0.19, 0.035);            // upper lip
    d -= 0.06 * g2(x, y, 0, -0.555, 0.22, 0.02);            // mouth line
    d += 0.08 * g2(x, y, 0, -0.615, 0.17, 0.04);            // lower lip
    d -= 0.05 * g2(x, y, 0, -0.70, 0.15, 0.025);            // groove above the chin
    d += 0.11 * g2(x, y, 0, -0.85, 0.15, 0.09);             // chin
    return d;
  }

  const N_EXP = 2.6;                                         // superellipse: a flatter face and sides than an ellipse
  const SE = 2 / N_EXP;
  const spow = (v) => Math.sign(v) * Math.pow(Math.abs(v), SE);
  function frontZ(x, y) {                                    // surface depth at (x, y) on the front of the face
    const p = prof(y), xn = clamp(x / p[0], -0.999, 0.999);
    return p[2] + p[1] * Math.pow(1 - Math.pow(Math.abs(xn), N_EXP), 1 / N_EXP) + feat(x, y);
  }

  // ---------------------------------------------------------------------------------------------- mesh
  const HALF = 24, NC = HALF * 2;                            // columns around the head, denser at the front
  const ang = new Float64Array(NC);
  {
    const K = 900, cum = new Float64Array(K + 1);
    let px = 0, pz = 0.84;
    for (let k = 1; k <= K; k += 1) {
      const a = (k / K) * Math.PI, c = Math.cos(a);
      const x = 0.63 * spow(Math.sin(a)), z = 0.84 * spow(c), w = 1 + 1.7 * Math.max(0, c) * Math.max(0, c);
      cum[k] = cum[k - 1] + Math.hypot(x - px, z - pz) * w;
      px = x; pz = z;
    }
    let k = 0;
    for (let j = 0; j <= HALF; j += 1) {
      const target = (cum[K] * j) / HALF;
      while (k < K && cum[k] < target) k += 1;
      const a = (k / K) * Math.PI;
      if (j <= HALF) ang[j] = a;
      if (j > 0 && j < HALF) ang[NC - j] = -a;
    }
    ang[0] = 0;
    ang[HALF] = Math.PI;
  }

  const YS = [1.095, 1.03, 0.96, 0.88, 0.80, 0.72, 0.64, 0.56];
  for (let k = 0; k <= 33; k += 1) YS.push(0.48 - 0.045 * k);  // the face, 4.5% steps, so lips and eyes get rows
  YS.push(-1.08, -1.18, -1.30, -1.42, -1.52);
  const NRING = YS.length, NV = NC * NRING;
  const MOUTH_ROW = YS.findIndex((y) => Math.abs(y + 0.555) < 0.01);   // the lip line
  const LOWER_ROW = MOUTH_ROW + 1;

  const bx = new Float32Array(NV), by = new Float32Array(NV), bz = new Float32Array(NV);
  const nx0 = new Float32Array(NV), ny0 = new Float32Array(NV), nz0 = new Float32Array(NV);
  const wJaw = new Float32Array(NV), wMouth = new Float32Array(NV), wUp = new Float32Array(NV), wBrow = new Float32Array(NV);
  const fEmph = new Float32Array(NV), fEar = new Float32Array(NV), fFront = new Float32Array(NV), fFade = new Float32Array(NV);
  for (let r = 0; r < NRING; r += 1) {
    const y = YS[r], p = prof(y), rx = p[0], rz = p[1], zc = p[2];
    for (let c = 0; c < NC; c += 1) {
      const v = r * NC + c, s = Math.sin(ang[c]), co = Math.cos(ang[c]);
      let x = rx * spow(s), z = zc + rz * spow(co);
      const sz = spow(co), fw = sstep(0.0, 0.45, sz);
      z += fw * feat(x, y);
      // ears: a flat flange standing out at the temples
      const ear = g2(0, y, 0, 0.03, 1, 0.20) * Math.exp(-Math.pow((z + 0.12) / 0.20, 2)) * sstep(0.72, 0.96, Math.abs(spow(s)));
      x += Math.sign(x) * 0.13 * ear;
      bx[v] = x; by[v] = y; bz[v] = z;
      fFront[v] = fw;
      fFade[v] = 1 - 0.98 * sstep(-0.97, -1.13, y);            // the neck dissolves into the beam of light
      fEar[v] = ear;
      // weights for the speaking rig
      const mz = Math.exp(-Math.pow(x / 0.30, 4));                         // the mouth zone, where the lips are
      const sharp = sstep(-0.555, -0.60, y), soft = sstep(-0.25, -0.72, y); // lips split cleanly, cheeks stretch softly
      let wj = (mz * sharp + (1 - mz) * soft) * (1 - sstep(-1.04, -1.24, y)) * sstep(-0.55, -0.12, z);
      const lowLip = sstep(-0.57, -0.60, y) * (1 - sstep(-0.68, -0.80, y)) * fw;
      const shape = clamp(1 - (x / 0.25) * (x / 0.25), 0, 1);
      wj *= 1 - 0.55 * (1 - shape) * lowLip;                                 // lip corners follow the jaw less
      wJaw[v] = wj;
      wMouth[v] = g2(x, y, 0, -0.56, 0.34, 0.11) * fw;
      wUp[v] = shape * (1 - sstep(-0.45, -0.38, y)) * sstep(-0.585, -0.555, y) * fw;
      wBrow[v] = g2(Math.abs(x), y, 0.27, 0.30, 0.22, 0.06) * fw;
      fEmph[v] = clamp(0.55 * g2(Math.abs(x), y, 0.29, 0.15, 0.17, 0.07) + 0.65 * g2(x, y, 0, -0.56, 0.26, 0.07)
        + 0.30 * g2(x, y, 0, -0.22, 0.10, 0.09) + 0.30 * g2(Math.abs(x), y, 0.27, 0.27, 0.19, 0.04), 0, 1) * fw;
    }
  }
  // rest-pose normals from the grid
  for (let r = 0; r < NRING; r += 1) {
    for (let c = 0; c < NC; c += 1) {
      const v = r * NC + c, l = r * NC + ((c + NC - 1) % NC), rr = r * NC + ((c + 1) % NC);
      const u = Math.max(0, r - 1) * NC + c, d = Math.min(NRING - 1, r + 1) * NC + c;
      const ax = bx[rr] - bx[l], ay = by[rr] - by[l], az = bz[rr] - bz[l];
      const cx = bx[d] - bx[u], cy = by[d] - by[u], cz = bz[d] - bz[u];
      let nx = ay * cz - az * cy, ny = az * cx - ax * cz, nz = ax * cy - ay * cx;
      const len = Math.hypot(nx, ny, nz) || 1;
      nx /= len; ny /= len; nz /= len;
      const p = prof(by[v]);
      if (nx * bx[v] + ny * (by[v] > 0.9 ? 1 : 0) + nz * (bz[v] - p[2]) < 0) { nx = -nx; ny = -ny; nz = -nz; }
      nx0[v] = nx; ny0[v] = ny; nz0[v] = nz;
    }
  }
  // edges: rings, meridians and one alternating diagonal per quad, so it reads as a triangulated mesh
  const edgeA = [], edgeB = [], edgeW = [];
  for (let r = 0; r < NRING; r += 1) {
    for (let c = 0; c < NC; c += 1) {
      const v = r * NC + c, cn = r * NC + ((c + 1) % NC);
      edgeA.push(v); edgeB.push(cn); edgeW.push(1);
      if (r < NRING - 1) {
        edgeA.push(v); edgeB.push((r + 1) * NC + c); edgeW.push(1);
        if ((r + c) % 2 === 0) { edgeA.push(v); edgeB.push((r + 1) * NC + ((c + 1) % NC)); } else { edgeA.push(cn); edgeB.push((r + 1) * NC + c); }
        edgeW.push(0.55);
      }
    }
  }
  const NE = edgeA.length, EA = Uint16Array.from(edgeA), EB = Uint16Array.from(edgeB), EW = Float32Array.from(edgeW);
  const mouthCols = [];                                       // columns spanning the lips, left to right
  for (let c = NC - 12; c < NC + 12; c += 1) { const cc = c % NC; if (Math.abs(bx[MOUTH_ROW * NC + cc]) <= 0.26 && bz[MOUTH_ROW * NC + cc] > 0) mouthCols.push(cc); }

  // ---------------------------------------------------------------------------------------------- chip, traces, particles
  let seed = 7;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; };
  const DIE = 0.78, PLATE = 1.22;
  const traces = [];                                          // polylines in the chip plane, from the plate edge outwards
  for (let side = 0; side < 4; side += 1) {
    for (let k = 0; k < 8; k += 1) {
      const lane = ((k + 0.5) / 8 - 0.5) * 2 * PLATE * 0.86;
      const len1 = 0.15 + rnd() * 0.45, len2 = 0.2 + rnd() * 0.5, jog = (rnd() < 0.5 ? -1 : 1) * (0.15 + rnd() * 0.25);
      const pts = [[PLATE, lane], [PLATE + len1, lane], [PLATE + len1 + Math.abs(jog), lane + jog], [PLATE + len1 + Math.abs(jog) + len2, lane + jog]];
      const q = pts.map(([a, b]) => (side === 0 ? [a, b] : side === 1 ? [-a, b] : side === 2 ? [b, a] : [b, -a]));
      let total = 0;
      const seg = [];
      for (let i = 1; i < q.length; i += 1) { const l = Math.hypot(q[i][0] - q[i - 1][0], q[i][1] - q[i - 1][1]); seg.push(l); total += l; }
      traces.push({ q, seg, total, phase: rnd(), speed: 0.35 + rnd() * 0.5 });
    }
  }
  const NPART = 70;
  const parts = Array.from({ length: NPART }, () => ({ ox: rnd() * 2 - 1, ph: rnd(), sp: 0.10 + rnd() * 0.22, sz: 0.8 + rnd() * 1.4 }));

  // ---------------------------------------------------------------------------------------------- state
  const PARAMS = {
    idle: { glow: 0.52, sweep: 0.20, pitch: 0.00, shim: 0.06, glitch: 0, ring: 0, chip: 0.30, read: 0, stream: 0.25, time: 1, err: 0 },
    listening: { glow: 0.78, sweep: 0.34, pitch: 0.06, shim: 0.18, glitch: 0, ring: 1, chip: 0.50, read: 0, stream: 0.4, time: 1, err: 0 },
    transcribing: { glow: 0.82, sweep: 0.80, pitch: 0.02, shim: 0.45, glitch: 0, ring: 0, chip: 0.60, read: 1, stream: 0.6, time: 1, err: 0 },
    thinking: { glow: 0.88, sweep: 1.00, pitch: -0.02, shim: 0.65, glitch: 0, ring: 0, chip: 0.70, read: 1, stream: 0.7, time: 1, err: 0 },
    acting: { glow: 1.00, sweep: 1.50, pitch: 0.01, shim: 0.70, glitch: 0, ring: 0, chip: 1.00, read: 0, stream: 1.0, time: 1, err: 0 },
    speaking: { glow: 0.92, sweep: 0.40, pitch: 0.02, shim: 0.18, glitch: 0, ring: 0, chip: 0.60, read: 0, stream: 0.5, time: 1, err: 0 },
    paused: { glow: 0.16, sweep: 0.02, pitch: 0.00, shim: 0.00, glitch: 0, ring: 0, chip: 0.08, read: 0, stream: 0, time: 0.03, err: 0 },
    error: { glow: 0.80, sweep: 0.50, pitch: 0.00, shim: 0.40, glitch: 1, ring: 0, chip: 0.50, read: 0, stream: 0.2, time: 1, err: 1 }
  };
  const cur = { glow: 0.5, sweep: 0.2, pitch: 0, shim: 0.06, glitch: 0, ring: 0, chip: 0.3, read: 0, stream: 0.25, time: 1, err: 0 };
  let cols = { a: [47, 123, 255], b: [143, 232, 255], accent: [62, 166, 255], hi: [125, 227, 255] };
  const RED = [255, 70, 80];

  let canvas = null, ctx = null, raf = 0, last = 0, ro = null;
  let state = "idle", level = 0, lastLevelAt = -9, smooth = 0, avg = 0, open = 0, vow = 0.5, jawVel = 0;
  let clock = 0, sweepY = 0, blinkIn = 2.5, blinkP = 1, gazeX = 0, gazeY = 0, nod = 0, nodVel = 0, nodCool = 0, browLift = 0;
  let ripple = 0, tearY = 0, tearDX = 0, tearIn = 0;
  let frameMs = 0, frames = 0;

  const px = new Float32Array(NV), py = new Float32Array(NV), pz = new Float32Array(NV), br = new Float32Array(NV);
  const NB = 8;
  const ebin = new Int8Array(NE), dbin = new Int8Array(NV);
  const R = new Float64Array(9);
  let camS = 100, camCx = 0, camCy = 0, dpr = 1;
  const CAM_D = 4.4, PIV = [0, -1.0, -0.12];
  let bobY = 0, tearActive = false;
  const tmp = { x: 0, y: 0, z: 0 };

  function project(x, y, z) {                                  // head-local point to canvas pixels (writes tmp)
    const X = x - PIV[0], Y = y - PIV[1], Z = z - PIV[2];
    const rx = R[0] * X + R[1] * Y + R[2] * Z + PIV[0], ry = R[3] * X + R[4] * Y + R[5] * Z + PIV[1] + bobY, rz = R[6] * X + R[7] * Y + R[8] * Z + PIV[2];
    const s = CAM_D / (CAM_D - rz);
    let sx = camCx + rx * camS * s;
    const sy = camCy - ry * camS * s;
    if (tearActive && Math.abs(sy - tearY) < camS * 0.09) sx += tearDX;
    tmp.x = sx; tmp.y = sy; tmp.z = rz;
    return tmp;
  }

  function refreshColors() {
    const cs = getComputedStyle(document.documentElement);
    const parse = (name, fallback) => {
      const m = /^#?([0-9a-f]{6})$/i.exec((cs.getPropertyValue(name) || "").trim());
      if (!m) return fallback;
      const n = parseInt(m[1], 16);
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    };
    const a = parse("--orb-a", cols.a), b = parse("--orb-b", cols.b);
    cols = { a, b, accent: parse("--accent", cols.accent), hi: parse("--accent2", b) };
  }

  function resize() {
    if (!canvas) return;
    dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(160, Math.round(canvas.clientWidth * dpr)), h = Math.max(160, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }

  // ---------------------------------------------------------------------------------------------- the chip pedestal
  function drawChip(W, H, S, t, tint) {
    const cx = W * 0.5, cy = H * 0.835, phi = 0.36, k = 0.46;
    const exx = Math.cos(phi) * S, exy = Math.sin(phi) * k * S, ezx = -Math.sin(phi) * S, ezy = Math.cos(phi) * k * S;
    const P = (x, y, z) => [cx + x * exx + z * ezx, cy + x * exy + z * ezy - y * S];
    const A = tint(cols.a), B = tint(cols.b), ACC = tint(cols.hi), AC2 = tint(cols.accent);
    const chip = cur.chip, g = cur.glow;

    // ground glow
    const gl = ctx.createRadialGradient(cx, cy + S * 0.1, S * 0.2, cx, cy + S * 0.1, S * 2.2);
    gl.addColorStop(0, rgba(A, 0.30 * g + 0.10 * chip)); gl.addColorStop(1, rgba(A, 0));
    ctx.fillStyle = gl;
    ctx.fillRect(0, 0, W, H);

    // traces with travelling pulses (acting pulses are brighter and faster)
    ctx.lineWidth = 1.1 * dpr; ctx.lineCap = "round"; ctx.lineJoin = "round";
    for (let i = 0; i < traces.length; i += 1) {
      const tr = traces[i], pts = tr.q.map(([x, z]) => P(x, 0, z));
      ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
      for (let j = 1; j < pts.length; j += 1) ctx.lineTo(pts[j][0], pts[j][1]);
      ctx.strokeStyle = rgba(A, 0.16 + 0.22 * chip);
      ctx.stroke();
      const end = pts[pts.length - 1];
      ctx.beginPath(); ctx.arc(end[0], end[1], 2.2 * dpr, 0, TAU); ctx.strokeStyle = rgba(B, 0.30 + 0.3 * chip); ctx.stroke();
      // a pulse running inwards, towards the chip
      const f = 1 - ((t * tr.speed * (0.4 + 1.6 * chip) + tr.phase) % 1);
      let d = f * tr.total, j = 0;
      while (j < tr.seg.length - 1 && d > tr.seg[j]) { d -= tr.seg[j]; j += 1; }
      const u = clamp(d / tr.seg[j], 0, 1);
      const qa = tr.q[j], qb = tr.q[j + 1], pp = P(qa[0] + (qb[0] - qa[0]) * u, 0, qa[1] + (qb[1] - qa[1]) * u);
      ctx.fillStyle = rgba(ACC, (0.25 + 0.7 * chip) * Math.sin(Math.PI * f));
      ctx.beginPath(); ctx.arc(pp[0], pp[1], (1.2 + 1.6 * chip) * dpr, 0, TAU); ctx.fill();
    }

    // plate
    const plate = [P(-PLATE, 0, -PLATE), P(PLATE, 0, -PLATE), P(PLATE, 0, PLATE), P(-PLATE, 0, PLATE)];
    const under = [P(-PLATE, -0.14, -PLATE), P(PLATE, -0.14, -PLATE), P(PLATE, -0.14, PLATE), P(-PLATE, -0.14, PLATE)];
    ctx.beginPath(); plate.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.closePath();
    ctx.fillStyle = rgba(A, 0.10 + 0.06 * chip); ctx.fill();
    ctx.strokeStyle = rgba(B, 0.45 + 0.3 * chip); ctx.lineWidth = 1.4 * dpr; ctx.stroke();
    ctx.lineWidth = 1 * dpr; ctx.strokeStyle = rgba(A, 0.38);
    ctx.beginPath();
    for (const i of [1, 2, 3]) { ctx.moveTo(plate[i][0], plate[i][1]); ctx.lineTo(under[i][0], under[i][1]); }
    ctx.moveTo(under[1][0], under[1][1]); ctx.lineTo(under[2][0], under[2][1]); ctx.lineTo(under[3][0], under[3][1]);
    ctx.stroke();
    // contact pads between the die and the plate edge
    ctx.fillStyle = rgba(B, 0.35 + 0.25 * chip);
    for (let i = 0; i < 12; i += 1) {
      const v = -PLATE + 0.22 + (i * (2 * PLATE - 0.44)) / 11;
      for (const e of [DIE + 0.16, PLATE - 0.12]) {
        for (const [x, z] of [[e, v], [-e, v], [v, e], [v, -e]]) { const p = P(x, 0, z); ctx.fillRect(p[0] - dpr, p[1] - dpr, 2 * dpr, 2 * dpr); }
      }
    }
    // die, raised
    const top = 0.16;
    const die = [P(-DIE, top, -DIE), P(DIE, top, -DIE), P(DIE, top, DIE), P(-DIE, top, DIE)];
    const dieBase = [P(-DIE, 0, -DIE), P(DIE, 0, -DIE), P(DIE, 0, DIE), P(-DIE, 0, DIE)];
    ctx.beginPath(); die.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.closePath();
    ctx.fillStyle = rgba(A, 0.22 + 0.12 * chip); ctx.fill();
    ctx.strokeStyle = rgba(B, 0.7 + 0.25 * chip); ctx.lineWidth = 1.6 * dpr; ctx.stroke();
    ctx.lineWidth = 1 * dpr; ctx.strokeStyle = rgba(A, 0.5);
    ctx.beginPath();
    for (const i of [1, 2, 3]) { ctx.moveTo(die[i][0], die[i][1]); ctx.lineTo(dieBase[i][0], dieBase[i][1]); }
    ctx.moveTo(dieBase[1][0], dieBase[1][1]); ctx.lineTo(dieBase[2][0], dieBase[2][1]); ctx.lineTo(dieBase[3][0], dieBase[3][1]);
    ctx.stroke();
    ctx.strokeStyle = rgba(A, 0.22);
    ctx.beginPath();
    for (let i = 1; i < 4; i += 1) {
      const f = -DIE + (2 * DIE * i) / 4, a = P(f, top, -DIE), b = P(f, top, DIE), c = P(-DIE, top, f), d = P(DIE, top, f);
      ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.moveTo(c[0], c[1]); ctx.lineTo(d[0], d[1]);
    }
    ctx.stroke();
    // the "AI" mark, written onto the die so it lies in the chip plane
    const o = P(0, top, 0);
    ctx.save();
    ctx.transform(exx / 40, exy / 40, ezx / 40, ezy / 40, o[0], o[1]);
    ctx.font = "800 46px Segoe UI, Arial, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    const pulse = 0.78 + 0.22 * Math.sin(t * (2 + 4 * chip));
    ctx.shadowColor = rgba(ACC, 1); ctx.shadowBlur = 16 * dpr;
    ctx.fillStyle = rgba(ACC, (0.55 + 0.4 * chip) * pulse);
    ctx.fillText("AI", 0, 2);
    ctx.shadowBlur = 0;
    ctx.lineWidth = 1.4; ctx.strokeStyle = rgba(mix3(ACC, [255, 255, 255], 0.5), 0.6 * pulse);
    ctx.strokeText("AI", 0, 2);
    ctx.restore();

    // projection rings above the die
    ctx.lineWidth = 1.2 * dpr;
    for (let i = 0; i < 3; i += 1) {
      const y = 0.55 + i * 0.30, rad = (1.25 - i * 0.2) * S, c = P(0, y, 0);
      ctx.setLineDash([rad * 0.35, rad * 0.22, rad * 0.08, rad * 0.22]);
      ctx.lineDashOffset = -t * (14 + i * 7) * dpr * (i % 2 ? -1 : 1);
      ctx.strokeStyle = rgba(i === 1 ? ACC : B, 0.20 + 0.25 * g);
      ctx.beginPath(); ctx.ellipse(c[0], c[1], rad, rad * k, 0, 0, TAU); ctx.stroke();
    }
    ctx.setLineDash([]);
    return { x: cx, y: P(0, 0.5, 0)[1] };
  }

  // ---------------------------------------------------------------------------------------------- frame
  function frame(ts) {
    raf = requestAnimationFrame(frame);
    if (!canvas || !ctx || document.hidden) { last = 0; return; }
    const t0 = performance.now();
    const now = ts / 1000, dt = Math.min(0.1, last ? now - last : 0.016);
    last = now;
    resize();
    const rm = reduced(), motion = rm ? 0.25 : 1;
    const W = canvas.width, H = canvas.height;
    const P = PARAMS[state] || PARAMS.idle;
    for (const key of Object.keys(P)) cur[key] = ease(cur[key], P[key], dt, 8);       // eased, 250 to 400 ms
    clock += dt * cur.time;
    const t = clock, speaking = state === "speaking";
    const lvlFresh = now - lastLevelAt < 0.6;

    // ---- speech: fast attack, slower release, plus a vowel-like width change and nods on emphasis
    smooth = ease(smooth, level, dt, level > smooth ? 30 : 9);
    avg = ease(avg, smooth, dt, 2.2);
    let target = 0;
    if (speaking) {
      if (lvlFresh) target = Math.pow(clamp(smooth / 0.55, 0, 1), 0.85);
      else target = clamp(0.5 + 0.28 * Math.sin(t * 17) * Math.sin(t * 5.3 + 1) + 0.2 * Math.sin(t * 11.1), 0, 1) * (Math.sin(t * 2.3) > -0.55 ? 1 : 0.1);
    }
    open = ease(open, target, dt, target > open ? 34 : 13);
    vow = ease(vow, 0.5 + 0.5 * Math.sin(t * 5.1) * Math.sin(t * 2.2 + 0.7), dt, 10);
    nodCool -= dt;
    if (speaking && target - avg > 0.22 && nodCool <= 0) { nodVel += 0.8 * motion; nodCool = 0.45; }
    nodVel += (-70 * nod - 9 * nodVel) * dt; nod += nodVel * dt;
    browLift = ease(browLift, (speaking ? 0.012 + 0.03 * open : 0) + (state === "listening" ? 0.035 : 0) + Math.max(0, nod) * 0.2, dt, 12);

    // ---- blink and gaze
    if (state !== "paused") blinkIn -= dt;
    if (blinkIn <= 0 && blinkP >= 1) { blinkP = 0; blinkIn = Math.random() < 0.12 ? 0.22 : 3 + Math.random() * 3; }
    if (blinkP < 1) blinkP = Math.min(1, blinkP + dt / 0.17);
    const blink = Math.sin(blinkP * Math.PI);
    const rd = (t * 0.55) % 1;
    const tgx = cur.read > 0.5 ? (rd * 2 - 1) * 0.045 : Math.sin(t * 0.4) * 0.008;
    const tgy = cur.read > 0.5 ? (Math.floor(t * 0.55) % 3 - 1) * 0.012 : 0;
    gazeX = ease(gazeX, tgx, dt, cur.read > 0.5 && rd < 0.06 ? 14 : 7);
    gazeY = ease(gazeY, tgy, dt, 6);

    // ---- pose
    const yaw = Math.sin(t * 0.45) * 0.21 * motion * (state === "listening" ? 0.6 : 1) + 0.03;
    const pitch = -0.04 + cur.pitch + nod * 0.9 + Math.sin(t * 0.3) * 0.012 * motion;
    const roll = Math.sin(t * 0.33) * 0.02 * motion;
    bobY = Math.sin(t * 1.25) * 0.022 * motion;
    const cyw = Math.cos(yaw), syw = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch), cr = Math.cos(roll), sr = Math.sin(roll);
    // R = Rz(roll) * Rx(pitch) * Ry(yaw)
    const m00 = cyw, m01 = 0, m02 = syw, m10 = sp * syw, m11 = cp, m12 = -sp * cyw, m20 = -cp * syw, m21 = sp, m22 = cp * cyw;
    R[0] = cr * m00 - sr * m10; R[1] = cr * m01 - sr * m11; R[2] = cr * m02 - sr * m12;
    R[3] = sr * m00 + cr * m10; R[4] = sr * m01 + cr * m11; R[5] = sr * m02 + cr * m12;
    R[6] = m20; R[7] = m21; R[8] = m22;
    camS = Math.min(H * 0.255, W * 0.22);
    camCx = W * 0.5; camCy = H * 0.355;

    // error stutter: a horizontal tear that jumps around
    tearActive = cur.glitch > 0.1 && !rm;
    if (tearActive) {
      tearIn -= dt;
      if (tearIn <= 0) { tearIn = 0.05 + Math.random() * 0.35; tearY = camCy + (Math.random() - 0.4) * camS * 1.6; tearDX = (Math.random() - 0.5) * camS * 0.22 * cur.glitch; if (Math.random() < 0.4) tearDX = 0; }
    }
    const tint = (c) => (cur.err > 0.01 ? mix3(c, RED, cur.err) : c);
    const A = tint(cols.a), B = tint(cols.b), ACC = tint(cols.hi), ACN = tint(cols.accent);
    const sweepSpd = cur.sweep * (rm ? 0.3 : 1);
    sweepY = (sweepY + dt * sweepSpd * 0.5) % 1;
    const ySw = 1.2 - sweepY * 2.4, sweepAmp = Math.sin(Math.PI * sweepY);   // the scan plane, in head units
    const flick = 1 - (rm ? 0 : 0.035 * Math.sin(t * 61) * Math.sin(t * 13.7));
    const gain = (0.55 + 0.6 * cur.glow) * flick * (1 + 0.25 * smooth * (speaking ? 1 : 0));

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.globalCompositeOperation = "lighter";

    // bloom behind the head
    const bl = ctx.createRadialGradient(camCx, camCy - camS * 0.05, camS * 0.15, camCx, camCy - camS * 0.05, camS * 1.7);
    bl.addColorStop(0, rgba(A, 0.34 * cur.glow * (1 + 0.5 * smooth))); bl.addColorStop(0.55, rgba(A, 0.10 * cur.glow)); bl.addColorStop(1, rgba(A, 0));
    ctx.fillStyle = bl; ctx.fillRect(0, 0, W, H);

    // pedestal, then the beam of light between chip and neck
    const chipTop = drawChip(W, H, camS * 0.8, t, tint);
    const neckY = camCy + camS * 1.0, beamTop = neckY;
    const bg = ctx.createLinearGradient(0, chipTop.y, 0, beamTop);
    bg.addColorStop(0, rgba(A, 0.30 * cur.glow + 0.10 * cur.chip)); bg.addColorStop(0.7, rgba(A, 0.07 * cur.glow)); bg.addColorStop(1, rgba(A, 0.0));
    ctx.fillStyle = bg;
    ctx.beginPath();
    ctx.moveTo(camCx - camS * 0.78, chipTop.y); ctx.lineTo(camCx + camS * 0.78, chipTop.y);
    ctx.lineTo(camCx + camS * 0.40, beamTop); ctx.lineTo(camCx - camS * 0.40, beamTop); ctx.closePath(); ctx.fill();
    for (let i = 0; i < NPART; i += 1) {                                           // data motes rising to the head
      const q = parts[i];
      if (i / NPART > 0.25 + 0.75 * cur.stream) continue;
      const f = (q.ph + t * q.sp * (0.6 + cur.stream * 1.4)) % 1;
      const halfW = camS * (0.74 - 0.34 * f);
      ctx.fillStyle = rgba(ACC, 0.65 * Math.sin(Math.PI * f) * (0.3 + 0.7 * cur.stream));
      const s = q.sz * dpr;
      ctx.fillRect(camCx + q.ox * halfW - s / 2, chipTop.y + (beamTop - chipTop.y) * f, s, s * 1.6);
    }

    // ---- head: pose, jaw and lips -> screen
    const jawA = open * 0.20 * (rm ? 0.8 : 1), ca = Math.cos(jawA), sa = Math.sin(jawA);
    const lipN = 0.10 * open + 0.16 * vow * open;                                   // the lips narrow as the mouth opens
    const lipP = 0.05 * vow * open;
    const emphGain = (state === "listening" ? 0.35 : 0) + 0.15;
    const L0 = -0.45, L1 = 0.62, L2 = 0.65, shimAmp = cur.shim;
    for (let v = 0; v < NV; v += 1) {
      let x = bx[v], y = by[v], z = bz[v];
      const wm = wMouth[v];
      if (wm > 0.01 && open > 0.005) { x *= 1 - wm * lipN; z += wm * lipP; }
      const wj = wJaw[v];
      if (wj > 0.001 && jawA > 0.002) {
        const dy = y - (-0.10), dz = z - (-0.12);
        y += wj * (dy * ca - dz * sa - dy);
        z += wj * (dy * sa + dz * ca - dz);
        const wu = wUp[v];
        if (wu > 0) y -= wu * open * 0.006;
      }
      if (wUp[v] > 0 && open > 0.005) y += wUp[v] * open * 0.02;
      if (browLift > 0.001) y += wBrow[v] * browLift;
      const p = project(x, y, z);
      px[v] = p.x; py[v] = p.y; pz[v] = p.z;
      // shading from the rotated rest normal: bright rims, facets facing the light, features, and the scan sweep
      const nxr = R[0] * nx0[v] + R[1] * ny0[v] + R[2] * nz0[v], nyr = R[3] * nx0[v] + R[4] * ny0[v] + R[5] * nz0[v], nzr = R[6] * nx0[v] + R[7] * ny0[v] + R[8] * nz0[v];
      const facing = nzr > 0 ? 1 : 0.30;
      const rim = Math.pow(1 - Math.abs(nzr), 2);
      const lamb = clamp(nxr * L0 + nyr * L1 + nzr * L2, 0, 1);
      let b = 0.20 + 0.55 * rim + 0.35 * lamb + fEmph[v] * (0.55 + emphGain) * (1 + 0.5 * smooth);
      if (state === "listening") b += fEar[v] * (0.5 + 0.8 * smooth) + wBrow[v] * 0.4;
      b *= facing * fFade[v];
      const ds = Math.abs(y - ySw);
      if (ds < 0.14) b += (1 - ds / 0.14) * 0.7 * sweepAmp;
      if (shimAmp > 0.02) b *= 1 + shimAmp * 0.55 * Math.sin(t * 13 + v * 1.7) * (rm ? 0.2 : 1);
      br[v] = b * gain;
    }

    // ---- mesh: translucent facets by brightness band, then edges, then vertex points
    ctx.lineWidth = 1 * dpr; ctx.lineCap = "round";
    for (let e = 0; e < NE; e += 1) {
      const b = (br[EA[e]] + br[EB[e]]) * 0.5 * EW[e];
      ebin[e] = b < 0.06 ? -1 : Math.min(NB - 1, Math.floor(b * (NB - 0.01) / 1.15));
    }
    for (let k = 0; k < NB; k += 1) {
      const f = (k + 0.5) / NB;
      ctx.beginPath();
      let any = false;
      for (let e = 0; e < NE; e += 1) {
        if (ebin[e] !== k) continue;
        ctx.moveTo(px[EA[e]], py[EA[e]]); ctx.lineTo(px[EB[e]], py[EB[e]]); any = true;
      }
      if (any) { ctx.strokeStyle = rgba(mix3(A, B, f), 0.10 + 0.82 * f); ctx.stroke(); }
    }
    for (let v = 0; v < NV; v += 1) { const b = br[v]; dbin[v] = pz[v] < -0.45 || b < 0.10 ? -1 : Math.min(NB - 1, Math.floor(b * (NB - 0.01) / 1.15)); }
    for (let k = 1; k < NB; k += 1) {
      const f = (k + 0.5) / NB, s = (1.0 + 1.4 * f) * dpr;
      ctx.beginPath();
      let any = false;
      for (let v = 0; v < NV; v += 1) { if (dbin[v] === k) { ctx.rect(px[v] - s / 2, py[v] - s / 2, s, s); any = true; } }
      if (any) { ctx.fillStyle = rgba(mix3(B, [255, 255, 255], f * 0.5), 0.35 + 0.65 * f); ctx.fill(); }
      if (k >= NB - 3) {                                                               // a soft halo on the brightest points
        ctx.beginPath();
        for (let v = 0; v < NV; v += 1) { if (dbin[v] === k) ctx.rect(px[v] - s * 1.6, py[v] - s * 1.6, s * 3.2, s * 3.2); }
        ctx.fillStyle = rgba(B, 0.07); ctx.fill();
      }
    }

    // ---- brows, eyes and mouth drawn over the mesh so the face reads at a glance
    const lw = dpr;
    for (const sgn of [-1, 1]) {                                                       // brows
      ctx.beginPath();
      for (let i = 0; i <= 8; i += 1) {
        const u = i / 8, x = sgn * (0.10 + 0.40 * u), y = 0.285 + 0.05 * Math.sin(u * Math.PI * 0.9) - 0.015 * u + browLift * (0.6 + 0.8 * u) + (state === "thinking" && sgn > 0 ? 0.018 : 0);
        const p = project(x, y, frontZ(x, y) + 0.035);
        if (i) ctx.lineTo(p.x, p.y); else ctx.moveTo(p.x, p.y);
      }
      ctx.strokeStyle = rgba(B, 0.55 * gain); ctx.lineWidth = 1.8 * lw; ctx.stroke();
    }
    for (const sgn of [-1, 1]) {                                                       // eyes
      const ex = sgn * 0.29, ey = 0.15, lid = 1 - 0.96 * blink;
      ctx.beginPath();
      for (let i = 0; i < 18; i += 1) {
        const a = (i / 18) * TAU, sy = Math.sin(a);
        const x = ex + 0.15 * Math.cos(a), y = ey + 0.058 * sy * (sy > 0 ? lid : 1 - 0.3 * blink) + 0.016 * sgn * -Math.cos(a) * 0 + 0.014 * (Math.abs(Math.cos(a)) > 0.8 ? 1 : 0) * (x * sgn > ex * sgn ? 1 : 0);
        const p = project(x, y, frontZ(x, y) + 0.03);
        if (i) ctx.lineTo(p.x, p.y); else ctx.moveTo(p.x, p.y);
      }
      ctx.closePath();
      ctx.strokeStyle = rgba(B, 0.85 * gain); ctx.lineWidth = 1.5 * lw; ctx.stroke();
      if (lid > 0.2) {
        const ix = ex + gazeX, iy = ey + gazeY, p = project(ix, iy, frontZ(ix, iy) + 0.06);
        const rr = camS * 0.036 * (1 + 0.1 * smooth);
        const ig = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, rr * 2.6);
        ig.addColorStop(0, rgba(mix3(ACC, [255, 255, 255], 0.6), 0.95 * lid * gain)); ig.addColorStop(0.35, rgba(ACC, 0.65 * lid * gain)); ig.addColorStop(1, rgba(ACC, 0));
        ctx.fillStyle = ig; ctx.beginPath(); ctx.arc(p.x, p.y, rr * 2.6, 0, TAU); ctx.fill();
      }
    }
    // the nose, as lines: bridge, tip and nostril wings
    {
      ctx.strokeStyle = rgba(B, 0.45 * gain); ctx.lineWidth = 1.3 * lw;
      for (const sgn of [-1, 1]) {
        ctx.beginPath();
        for (let i = 0; i <= 6; i += 1) {
          const u = i / 6, x = sgn * (0.055 + 0.035 * u * u), y = 0.11 - 0.33 * u, p = project(x, y, frontZ(x, y) + 0.03);
          if (i) ctx.lineTo(p.x, p.y); else ctx.moveTo(p.x, p.y);
        }
        ctx.stroke();
      }
      ctx.beginPath();
      for (let i = 0; i <= 14; i += 1) {
        const u = i / 14, x = (u * 2 - 1) * 0.16, y = -0.24 - 0.10 * (1 - Math.pow(Math.abs(u * 2 - 1), 1.6)) * 0.9 + (Math.abs(u * 2 - 1) > 0.7 ? 0.04 : 0), p = project(x, y, frontZ(x, y) + 0.03);
        if (i) ctx.lineTo(p.x, p.y); else ctx.moveTo(p.x, p.y);
      }
      ctx.stroke();
    }
    if (mouthCols.length > 3) {                                                        // lips and the glowing opening
      const ru = MOUTH_ROW * NC, rl = LOWER_ROW * NC;
      ctx.beginPath();
      mouthCols.forEach((c, i) => (i ? ctx.lineTo(px[ru + c], py[ru + c]) : ctx.moveTo(px[ru + c], py[ru + c])));
      for (let i = mouthCols.length - 1; i >= 0; i -= 1) ctx.lineTo(px[rl + mouthCols[i]], py[rl + mouthCols[i]]);
      ctx.closePath();
      ctx.fillStyle = rgba(ACC, 0.04 + 0.26 * open * gain); ctx.fill();
      for (const row of [ru, rl]) {
        ctx.beginPath();
        mouthCols.forEach((c, i) => (i ? ctx.lineTo(px[row + c], py[row + c]) : ctx.moveTo(px[row + c], py[row + c])));
        ctx.strokeStyle = rgba(mix3(B, ACC, 0.4), (0.50 + 0.4 * open) * gain); ctx.lineWidth = 1.7 * lw; ctx.stroke();
      }
    }

    // ---- listening: ripples of the microphone level, circling the head
    if (cur.ring > 0.02) {
      ripple = (ripple + dt * (0.35 + 0.7 * smooth)) % 1;
      for (let r2 = 0; r2 < 2; r2 += 1) {
        const f = (ripple + r2 * 0.5) % 1, rad = 0.85 + 0.35 * f;
        ctx.beginPath();
        for (let i = 0; i <= 56; i += 1) {
          const a = (i / 56) * TAU, p = project(Math.cos(a) * rad, 0.12, -0.1 + Math.sin(a) * rad);
          if (i) ctx.lineTo(p.x, p.y); else ctx.moveTo(p.x, p.y);
        }
        ctx.strokeStyle = rgba(ACN, cur.ring * (1 - f) * (0.12 + 0.6 * smooth)); ctx.lineWidth = 1.4 * dpr; ctx.stroke();
      }
    }

    // ---- scan plane: a ring that rises through the head, then the fine scan lines
    if (sweepAmp > 0.03) {
      const pr = prof(clamp(ySw, -1.0, 1.05)), rxs = pr[0] * 1.08 + 0.03, rzs = pr[1] * 1.04 + 0.03, zcs = pr[2];
      for (const [dy, al, wd] of [[0, 0.55, 1.6], [0.035, 0.18, 1.0], [-0.035, 0.18, 1.0]]) {
        ctx.beginPath();
        for (let i = 0; i <= 48; i += 1) {
          const a = (i / 48) * TAU, p = project(Math.sin(a) * rxs, ySw + dy, zcs + Math.cos(a) * rzs);
          if (i) ctx.lineTo(p.x, p.y); else ctx.moveTo(p.x, p.y);
        }
        ctx.strokeStyle = rgba(B, al * sweepAmp * cur.glow); ctx.lineWidth = wd * dpr; ctx.stroke();
      }
    }
    ctx.globalCompositeOperation = "source-over";
    ctx.beginPath();                                                                // fine scan lines, one batched path
    const step = Math.max(3, Math.round(3 * dpr)), lh = Math.max(1, Math.round(dpr));
    for (let yy = 0; yy < H; yy += step) ctx.rect(0, yy, W, lh);
    ctx.fillStyle = "rgba(255,255,255,0.035)"; ctx.fill();

    frameMs += performance.now() - t0; frames += 1;
  }

  function onVisibility() {
    if (!canvas) return;
    cancelAnimationFrame(raf); raf = 0; last = 0;
    if (!document.hidden) raf = requestAnimationFrame(frame);                     // nothing is drawn, or scheduled, while hidden
  }

  return {
    mount(el) {
      this.unmount();
      canvas = el; ctx = el.getContext("2d");
      refreshColors(); last = 0; frameMs = 0; frames = 0;
      document.addEventListener("visibilitychange", onVisibility);
      if (window.ResizeObserver) { ro = new ResizeObserver(() => resize()); ro.observe(el); }
      if (!document.hidden) raf = requestAnimationFrame(frame);
    },
    unmount() {
      cancelAnimationFrame(raf); raf = 0;
      document.removeEventListener("visibilitychange", onVisibility);
      if (ro) { ro.disconnect(); ro = null; }
      canvas = null; ctx = null;
    },
    setState(s) { state = PARAMS[s] ? s : "idle"; },
    setLevel(v) { level = clamp(Number(v) || 0, 0, 1); lastLevelAt = performance.now() / 1000; },
    refreshColors,
    stats() { return { frames, avgMs: frames ? frameMs / frames : 0, points: NV, edges: NE, running: raf !== 0 }; }
  };
})();
