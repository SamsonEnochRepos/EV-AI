"use strict";
// The orb on the Live page: a swirl of particles inside a few HUD rings. It is only a picture of EV's state and voice level;
// it draws on a canvas and never touches page markup. Colours come from the theme's CSS variables.

const EVOrb = (() => {
  const N = 2400;
  const TILT = 1.0;
  const BUCKETS = 8;
  const PARAMS = {            // spin (rad/s), swirl (how tightly the arms wind), size (of the swirl), glow
    idle: { spin: 0.12, swirl: 4.4, size: 1.02, glow: 0.10 },
    listening: { spin: 0.25, swirl: 4.8, size: 1.08, glow: 0.22 },
    transcribing: { spin: 0.55, swirl: 5.6, size: 1.04, glow: 0.18 },
    thinking: { spin: 0.95, swirl: 6.6, size: 1.02, glow: 0.2 },
    acting: { spin: 1.25, swirl: 7.4, size: 0.9, glow: 0.34 },
    speaking: { spin: 0.35, swirl: 4.2, size: 1.08, glow: 0.26 },
    paused: { spin: 0.03, swirl: 4.0, size: 0.85, glow: 0.04 },
    error: { spin: 0.05, swirl: 4.0, size: 0.9, glow: 0.3 },
  };
  const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // fixed pseudo-random swirl, so it looks the same every time the page loads
  let seed = 20260;
  const rnd = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
  const gauss = () => (rnd() + rnd() + rnd() + rnd() - 2) / 2;
  const pts = Array.from({ length: N }, (_, i) => {
    const halo = rnd() < 0.12;                                  // a soft disc of dust behind the arms
    const t = Math.pow(rnd(), halo ? 0.7 : 0.8);
    return { t, arm: (i % 3) * (Math.PI * 2 / 3), jitter: halo ? (rnd() - 0.5) * 6.283 : gauss() * (1.1 - 0.7 * t), z: gauss() * 0.2 * (1 - 0.6 * t),
             size: 0.5 + rnd() * 1.2, bucket: Math.min(BUCKETS - 1, Math.floor(t * BUCKETS)), phase: rnd() * 6.283 };
  });

  const groups = Array.from({ length: BUCKETS }, () => []);      // particles by colour bucket: one fillStyle per bucket per frame
  for (const q of pts) groups[q.bucket].push(q);

  let canvas = null, ctx = null, raf = 0, last = 0;
  let state = "idle", level = 0, smoothLevel = 0, angle = 0, swirl = 4.4, size = 1.0, glow = 0.1;
  let cols = { a: [255, 45, 63], b: [255, 177, 153], accent: [255, 45, 63] };

  const parse = (hex) => {
    const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
    if (!m) return null;
    const n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  };
  function refreshColors() {
    const cs = getComputedStyle(document.documentElement);
    cols = { a: parse(cs.getPropertyValue("--orb-a")) || cols.a, b: parse(cs.getPropertyValue("--orb-b")) || cols.b,
             accent: parse(cs.getPropertyValue("--accent")) || cols.accent };
  }
  const mix = (c1, c2, t) => c1.map((v, i) => Math.round(v + (c2[i] - v) * t));
  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  const ease = (cur, tgt, dt, rate) => tgt + (cur - tgt) * Math.exp(-rate * dt);

  function resize() {
    if (!canvas) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(160, canvas.clientWidth), h = Math.max(160, canvas.clientHeight);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
  }

  function rings(cx, cy, R, now) {
    ctx.lineWidth = 1;
    ctx.strokeStyle = rgba(cols.a, 0.35);
    ctx.setLineDash([2, 7]);
    ctx.lineDashOffset = -now * 6;
    ctx.beginPath(); ctx.arc(cx, cy, R * 1.12, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([22, 14, 4, 14]);
    ctx.lineDashOffset = now * 14;
    ctx.strokeStyle = rgba(cols.b, 0.22);
    ctx.beginPath(); ctx.arc(cx, cy, R * 1.24, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = rgba(cols.a, 0.16);
    ctx.beginPath(); ctx.arc(cx, cy, R * 1.38, 0, Math.PI * 2); ctx.stroke();
    // tick marks that creep around the outer ring
    ctx.strokeStyle = rgba(cols.accent, 0.5);
    ctx.lineWidth = 1.5;
    const ticks = 72, off = now * 0.05;
    for (let i = 0; i < ticks; i += 1) {
      const a = (i / ticks) * Math.PI * 2 + off;
      const long = i % 6 === 0;
      const r1 = R * 1.38, r2 = R * (long ? 1.47 : 1.43);
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
      ctx.lineTo(cx + Math.cos(a) * r2, cy + Math.sin(a) * r2);
      ctx.stroke();
    }
    ctx.lineWidth = 1;
  }

  function frame(ts) {
    raf = requestAnimationFrame(frame);
    if (!canvas || !ctx) return;
    const now = ts / 1000;
    const dt = Math.min(0.1, last ? now - last : 0.016);
    last = now;
    resize();
    const W = canvas.width, H = canvas.height;
    const R = Math.min(W, H) * 0.32;
    const cx = W / 2, cy = H / 2;
    const p = PARAMS[state] || PARAMS.idle;
    smoothLevel = ease(smoothLevel, level, dt, level > smoothLevel ? 28 : 9);
    angle += dt * p.spin * (reduced ? 0.3 : 1) * (1 + smoothLevel * (state === "listening" ? 1.2 : 0.4));
    swirl = ease(swirl, p.swirl, dt, 2.5);
    size = ease(size, p.size, dt, 3);
    glow = ease(glow, p.glow + smoothLevel * 0.18, dt, 6);
    const dpr = W / Math.max(1, canvas.clientWidth);
    ctx.clearRect(0, 0, W, H);
    ctx.globalCompositeOperation = "lighter";
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 1.25);
    g.addColorStop(0, rgba(cols.a, glow));
    g.addColorStop(1, rgba(cols.a, 0));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
    rings(cx, cy, R, now);
    const breathe = 1 + 0.025 * Math.sin(now * 1.3) + smoothLevel * (state === "speaking" ? 0.1 : state === "listening" ? 0.14 : 0);
    const cosT = Math.cos(TILT), sinT = Math.sin(TILT);
    const waves = state === "speaking" || state === "listening";
    const paused = state === "paused";
    for (let b = 0; b < BUCKETS; b += 1) {
      const col = mix(cols.a, cols.b, b / (BUCKETS - 1));
      ctx.fillStyle = rgba(state === "acting" ? mix(col, cols.accent, 0.55) : col, 1);
      for (const q of groups[b]) {
        const wave = waves ? 1 + 0.07 * smoothLevel * Math.sin(q.t * 16 - now * 7 + q.phase) : 1;
        const r = q.t * size * breathe * wave;
        const th = q.arm + q.t * swirl + q.jitter + angle;
        const x = Math.cos(th) * r, y0 = Math.sin(th) * r;
        const z0 = q.z * (1 + smoothLevel);
        const y = y0 * cosT - z0 * sinT, z = y0 * sinT + z0 * cosT;
        const s = 1 / (1 - z * 0.35);
        const tw = 0.7 + 0.3 * Math.sin(now * 2 + q.phase);
        ctx.globalAlpha = Math.min(1, (0.28 + 0.72 * (0.5 + z * 0.9)) * tw * (paused ? 0.35 : 1));
        const d = q.size * s * dpr * (1 + smoothLevel * 0.5);
        ctx.fillRect(cx + x * R * s - d, cy + y * R * s * 0.92 - d, d * 2, d * 2);
      }
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
  }

  return {
    mount(el) {
      canvas = el;
      ctx = el.getContext("2d");
      refreshColors();
      last = 0;
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(frame);
    },
    unmount() { cancelAnimationFrame(raf); raf = 0; canvas = null; ctx = null; },
    setState(s) { state = s; },
    setLevel(v) { level = Math.max(0, Math.min(1, Number(v) || 0)); },
    refreshColors,
  };
})();
