"use strict";
// Every piece of text from EV, the model, the web or your notes goes into the page as text (textContent), never as markup.

const $ = (id) => document.getElementById(id);
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, "");
    else if (v !== false && v != null) el.setAttribute(k, String(v));
  }
  for (const kid of kids.flat()) if (kid != null) el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  return el;
}
const clear = (el) => el.replaceChildren();

function say(msg, kind) {
  const box = h("div", { class: kind || "info" }, msg);
  $("notice").replaceChildren(box);
  setTimeout(() => box.isConnected && box.remove(), kind === "error" ? 9000 : 4500);
}
async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  let data = {};
  try { data = await r.json(); } catch { /* not JSON */ }
  if (!r.ok) throw new Error(data.error || `request failed (${r.status})`);
  return data;
}
const guard = (fn) => async (...a) => { try { await fn(...a); } catch (e) { say(e.message, "error"); } };

// ---- theme (a per-browser preference; the page works without storage)
const THEMES = ["crimson", "arc", "nebula"];
function applyTheme(name) {
  const t = THEMES.includes(name) ? name : "crimson";
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("ev-theme", t); } catch { /* private window or blocked storage */ }
  EVOrb.refreshColors();
}
let savedTheme = "crimson";
try { savedTheme = localStorage.getItem("ev-theme") || "crimson"; } catch { /* storage blocked */ }
applyTheme(savedTheme);
document.querySelectorAll("[data-theme-set]").forEach((b) => b.addEventListener("click", () => applyTheme(b.dataset.themeSet)));

// the brand's two eyes follow the pointer, like the island on the desktop
const eyes = $("eyes");
window.addEventListener("pointermove", (e) => {
  const r = eyes.getBoundingClientRect();
  const dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2);
  eyes.style.setProperty("--ex", `${(Math.tanh(dx / 300) * 5).toFixed(2)}px`);
  eyes.style.setProperty("--ey", `${(Math.tanh(dy / 200) * 3).toFixed(2)}px`);
});

// ---- tabs
const loaders = { live: loadLive, settings: loadSettings, voice: loadVoice, skills: loadSkills, memory: loadMemory, policy: loadPolicy };
const TITLES = { live: "Live", settings: "Settings", voice: "Voice & model", skills: "Skills", memory: "Memory", policy: "Policy" };
document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("on", x === b));
  for (const name of Object.keys(loaders)) $("tab-" + name).hidden = name !== b.dataset.tab;
  $("page-title").textContent = TITLES[b.dataset.tab];
  if (b.dataset.tab !== "live") EVOrb.unmount();
  guard(loaders[b.dataset.tab])();
}));
document.querySelectorAll("[data-control]").forEach((b) => b.addEventListener("click", guard(async () => {
  const a = b.dataset.control;
  const act = a === "pause" ? (status.paused ? "resume" : "pause") : a === "mute" ? (status.muted ? "unmute" : "mute") : a;
  await api("/api/control", { action: act });
  await refreshStatus();
})));

// ---- live
let status = {};
const tools = [], events = [], feed = [], seen = new Set();
const ORB_LABEL = { idle: "Standby", listening: "Listening", transcribing: "Transcribing", thinking: "Thinking", acting: "Acting",
                    speaking: "Speaking", paused: "Paused", muted: "Paused", error: "Fault", offline: "Offline" };
const clock = (at) => new Date((at || Date.now() / 1000) * 1000).toLocaleTimeString([], { hour12: false });

function setState(s) {
  status.state = s;
  document.body.dataset.state = s === "muted" ? "paused" : s;
  $("state").textContent = ORB_LABEL[s] || s;
  EVOrb.setState(s === "muted" ? "paused" : s === "offline" ? "error" : s);
  const label = $("orb-state");
  if (label) label.textContent = ORB_LABEL[s] || s;
}
async function refreshStatus() {
  status = await api("/api/status");
  $("model").textContent = status.model + (status.voice.engine ? ` · ${status.voice.engine} ${status.voice.voice || ""}` : "");
  $("b-pause").textContent = status.paused ? "Resume" : "Pause";
  $("b-mute").textContent = status.muted ? "Unmute" : "Mute";
  setState(status.state);
  renderSide();
  renderTiles();
}
function loadLive() {
  const el = $("tab-live");
  clear(el);
  el.append(h("div", { class: "hud" },
    h("div", { class: "col left" },
      h("div", { class: "panel" }, h("h2", {}, "Skills"), h("div", { class: "skill-list", id: "skilllist" })),
      h("div", { class: "panel", id: "side" })),
    h("div", { class: "col" },
      h("div", { class: "panel orb-panel" },
        h("div", { class: "orb-wrap" }, h("canvas", { id: "orb", "aria-label": "EV's state" }),
          h("div", { class: "orb-core" }, h("div", { class: "orb-state", id: "orb-state" }), h("div", { class: "orb-sub", id: "orb-sub" }))),
        h("div", { class: "processing" }, h("span", { class: "label" }, "Processing"), h("div", { class: "meter" }, h("i"))),
        h("div", { class: "tiles", id: "tiles" })),
      h("div", { class: "panel" }, h("h2", {}, "What EV did"), h("table", {}, h("tbody", { id: "toolrows" })))),
    h("div", { class: "col" },
      h("div", { class: "panel" }, h("h2", {}, "Live feed"), h("div", { class: "feed", id: "feed" })),
      h("div", { class: "panel" }, h("h2", {}, "Spoke up on its own"), h("div", { id: "proactive" })))));
  EVOrb.mount($("orb"));
  renderFeed();
  renderTools();
  renderEvents();
  guard(loadSkillList)();
  return refreshStatus();
}
async function loadSkillList() {
  const d = await api("/api/skills");
  const box = $("skilllist");
  if (!box) return;
  clear(box);
  for (const s of d.skills) {
    box.append(h("div", { class: "skill-item" }, h("span", { class: "name" }, s.name, s.description ? h("span", { class: "desc" }, s.description) : null),
      h("button", { onclick: guard(async () => { say(`Running ${s.name}…`); const r = await api("/api/skills/run", { name: s.name }); say(r.message, r.ok ? "info" : "error"); }) }, "Run")));
  }
  if (!d.skills.length) box.append(h("p", { class: "empty" }, "No skills yet. Say “save that as …” after doing something."));
}
function pushFeed(kind, tag, text, at, key) {
  const id = key || `${kind}:${at}:${text}`;
  if (seen.has(id)) return;                       // a page that reconnects is replayed recent events; skip the ones it already has
  seen.add(id);
  feed.push({ kind, tag, text, at });
  if (feed.length > 80) feed.shift();
  renderFeed();
}
function renderFeed() {
  const box = $("feed");
  if (!box) return;
  clear(box);
  for (const f of feed.slice().reverse()) {
    box.append(h("div", { class: `feed-item ${f.kind}` }, h("time", {}, clock(f.at)), h("div", {}, h("span", { class: "tag" }, f.tag), h("div", { class: "body" }, f.text))));
  }
  if (!feed.length) box.append(h("p", { class: "empty" }, "Nothing yet. Say “Hey Evie”."));
}
function renderTools() {
  const body = $("toolrows");
  if (!body) return;
  clear(body);
  for (const t of tools.slice(-12).reverse()) {
    body.append(h("tr", {}, h("td", {}, t.tool), h("td", { class: "muted" }, Object.values(t.args || {}).join(", ").slice(0, 60)),
      h("td", { class: t.decision }, t.decision), h("td", { class: "muted" }, `${t.ms} ms`)));
  }
  if (!tools.length) body.append(h("tr", {}, h("td", { class: "muted" }, "No tool calls yet.")));
  renderTiles();
}
function renderEvents() {
  const box = $("proactive");
  if (!box) return;
  clear(box);
  for (const e of events.slice(-8).reverse()) box.append(h("p", {}, `${e.rule || e.name || "rule"}: `, h("span", { class: "muted" }, e.text || e.action || "")));
  if (!events.length) box.append(h("p", { class: "empty" }, "Nothing."));
}
function renderTiles() {
  const box = $("tiles");
  if (!box) return;
  clear(box);
  const tile = (label, val, extra) => h("div", { class: "tile" }, h("div", { class: "label" }, label), h("div", { class: "val" }, val), extra);
  box.append(
    tile("Model", status.model || "…"),
    tile("Voice", status.voice && status.voice.engine ? `${status.voice.engine} · ${status.voice.voice || ""}` : "…"),
    tile("Tools run", String(tools.length)),
    tile("Display mode", status.display_mode ? "On" : "Off", h("button", { onclick: guard(async () => {
      await api("/api/settings", { changes: { "overlay.display_mode": !status.display_mode } });
      await refreshStatus();
    }) }, status.display_mode ? "Turn off" : "Turn on")));
}
function renderSide() {
  const el = $("side");
  if (!el) return;
  clear(el);
  el.append(h("h2", {}, "Timers and reminders"));
  for (const t of status.timers || []) el.append(h("p", {}, `${t.label || "timer"}: ${Math.floor(t.seconds_left / 60)}m ${t.seconds_left % 60}s left`));
  for (const r of status.reminders || []) el.append(h("p", {}, `${r.text} `, h("span", { class: "muted" }, new Date(r.due * 1000).toLocaleString())));
  if (!(status.timers || []).length && !(status.reminders || []).length) el.append(h("p", { class: "empty" }, "None."));
  if (status.proactive) {
    el.append(h("h2", {}, "Speaking up"));
    el.append(h("p", { class: "muted" }, status.proactive.snoozed_min ? `Quiet for ${status.proactive.snoozed_min} more minutes.` : `${status.proactive.rules} rules active.`));
    const mins = h("select", {}, [["30", "30 min"], ["60", "1 hour"], ["180", "3 hours"], ["0", "Resume"]].map(([v, t]) => h("option", { value: v }, t)));
    el.append(h("div", { class: "row" }, mins, h("button", { onclick: guard(async () => { await api("/api/snooze", { minutes: Number(mins.value) }); await refreshStatus(); }) }, "Quiet")));
    for (const p of status.proactive.problems) el.append(h("p", { class: "muted" }, p));
  }
  if (status.vault && status.vault.unconfirmed) el.append(h("p", { class: "confirm" }, `${status.vault.unconfirmed} unconfirmed fact(s): see Memory.`));
}
function connect() {
  const es = new EventSource("/api/events");
  const on = (name, fn) => es.addEventListener(name, (m) => { try { fn(JSON.parse(m.data)); } catch { /* ignore a malformed event */ } });
  es.onopen = () => { document.body.dataset.conn = "on"; $("online").textContent = "ONLINE"; };
  on("state", (d) => setState(d.new));
  on("level", (d) => EVOrb.setLevel(d.v));
  on("turn", (d) => {
    const sub = $("orb-sub");
    if (d.user) { pushFeed("heard", "You", d.user, d.at, `heard:${d.at}`); if (sub) sub.textContent = d.user; }
    if (d.reply) pushFeed("reply", d.interrupted ? "EV (interrupted)" : "EV", d.reply, d.at, `reply:${d.at}`);
  });
  on("tool", (d) => {
    const key = `tool:${d.at}`;
    if (seen.has(key)) return;
    seen.add(key);
    tools.push(d);
    renderTools();
    pushFeed("tool " + (d.decision || ""), `Tool · ${d.decision || ""}`, `${d.tool}${d.args ? " " + Object.values(d.args).join(", ").slice(0, 80) : ""}`, d.at, key + ":f");
  });
  on("proactive", (d) => {
    const key = `pro:${d.at}`;
    if (seen.has(key)) return;
    seen.add(key);
    events.push(d);
    renderEvents();
    pushFeed("proactive", "Spoke up", d.text || d.what || d.rule || "", d.at, key + ":f");
  });
  es.onerror = () => { document.body.dataset.conn = "off"; $("online").textContent = "OFFLINE"; setState("offline"); };
}

// ---- settings
async function loadSettings() {
  const d = await api("/api/settings");
  const el = $("tab-settings");
  clear(el);
  const inputs = {};
  const card = h("div", { class: "card" });
  for (const s of d.settings) {
    let input;
    if (s.kind === "bool") input = h("input", { type: "checkbox", checked: !!s.value });
    else if (s.kind === "time") input = h("input", { type: "time", value: s.value || "" });
    else input = h("input", { type: "number", value: s.value ?? "", min: s.lo, max: s.hi, step: s.kind === "int" ? 1 : "any" });
    inputs[s.key] = { s, input };
    card.append(h("div", { class: "row" }, h("label", {}, s.label), input, h("span", { class: "muted" }, s.live ? "" : "restart needed"), s.help ? h("span", { class: "muted" }, s.help) : null));
  }
  const save = h("button", { class: "primary", onclick: guard(async () => {
    const changes = {};
    for (const [k, { s, input }] of Object.entries(inputs)) {
      const v = s.kind === "bool" ? input.checked : s.kind === "time" ? input.value : input.value === "" ? null : Number(input.value);
      if (v !== null && v !== (s.value ?? null) && v !== "") changes[k] = v;
    }
    if (!Object.keys(changes).length) return say("Nothing changed.");
    const r = await api("/api/settings", { changes });
    say(`Saved ${r.saved.length}.` + (r.restart_needed.length ? ` Restart EV for: ${r.restart_needed.join(", ")}.` : " Applied now."));
    await loadSettings();
  }) }, "Save");
  el.append(card, h("div", { class: "row" }, save), h("p", { class: "muted" }, d.file_only));
}

// ---- voice and model
async function loadVoice() {
  const [v, m] = await Promise.all([api("/api/voices"), api("/api/models")]);
  const el = $("tab-voice");
  clear(el);
  const engine = h("select", {}, ["piper", "kokoro"].map((e) => h("option", { value: e, selected: e === v.engine }, e)));
  const voice = h("select", {});
  const speed = h("input", { type: "number", step: "0.05", min: v.speed_range[0], max: v.speed_range[1], value: v.engine === "kokoro" ? v.kokoro_speed : (1 / v.length_scale).toFixed(2) });
  const fill = () => {
    clear(voice);
    const list = v[engine.value], cur = engine.value === "piper" ? v.piper_voice : v.kokoro_voice;
    for (const n of list) voice.append(h("option", { value: n, selected: n === cur }, n));
  };
  engine.addEventListener("change", fill); fill();
  const send = (save) => guard(async () => {
    const r = await api("/api/voice", { engine: engine.value, voice: voice.value, speed: Number(speed.value), preview: true, save });
    say(`Now using ${r.voice}` + (save ? " (saved)." : " until EV restarts."));
  });
  el.append(h("div", { class: "card" }, h("h2", {}, "Voice"),
    h("div", { class: "row" }, h("label", {}, "Engine"), engine),
    h("div", { class: "row" }, h("label", {}, "Voice"), voice),
    h("div", { class: "row" }, h("label", {}, "Speed"), speed),
    h("div", { class: "row" }, h("button", { onclick: send(false) }, "Try it"), h("button", { class: "primary", onclick: send(true) }, "Use and save"))));
  const rows = m.models.map((x) => h("tr", {}, h("td", {}, x.name, x.name === m.current ? " (current)" : ""), h("td", {}, `${x.gb} GB`),
    h("td", {}, x.too_big ? h("span", { class: "confirm" }, `over ${m.vram_budget_gb} GB: may not fit the GPU`) : ""),
    h("td", {}, h("button", { onclick: guard(async () => { await api("/api/model", { model: x.name, save: false }); say(`Using ${x.name} until restart.`); await refreshStatus(); loadVoice(); }) }, "Use"),
      " ", h("button", { onclick: guard(async () => { await api("/api/model", { model: x.name, save: true }); say(`Saved ${x.name}.`); await refreshStatus(); loadVoice(); }) }, "Use and save"))));
  el.append(h("div", { class: "card" }, h("h2", {}, "Chat model"), h("table", {}, h("tbody", {}, rows)),
    h("p", { class: "muted" }, "Switching reloads the model; the first reply afterwards may be slow. Persona and tool calling were tuned on qwen2.5:3b.")));
}

// ---- skills
async function loadSkills() {
  const d = await api("/api/skills");
  const el = $("tab-skills");
  clear(el);
  for (const p of d.problems) el.append(h("div", { class: "card confirm" }, p));
  for (const s of d.skills) {
    el.append(h("div", { class: "card" }, h("h2", {}, s.name), h("p", { class: "muted" }, s.description),
      h("p", {}, "Say: ", s.triggers.join(" · ")),
      h("ol", {}, s.steps.map((x) => h("li", {}, `${x.tool} ${JSON.stringify(x.args)}${x.optional ? " (optional)" : ""}`))),
      h("div", { class: "row" },
        h("button", { onclick: guard(async () => { say("Running…"); const r = await api("/api/skills/run", { name: s.name }); say(r.message, r.ok ? "info" : "error"); }) }, "Run"),
        h("button", { onclick: () => edit(s, d.tools) }, "Edit"),
        h("button", { class: "danger", onclick: guard(async () => {
          say("EV will ask you out loud to confirm.");
          const r = await api("/api/skills/delete", { name: s.name });
          say(r.deleted ? "Deleted." : (r.message || "Not deleted."), r.deleted ? "info" : "error"); loadSkills();
        }) }, "Delete"))));
  }
  el.append(h("div", { class: "row" }, h("button", { onclick: () => edit({ name: "", description: "", triggers: [], say: "", params: {}, steps: [{ tool: "", args: {} }] }, d.tools) }, "New skill")),
    h("div", { id: "editor" }));
}
function edit(s, tools) {
  const box = $("editor");
  clear(box);
  const isNew = !s.name;
  const text = h("textarea", {}, JSON.stringify({ name: s.name, description: s.description, triggers: s.triggers, say: s.say, params: s.params, steps: s.steps }, null, 2));
  box.append(h("div", { class: "card" }, h("h2", {}, isNew ? "New skill" : `Edit ${s.name}`),
    h("p", { class: "muted" }, "Steps are ordinary tool calls and each one is checked by the safety gate when it runs. Tools: " + tools.join(", ")),
    text, h("div", { class: "row" }, h("button", { class: "primary", onclick: guard(async () => {
      let spec;
      try { spec = JSON.parse(text.value); } catch { throw new Error("That isn't valid JSON."); }
      if (!isNew && spec.name !== s.name) throw new Error("To rename a skill, say “rename skill …” to EV.");
      await api("/api/skills/save", spec); say("Saved."); loadSkills();
    }) }, "Save"), h("button", { onclick: () => clear(box) }, "Cancel"))));
  box.scrollIntoView();
}

// ---- memory
async function loadMemory() {
  const d = await api("/api/facts");
  const el = $("tab-memory");
  clear(el);
  if (!d.vault) { el.append(h("div", { class: "card" }, "The vault is off.")); return; }
  el.append(h("div", { class: "card" }, h("h2", {}, "What EV knows (verified)"), h("table", {}, h("tbody", {},
    Object.entries(d.facts).map(([k, v]) => h("tr", {}, h("td", {}, k), h("td", {}, v),
      h("td", {}, h("button", { class: "danger", onclick: guard(async () => {
        say("EV will ask you out loud to confirm.");
        const r = await api("/api/forget", { key: k }); say(r.message || (r.ok ? "Done." : "Not done."), r.ok ? "info" : "error"); loadMemory();
      }) }, "Forget"))))))));
  if (d.unconfirmed.length) {
    el.append(h("div", { class: "card" }, h("h2", {}, "Unconfirmed lines in Facts.md"),
      h("p", { class: "muted" }, "EV did not write these, so it doesn't use them. Pressing Confirm makes EV read one back and wait for your spoken “yes”."),
      h("table", {}, h("tbody", {}, d.unconfirmed.map((u) => h("tr", {}, h("td", {}, u.key), h("td", {}, u.value),
        h("td", {}, h("button", { onclick: guard(async () => { const r = await api("/api/keep", { id: u.id }); say(r.message || "Done.", r.ok ? "info" : "error"); loadMemory(); }) }, "Confirm by voice"))))))));
  }
  const q = h("input", { type: "search", placeholder: "Search the vault" }), when = h("input", { placeholder: "when (yesterday, last week…)" });
  const out = h("div", {});
  const go = guard(async () => {
    const r = await api(`/api/search?q=${encodeURIComponent(q.value)}&when=${encodeURIComponent(when.value)}`);
    clear(out);
    for (const x of r.hits) out.append(h("div", { class: "card" }, h("div", { class: "muted" }, `${x.path} · ${x.date}${x.heading ? " · " + x.heading : ""}`), h("pre", {}, x.text)));
    if (!r.hits.length) out.append(h("p", { class: "muted" }, "No matches."));
  });
  q.addEventListener("keydown", (e) => e.key === "Enter" && go());
  el.append(h("div", { class: "card" }, h("h2", {}, "Search"), h("div", { class: "row" }, q, when, h("button", { onclick: go }, "Search")), out));
}

// ---- policy (read-only)
async function loadPolicy() {
  const d = await api("/api/policy");
  const el = $("tab-policy");
  clear(el);
  el.append(h("div", { class: "card" }, h("h2", {}, d.path + " (read-only)"),
    h("p", { class: "muted" }, "What EV may do is decided by this file alone. The dashboard can't change it; edit it by hand."), h("pre", {}, d.text)));
}

guard(loadLive)();
connect();
setInterval(() => guard(refreshStatus)(), 5000);
