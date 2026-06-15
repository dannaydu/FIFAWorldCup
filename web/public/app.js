// World Cup 2026 dashboard.
// Data source: Firestore (collection "snapshots") when firebase-config.js is
// filled in, otherwise the bundled /data/*.json. Renders three tabs.

const DOCS = ["tournament", "edges", "matches", "results", "scorecard", "history", "ledger", "meta"];
let edgeVenue = "all";   // all | polymarket | kalshi
let EDGES = null;

const intro = (t) => `<p class="tab-intro">${t}</p>`;
const miniBar = (a, d, b) => `<span class="mini"><i style="width:${a * 100}%;background:var(--buy)"></i><i style="width:${d * 100}%;background:var(--muted)"></i><i style="width:${b * 100}%;background:var(--fade)"></i></span>`;

// Minimal dependency-free SVG line chart.
function lineChart(values, { w = 520, h = 120, color = "#38bdf8", baseline = null, ymin, ymax, pctY = false } = {}) {
  const n = values.length;
  if (!n) return "";
  const pad = { l: 40, r: 12, t: 12, b: 16 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  const refs = baseline != null ? [baseline] : [];
  let lo = ymin != null ? ymin : Math.min(...values, ...refs);
  let hi = ymax != null ? ymax : Math.max(...values, ...refs);
  if (lo === hi) { lo -= 0.05; hi += 0.05; }
  const x = (i) => pad.l + (n === 1 ? iw / 2 : (i / (n - 1)) * iw);
  const y = (v) => pad.t + ih - ((v - lo) / (hi - lo)) * ih;
  const pts = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const dots = values.map((v, i) => `<circle cx="${x(i).toFixed(1)}" cy="${y(v).toFixed(1)}" r="2.6" fill="${color}"><title>${pctY ? (v * 100).toFixed(1) + "%" : v.toFixed(3)}</title></circle>`).join("");
  const base = baseline != null
    ? `<line x1="${pad.l}" y1="${y(baseline).toFixed(1)}" x2="${w - pad.r}" y2="${y(baseline).toFixed(1)}" stroke="#fb7185" stroke-dasharray="4 4" stroke-width="1"/>`
    : "";
  const yl = (v) => (pctY ? (v * 100).toFixed(0) + "%" : v.toFixed(2));
  const axis = `<text x="4" y="${pad.t + 9}" fill="#8aa0b8" font-size="10">${yl(hi)}</text>
    <text x="4" y="${pad.t + ih}" fill="#8aa0b8" font-size="10">${yl(lo)}</text>`;
  return `<svg viewBox="0 0 ${w} ${h}" class="chart">${base}<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2"/>${dots}${axis}</svg>`;
}
const pct = (x) => (x == null ? "—" : (x * 100).toFixed(1) + "%");
const signed = (x) => (x == null ? "—" : (x >= 0 ? "+" : "") + (x * 100).toFixed(1) + "%");
const money = (x) => (x == null ? "—" : "$" + Number(x).toLocaleString());
const el = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstChild; };

const cfg = window.FIREBASE_CONFIG || {};
const liveMode = cfg.apiKey && cfg.apiKey !== "REPLACE_ME";

async function loadFromFirestore() {
  const appMod = await import("https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js");
  const fs = await import("https://www.gstatic.com/firebasejs/10.12.0/firebase-firestore.js");
  const app = appMod.initializeApp(cfg);
  const db = fs.getFirestore(app);
  const out = {};
  await Promise.all(DOCS.map(async (name) => {
    const snap = await fs.getDoc(fs.doc(db, "snapshots", name));
    out[name] = snap.exists() ? snap.data() : null;
  }));
  return out;
}

async function loadFromJson() {
  const out = {};
  await Promise.all(DOCS.map(async (name) => {
    try {
      const r = await fetch(`data/${name}.json`, { cache: "no-store" });
      out[name] = r.ok ? await r.json() : null;
    } catch { out[name] = null; }
  }));
  return out;
}

function renderTournament(d) {
  const host = document.getElementById("tournament");
  if (!d || !d.teams) { host.innerHTML = `<div class="empty">No tournament data yet. Run <code>python scripts/export_web.py</code>.</div>`; return; }
  const teams = [...d.teams].sort((a, b) => b.p_champion - a.p_champion);
  const max = Math.max(...teams.map((t) => t.p_champion), 0.01);
  const rows = teams.map((t) => `
    <tr>
      <td>${t.team}</td><td>${t.group ?? ""}</td>
      <td>${pct(t.p_group_winner)}</td>
      <td>${pct(t.p_quarterfinal)}</td>
      <td>${pct(t.p_semifinal)}</td>
      <td>${pct(t.p_final)}</td>
      <td>${pct(t.p_champion)} <span class="bar" style="width:${(t.p_champion / max) * 70}px"></span></td>
    </tr>`).join("");
  host.innerHTML = `<h2>Championship & advancement probabilities</h2>
    ${intro("The model's forecast for the whole tournament — from 50,000 simulated runs of the real 2026 draw. Sorted by chance of winning it all.")}
    <table><thead><tr><th>Team</th><th>Grp</th><th>Win group</th><th>Reach QF</th>
    <th>Reach SF</th><th>Reach final</th><th>Champion</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderScorecard(d, history) {
  const host = document.getElementById("results");
  const s = (d && d.summary) || {};
  const games = (d && d.games) || [];
  if (!games.length) {
    host.innerHTML = `<h2>Results — how the model is doing</h2>` +
      intro("Once group games are played, this scores the model's pre-match predictions against what actually happened.") +
      `<div class="empty">No games scored yet.</div>`;
    return;
  }
  const beatRandom = s.avg_logloss != null && s.avg_logloss < s.random_logloss;
  const cards = [
    ["Games scored", s.n],
    ["Correct picks", `${Math.round(s.accuracy * s.n)} / ${s.n} (${pct(s.accuracy)})`],
    ["Avg Brier", s.avg_brier],
    ["Avg log-loss", `${s.avg_logloss} ${beatRandom ? "✅" : "⚠️"}`],
    ["vs random guess", s.random_logloss],
  ].map(([k, v]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  const rows = games.map((g) => `<tr class="${g.correct ? "row-won" : "row-lost"}">
      <td>${g.team_a} vs ${g.team_b}</td>
      <td>${miniBar(g.p_a, g.p_draw, g.p_b)}</td>
      <td>${g.model_pick_label} <span class="muted">(${pct(g.model_pick_prob)})</span></td>
      <td><b>${g.result_label}</b></td>
      <td>${g.correct ? "✓" : "✗"}</td>
    </tr>`).join("");

  // Accuracy / log-loss over time, from the rolling history snapshots.
  const scored = (history || []).filter((e) => e.scorecard && e.scorecard.n > 0);
  let charts = "";
  if (scored.length >= 2) {
    const acc = scored.map((e) => e.scorecard.accuracy);
    const ll = scored.map((e) => e.scorecard.avg_logloss);
    const rnd = scored[scored.length - 1].scorecard.random_logloss;
    charts = `<div class="chart-row">
      <div class="chart-box"><div class="chart-title">Accuracy over time</div>${lineChart(acc, { color: "#34d399", ymin: 0, ymax: 1, pctY: true })}</div>
      <div class="chart-box"><div class="chart-title">Log-loss vs random (dashed = ${rnd})</div>${lineChart(ll, { color: "#38bdf8", baseline: rnd })}</div>
    </div>`;
  } else if (scored.length === 1) {
    charts = intro("📈 The accuracy-over-time chart fills in after a few refreshes (the cron runs every 3h).");
  }

  host.innerHTML = `<h2>Results — how the model is doing</h2>
    ${intro(`Scoring the model's <b>pre-match</b> predictions (locked before kickoff) against actual results. Lower log-loss is better; below ${s.random_logloss} means it's beating a random guess.`)}
    <div class="cards">${cards}</div>
    ${charts}
    <table><thead><tr><th>Game</th><th>Model (win/draw/win)</th><th>Model pick</th><th>Actual</th><th></th></tr></thead><tbody>${rows}</tbody></table>`;
}

const VENUE = { polymarket: "🟣 Polymarket", kalshi: "🔵 Kalshi" };

function renderEdges(d) {
  EDGES = d || EDGES;
  const host = document.getElementById("edges");
  const all = (EDGES && EDGES.opportunities) || [];
  const count = (v) => all.filter((o) => v === "all" || o.platform === v).length;
  const pills = [["all", `All (${all.length})`], ["polymarket", `${VENUE.polymarket} (${count("polymarket")})`],
    ["kalshi", `${VENUE.kalshi} (${count("kalshi")})`]]
    .map(([v, l]) => `<button class="venue-pill ${edgeVenue === v ? "active" : ""}" data-venue="${v}">${l}</button>`).join("");

  const ops = all.filter((o) => edgeVenue === "all" || o.platform === edgeVenue);
  const header = `<h2>Market edges — where the model disagrees with the betting market</h2>
    ${intro("Contracts where the model's probability differs from the live Kalshi/Polymarket price by enough to matter (after fees/spread). <b>BUY</b> = model thinks it's underpriced; <b>FADE</b> = overpriced. Research only, not betting advice.")}
    <div class="venue-filter">${pills}</div>`;

  if (!ops.length) {
    host.innerHTML = header + `<div class="empty">No edges on ${edgeVenue === "all" ? "either venue" : VENUE[edgeVenue]} right now (markets are thin pre-tournament).</div>`;
    return;
  }
  const rows = ops.map((o) => {
    const side = o.side === "YES" ? "BUY" : "FADE";
    const market = o.market_type === "match"
      ? (o.contract || "").replace(/\s*Winner\??$/i, "")
      : (o.market_type ?? "");
    return `<tr>
      <td><span class="venue ${o.platform}">${o.platform === "kalshi" ? "🔵" : "🟣"}</span> ${o.team ?? o.contract}</td>
      <td><span class="tag ${side}">${side}</span></td>
      <td>${market}</td>
      <td>${pct(o.model_prob)}</td>
      <td>${pct(o.market_prob)}</td>
      <td class="${o.side === "YES" ? "pos" : "neg"}">${signed(o.edge * (o.side === "YES" ? 1 : -1))}</td>
      <td>${pct(o.entry_price)}</td>
      <td>${money(o.liquidity)}</td>
    </tr>`;
  }).join("");
  host.innerHTML = header +
    `<table><thead><tr><th>Contract</th><th>Side</th><th>Market</th><th>Model</th>
    <th>Market</th><th>Edge</th><th>Entry</th><th>Liquidity</th></tr></thead><tbody>${rows}</tbody></table>`;

  host.querySelectorAll("[data-venue]").forEach((b) =>
    b.addEventListener("click", () => { edgeVenue = b.dataset.venue; renderEdges(); }));
}

function renderPaper(d) {
  const host = document.getElementById("paper");
  if (!d || !d.summary) { host.innerHTML = `<div class="empty">No paper-trading ledger yet. Run <code>python scripts/paper_trade.py</code>.</div>`; return; }
  const s = d.summary;
  const cards = [
    ["Open positions", s.open ?? 0],
    ["Settled", s.settled ?? 0],
    ["Bankroll", money(s.bankroll)],
    ["Avg open CLV", s.avg_open_clv == null ? "—" : signed(s.avg_open_clv)],
    ["Settled P/L", s.settled_pnl == null ? "—" : money(s.settled_pnl)],
    ["ROI", s.roi == null ? "—" : (s.roi * 100).toFixed(1) + "%"],
    ["Hit rate", s.hit_rate == null ? "—" : pct(s.hit_rate)],
  ].map(([k, v]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");

  const bets = [...(d.bets || [])].sort((a, b) => (b.edge || 0) - (a.edge || 0));
  const rows = bets.map((b) => `
    <tr>
      <td>${b.contract}</td>
      <td><span class="tag ${b.side}">${b.side}</span></td>
      <td>${pct(b.entry_price)}</td>
      <td>${pct(b.last_price)}</td>
      <td class="${(b.clv || 0) >= 0 ? "pos" : "neg"}">${signed(b.clv)}</td>
      <td>${money(b.stake)}</td>
      <td>${b.status}${b.pnl != null ? " (" + money(b.pnl) + ")" : ""}</td>
    </tr>`).join("");
  host.innerHTML = `<h2>Model bets — automated paper-trading ledger</h2>
    ${intro("The model bets its own (fake) bankroll on the edges above, sized by fractional-Kelly. <b>CLV</b> (closing-line value) tracks whether the price moved the model's way — the cleanest sign it's finding real value. No real money.")}
    <div class="cards">${cards}</div>
    ${bets.length ? `<table><thead><tr><th>Contract</th><th>Side</th><th>Entry</th>
    <th>Last</th><th>CLV</th><th>Stake</th><th>Status</th></tr></thead><tbody>${rows}</tbody></table>`
    : `<div class="empty">No positions logged yet.</div>`}`;
}

function setupTabs() {
  const btns = document.querySelectorAll("#tabs button");
  btns.forEach((b) => b.addEventListener("click", () => {
    btns.forEach((x) => x.classList.toggle("active", x === b));
    ["play", "tournament", "results", "edges", "paper"].forEach((id) =>
      document.getElementById(id).classList.toggle("hidden", id !== b.dataset.tab));
  }));
}

async function main() {
  setupTabs();
  const status = document.getElementById("status");
  status.textContent = liveMode ? "Loading live data from Firestore…" : "Loading bundled snapshot…";
  let data;
  try {
    data = liveMode ? await loadFromFirestore() : await loadFromJson();
  } catch (e) {
    status.textContent = "Firestore read failed, falling back to bundled JSON.";
    data = await loadFromJson();
  }
  renderTournament(data.tournament);
  renderScorecard(data.scorecard, data.history);
  renderEdges(data.edges);
  renderPaper(data.ledger);
  if (window.FANTASY) window.FANTASY.init(data);

  const m = data.meta || {};
  document.getElementById("meta").textContent =
    `${liveMode ? "live (Firestore)" : "static snapshot"} · ${m.generated_at || "—"} · `
    + `${m.n_matches ? m.n_matches.toLocaleString() + " matches · " : ""}${m.n_sims ? m.n_sims.toLocaleString() + " sims" : ""}`;
  status.textContent = m.note || "";
}

main();
