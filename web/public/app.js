// World Cup 2026 dashboard.
// Data source: Firestore (collection "snapshots") when firebase-config.js is
// filled in, otherwise the bundled /data/*.json. Renders three tabs.

const DOCS = ["tournament", "edges", "ledger", "meta"];
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
    <table><thead><tr><th>Team</th><th>Grp</th><th>Win group</th><th>Reach QF</th>
    <th>Reach SF</th><th>Reach final</th><th>Champion</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderEdges(d) {
  const host = document.getElementById("edges");
  const ops = (d && d.opportunities) || [];
  if (!ops.length) { host.innerHTML = `<div class="empty">No edges clear the filters right now (markets thin pre-tournament, or run the export).</div>`; return; }
  const rows = ops.map((o) => {
    const side = o.side === "YES" ? "BUY" : "FADE";
    const edgeYes = (o.blended ?? o.model_prob) - o.market_prob;
    return `<tr>
      <td>${o.team ?? o.contract}</td>
      <td><span class="tag ${side}">${side}</span></td>
      <td>${o.market_type ?? ""}</td>
      <td>${pct(o.model_prob)}</td>
      <td>${pct(o.market_prob)}</td>
      <td class="${edgeYes >= 0 ? "pos" : "neg"}">${signed(o.edge * (o.side === "YES" ? 1 : -1))}</td>
      <td>${pct(o.entry_price)}</td>
      <td>${money(o.liquidity)}</td>
    </tr>`;
  }).join("");
  host.innerHTML = `<h2>Live edge board <span style="color:var(--muted);font-weight:400">(λ=${d.lambda ?? "—"} shrinkage)</span></h2>
    <table><thead><tr><th>Team</th><th>Side</th><th>Market</th><th>Model</th>
    <th>Market</th><th>Edge</th><th>Entry</th><th>Liquidity</th></tr></thead><tbody>${rows}</tbody></table>`;
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
  host.innerHTML = `<h2>Paper-trading ledger</h2><div class="cards">${cards}</div>
    ${bets.length ? `<table><thead><tr><th>Contract</th><th>Side</th><th>Entry</th>
    <th>Last</th><th>CLV</th><th>Stake</th><th>Status</th></tr></thead><tbody>${rows}</tbody></table>`
    : `<div class="empty">No positions logged yet.</div>`}`;
}

function setupTabs() {
  const btns = document.querySelectorAll("#tabs button");
  btns.forEach((b) => b.addEventListener("click", () => {
    btns.forEach((x) => x.classList.toggle("active", x === b));
    ["tournament", "edges", "paper"].forEach((id) =>
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
  renderEdges(data.edges);
  renderPaper(data.ledger);

  const m = data.meta || {};
  document.getElementById("meta").textContent =
    `${liveMode ? "live (Firestore)" : "static snapshot"} · ${m.generated_at || "—"} · `
    + `${m.n_matches ? m.n_matches.toLocaleString() + " matches · " : ""}${m.n_sims ? m.n_sims.toLocaleString() + " sims" : ""}`;
  status.textContent = m.note || "";
}

main();
