// World Cup 2026 dashboard.
// Data loads from the deployed /data/*.json snapshots (refreshed by the cron).
// Optional Google sign-in for fantasy cards is handled separately in auth.js.

const DOCS = ["tournament", "edges", "matches", "results", "scorecard", "history", "ledger", "meta"];
let edgeVenue = "all";   // all | polymarket | kalshi
let EDGES = null;

const intro = (t) => `<p class="tab-intro">${t}</p>`;
const sectionHead = (kicker, title, copy, meta = "") => `<div class="section-head">
  <div><div class="eyebrow">${kicker}</div><h2>${title}</h2><p>${copy}</p></div>
  ${meta ? `<div class="section-meta">${meta}</div>` : ""}
</div>`;
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

// Data always loads from the deployed JSON snapshots (the cron refreshes them).
// Firebase config (if present) is only used by auth.js for optional sign-in.
async function loadFromJson() {
  const out = { __errors: {} };
  await Promise.all(DOCS.map(async (name) => {
    try {
      const r = await fetch(`data/${name}.json`, { cache: "no-store" });
      out[name] = r.ok ? await r.json() : null;
      if (!r.ok) out.__errors[name] = `HTTP ${r.status}`;
    } catch (e) {
      out[name] = null;
      out.__errors[name] = e.message || "invalid snapshot";
    }
  }));
  return out;
}

function renderTournament(d) {
  const host = document.getElementById("tournament");
  if (!d || !d.teams) { host.innerHTML = `<div class="empty">No tournament data yet. Run <code>python scripts/export_web.py</code>.</div>`; return; }
  const teams = [...d.teams].sort((a, b) => b.p_champion - a.p_champion);
  const max = Math.max(...teams.map((t) => t.p_champion), 0.01);
  const leaders = teams.slice(0, 4).map((t, i) => `<div class="leader">
    <span class="rank">${String(i + 1).padStart(2, "0")}</span>
    <div><strong>${t.team}</strong><small>Group ${t.group}</small></div>
    <b>${pct(t.p_champion)}</b>
    <i style="width:${(t.p_champion / max) * 100}%"></i>
  </div>`).join("");
  const rows = teams.map((t) => `
    <tr>
      <td>${t.team}</td><td>${t.group ?? ""}</td>
      <td>${pct(t.p_group_winner)}</td>
      <td>${pct(t.p_quarterfinal)}</td>
      <td>${pct(t.p_semifinal)}</td>
      <td>${pct(t.p_final)}</td>
      <td>${pct(t.p_champion)} <span class="bar" style="width:${(t.p_champion / max) * 70}px"></span></td>
    </tr>`).join("");
  host.innerHTML = `${sectionHead("Tournament forecast", "The field, priced", "Advancement probabilities from the current model and official draw.", `${teams.length} teams`)}
    <div class="leader-grid">${leaders}</div>
    <div class="table-shell"><table><thead><tr><th>Team</th><th>Grp</th><th>Win group</th><th>Reach QF</th>
    <th>Reach SF</th><th>Reach final</th><th>Champion</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function renderScorecard(d, history) {
  const host = document.getElementById("results");
  const s = (d && d.summary) || {};
  const games = (d && d.games) || [];
  if (!games.length) {
    host.innerHTML = sectionHead("Model audit", "Results", "Locked pre-match probabilities scored against final results.") +
      `<div class="empty">No games scored yet.</div>`;
    return;
  }
  const beatRandom = s.avg_logloss != null && s.avg_logloss < s.random_logloss;
  const actualCorrect = Math.round(s.accuracy * s.n);
  const expectedCorrect = s.expected_correct ?? games.reduce((n, g) => n + g.model_pick_prob, 0);
  const actualDraws = s.actual_draws ?? games.filter((g) => g.result === "D").length;
  const expectedDraws = s.expected_draws ?? games.reduce((n, g) => n + g.p_draw, 0);
  const skill = s.logloss_skill ?? (1 - s.avg_logloss / s.random_logloss);
  const deltaGames = actualCorrect - expectedCorrect;
  const drawGap = actualDraws - expectedDraws;
  const cards = [
    ["Games scored", s.n],
    ["Top-pick record", `${actualCorrect}–${s.n - actualCorrect}`],
    ["Expected correct", expectedCorrect.toFixed(2)],
    ["Avg log-loss", s.avg_logloss],
    ["Random baseline", s.random_logloss],
    ["Log-loss skill", signed(skill)],
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
    charts = intro("Trend charts appear after a few scored snapshots.");
  }

  const audit = `<div class="audit ${beatRandom ? "good" : "warn"}">
    <div><span>Probability score</span><strong>${beatRandom ? "Ahead of random" : "Behind random"}</strong></div>
    <p>Top-pick results are ${Math.abs(deltaGames).toFixed(1)} games ${deltaGames >= 0 ? "above" : "below"} expectation.
    Draws landed ${Math.abs(drawGap).toFixed(1)} ${drawGap >= 0 ? "more" : "fewer"} times than the model's total draw probability. At ${s.n} games, treat both as early signal.</p>
  </div>`;

  host.innerHTML = `${sectionHead("Model audit", "Results", `Locked forecasts scored before kickoff. Lower log-loss is better; ${s.random_logloss} is the equal-probability baseline.`, `${s.n} decisions`)}
    <div class="cards">${cards}</div>
    ${audit}
    ${charts}
    <div class="table-shell"><table><thead><tr><th>Game</th><th>Model (win/draw/win)</th><th>Top pick</th><th>Actual</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

const VENUE = { polymarket: "Polymarket", kalshi: "Kalshi" };

function renderEdges(d) {
  EDGES = d || EDGES;
  const host = document.getElementById("edges");
  const all = (EDGES && EDGES.opportunities) || [];
  const count = (v) => all.filter((o) => v === "all" || o.platform === v).length;
  const pills = [["all", "All venues"], ["polymarket", VENUE.polymarket], ["kalshi", VENUE.kalshi]]
    .map(([v, l]) => `<button class="venue-pill ${edgeVenue === v ? "active" : ""}" data-venue="${v}">
      <span class="venue-dot ${v}"></span><span>${l}</span><b>${count(v)}</b></button>`).join("");

  const ops = all.filter((o) => edgeVenue === "all" || o.platform === edgeVenue);
  const header = `${sectionHead("Live price comparison", "Market board", "Filtered disagreements after spread, liquidity, and market-prior shrinkage.", `${all.length} qualified gaps`)}
    <div class="venue-filter">${pills}</div>`;

  if (!ops.length) {
    host.innerHTML = header + `<div class="empty">No qualified gaps on ${edgeVenue === "all" ? "either venue" : VENUE[edgeVenue]} in this snapshot.</div>`;
    bindVenueFilters(host);
    return;
  }
  const rows = ops.map((o) => {
    const side = o.side === "YES" ? "BUY" : "FADE";
    const market = o.market_type === "match"
      ? (o.contract || "").replace(/\s*Winner\??$/i, "")
      : (o.market_type ?? "");
    return `<tr>
      <td><span class="venue ${o.platform}">${o.platform}</span> ${o.team ?? o.contract}</td>
      <td><span class="tag ${side}">${side}</span></td>
      <td>${market}</td>
      <td>${pct(o.model_prob)}</td>
      <td>${pct(o.market_prob)}</td>
      <td class="pos">${signed(o.edge)}</td>
      <td>${pct(o.entry_price)}</td>
      <td>${money(o.liquidity)}</td>
    </tr>`;
  }).join("");
  host.innerHTML = header +
    `<div class="table-shell"><table><thead><tr><th>Contract</th><th>Action</th><th>Market</th><th>Model</th>
    <th>Market</th><th>Gap</th><th>Entry</th><th>Liquidity</th></tr></thead><tbody>${rows}</tbody></table></div>`;

  bindVenueFilters(host);
}

function bindVenueFilters(host) {
  host.querySelectorAll("[data-venue]").forEach((b) =>
    b.addEventListener("click", () => { edgeVenue = b.dataset.venue; renderEdges(); }));
}

function renderPaper(d) {
  const host = document.getElementById("paper");
  const heading = sectionHead("Paper execution", "Model ledger", "Fractional-Kelly positions opened from qualified market gaps. No real money.");
  if (!d || !d.summary || !d.summary.n_bets) {
    host.innerHTML = heading + `<div class="empty">No open model bets right now — the model opens positions automatically when markets show enough edge. Check back after the next refresh.</div>`;
    return;
  }
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
  host.innerHTML = `${heading}
    <div class="cards">${cards}</div>
    ${bets.length ? `<div class="table-shell"><table><thead><tr><th>Contract</th><th>Side</th><th>Entry</th>
    <th>Last</th><th>CLV</th><th>Stake</th><th>Status</th></tr></thead><tbody>${rows}</tbody></table></div>`
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
  status.textContent = "Loading latest model snapshot…";
  let data = {};
  try {
    data = await loadFromJson();
  } catch (e) {
    status.textContent = "Couldn't load data — try a refresh.";
  }
  renderTournament(data.tournament);
  renderScorecard(data.scorecard, data.history);
  renderEdges(data.edges);
  renderPaper(data.ledger);
  if (window.FANTASY) window.FANTASY.init(data);

  const m = data.meta || {};
  document.getElementById("meta").textContent =
    `updated ${m.generated_at || "—"}`
    + `${m.n_matches ? " · " + m.n_matches.toLocaleString() + " matches" : ""}`
    + `${m.n_sims ? " · " + m.n_sims.toLocaleString() + " sims" : ""}`;
  const failures = Object.keys(data.__errors || {});
  status.textContent = failures.length
    ? `Snapshot warning: ${failures.join(", ")} failed to load.`
    : (m.note || "");
}

main();
