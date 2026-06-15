// "Beat the Oracle" — fantasy prediction game.
// Each player gets a virtual bankroll, stakes it on World Cup markets at the
// model's implied odds, and tries to beat the model. State lives in the browser
// (localStorage) — no login, no backend. Cards are shareable via URL.

window.FANTASY = (() => {
  const START = 1000;
  const KEY = "wc2026-fantasy-v2";
  const MIN_PROB = 0.02, MAX_ODDS = 25;

  let DATA = { matches: { fixtures: [] }, tournament: { teams: [] } };
  let RESULTS = {};           // { marketId: winningOptionKey }
  let MARKETS = [];
  let MBY = {};
  let state = load();
  let sub = "matches";        // matches | champion | groups | card
  let sel = null;             // {marketId, optionKey, stake} being edited

  // ---- flags -------------------------------------------------------------
  const CC = { Mexico:"mx","South Korea":"kr","South Africa":"za","Czech Republic":"cz",
    Canada:"ca",Qatar:"qa","Bosnia and Herzegovina":"ba",Switzerland:"ch",Brazil:"br",
    Haiti:"ht",Morocco:"ma",Paraguay:"py",Turkey:"tr","United States":"us",Australia:"au",
    Ecuador:"ec",Germany:"de","Ivory Coast":"ci",Tunisia:"tn",Japan:"jp",Netherlands:"nl",
    Sweden:"se","New Zealand":"nz",Iran:"ir",Egypt:"eg",Belgium:"be",Uruguay:"uy",Spain:"es",
    "Saudi Arabia":"sa",Senegal:"sn",Norway:"no",France:"fr",Iraq:"iq",Algeria:"dz",Jordan:"jo",
    Argentina:"ar",Austria:"at",Colombia:"co",Portugal:"pt",Uzbekistan:"uz",Ghana:"gh",
    Croatia:"hr",Panama:"pa","Cape Verde":"cv","Curaçao":"cw","DR Congo":"cd" };
  const iso = (cc) => cc.toUpperCase().replace(/./g, (c) => String.fromCodePoint(0x1f1e6 - 65 + c.charCodeAt(0)));
  function flag(team) {
    if (team === "Draw") return "🤝";
    if (team === "England") return "🏴󠁧󠁢󠁥󠁮󠁧󠁿";
    if (team === "Scotland") return "🏴󠁧󠁢󠁳󠁣󠁴󠁿";
    return CC[team] ? iso(CC[team]) : "🏳️";
  }

  // ---- state -------------------------------------------------------------
  function fresh() { return { bankroll: START, picks: {}, created: Date.now() }; }
  function load() { try { return JSON.parse(localStorage.getItem(KEY)) || fresh(); } catch { return fresh(); } }
  function save() {
    localStorage.setItem(KEY, JSON.stringify(state));
    if (window.AUTH && window.AUTH.available && window.AUTH.currentUser()) window.AUTH.saveCard(state.picks);
  }

  const oddsOf = (p) => Math.min(MAX_ODDS, +(1 / Math.max(p, MIN_PROB)).toFixed(2));
  const picks = () => Object.values(state.picks);
  const pending = (p) => !p.status || p.status === "pending";
  const realized = () => picks().filter((p) => !pending(p))
    .reduce((s, p) => s + (p.status === "won" ? p.stake * (p.odds - 1) : -p.stake), 0);
  const balance = () => START + realized();
  const pendingStake = () => picks().filter(pending).reduce((s, p) => s + p.stake, 0);
  const free = () => balance() - pendingStake();
  const potential = () => picks().filter(pending).reduce((s, p) => s + p.stake * p.odds, 0);
  const record = () => ({ won: picks().filter((p) => p.status === "won").length,
                          lost: picks().filter((p) => p.status === "lost").length });
  const modelPick = (m) => m.options.reduce((a, b) => (b.prob > a.prob ? b : a)).key;
  const fadingCount = () => picks().filter((p) => pending(p) && p.optionKey !== modelPick(MBY[p.marketId])).length;
  const resultFor = (mId) => RESULTS[mId];

  // Settle pending picks against live results (auto-runs each load).
  function settle() {
    let changed = false;
    picks().forEach((p) => {
      if (!pending(p)) return;
      const r = RESULTS[p.marketId];
      if (r == null) return;
      p.status = r === p.optionKey ? "won" : "lost";
      changed = true;
    });
    if (changed) save();
  }

  // ---- build markets from data ------------------------------------------
  function buildMarkets() {
    MARKETS = []; MBY = {};
    (DATA.matches.fixtures || []).forEach((f) => {
      MARKETS.push({
        id: "m:" + f.key, kind: "matches", title: `${f.team_a} vs ${f.team_b}`,
        date: f.date, result: f.result,
        options: [
          { key: "A", team: f.team_a, label: f.team_a, prob: f.p_a },
          { key: "D", team: "Draw", label: "Draw", prob: f.p_draw },
          { key: "B", team: f.team_b, label: f.team_b, prob: f.p_b },
        ],
      });
    });
    const teams = DATA.tournament.teams || [];
    if (teams.length) {
      MARKETS.push({
        id: "champion", kind: "champion", title: "World Cup Winner",
        options: [...teams].sort((a, b) => b.p_champion - a.p_champion)
          .map((t) => ({ key: t.team, team: t.team, label: t.team, prob: Math.max(t.p_champion, 0.004) })),
      });
      const byGroup = {};
      teams.forEach((t) => { (byGroup[t.group] ||= []).push(t); });
      Object.keys(byGroup).sort().forEach((g) => {
        MARKETS.push({
          id: "grp:" + g, kind: "groups", title: `Group ${g} — Winner`,
          options: byGroup[g].sort((a, b) => b.p_group_winner - a.p_group_winner)
            .map((t) => ({ key: t.team, team: t.team, label: t.team, prob: Math.max(t.p_group_winner, 0.01) })),
        });
      });
    }
    MARKETS.forEach((m) => { MBY[m.id] = m; });
  }

  // ---- rendering ---------------------------------------------------------
  const fmt = (n) => Math.round(n).toLocaleString();

  function dashboard() {
    const rec = record(); const rp = realized();
    const pct = Math.min((pendingStake() / Math.max(balance(), 1)) * 100, 100);
    return `<div class="f-dash">
      <div class="f-stat"><span class="k">Bankroll</span><span class="v">${fmt(balance())}</span></div>
      <div class="f-stat"><span class="k">Free</span><span class="v">${fmt(free())}</span></div>
      <div class="f-stat"><span class="k">At stake</span><span class="v">${fmt(pendingStake())}</span></div>
      <div class="f-stat"><span class="k">Live payout</span><span class="v pos">${fmt(potential())}</span></div>
      <div class="f-stat"><span class="k">Record</span><span class="v">${rec.won}–${rec.lost}</span></div>
      <div class="f-stat"><span class="k">Settled P/L</span><span class="v ${rp >= 0 ? "pos" : "neg"}">${rp >= 0 ? "+" : "−"}${fmt(Math.abs(rp))}</span></div>
      <div class="f-stat"><span class="k">Against model</span><span class="v">${fadingCount()}</span></div>
      <div class="f-alloc"><div class="f-alloc-bar" style="width:${pct}%"></div></div>
    </div>`;
  }

  function subnav() {
    const tabs = [["matches", "Matches"], ["champion", "Champion"], ["groups", "Groups"],
      ["card", `My Card (${Object.keys(state.picks).length})`]];
    return `<div class="f-pills">${tabs.map(([k, l]) =>
      `<button class="f-pill ${sub === k ? "active" : ""}" data-action="sub" data-sub="${k}">${l}</button>`).join("")}</div>`;
  }

  function optionChip(m, o) {
    const isPick = state.picks[m.id]?.optionKey === o.key;
    const isSel = sel && sel.marketId === m.id && sel.optionKey === o.key;
    const isOracle = modelPick(m) === o.key;
    const res = resultFor(m.id);
    const settled = res != null;
    const winner = settled && res === o.key;
    const cls = ["f-opt", isPick && "picked", isSel && "sel", winner && "won",
      settled && !winner && "dim"].filter(Boolean).join(" ");
    return `<button class="${cls}" data-action="pick" data-m="${m.id}" data-o="${o.key}" ${settled ? "disabled" : ""}>
      <span class="f-opt-team">${flag(o.team)} ${o.label}${winner ? " ✓" : ""}</span>
      <span class="f-opt-meta">${(o.prob * 100).toFixed(0)}% · <b>${oddsOf(o.prob).toFixed(2)}×</b>${isOracle ? ` <span class="oracle">🔮</span>` : ""}</span>
    </button>`;
  }

  function stakeRow(m) {
    if (!sel || sel.marketId !== m.id) return "";
    const o = m.options.find((x) => x.key === sel.optionKey);
    const existing = state.picks[m.id]?.stake || 0;
    const cap = free() + existing;
    const stake = Math.min(sel.stake, cap);
    const chips = [10, 25, 50, 100].map((v) =>
      `<button class="f-chip" data-action="stake" data-m="${m.id}" data-v="${v}">${v}</button>`).join("")
      + `<button class="f-chip" data-action="stake" data-m="${m.id}" data-v="half">½</button>`
      + `<button class="f-chip" data-action="stake" data-m="${m.id}" data-v="max">Max</button>`;
    return `<div class="f-stake">
      <div class="f-stake-head">Stake on <b>${flag(o.team)} ${o.label}</b> @ ${oddsOf(o.prob).toFixed(2)}×</div>
      <div class="f-chips">${chips}</div>
      <input type="range" min="0" max="${cap}" value="${stake}" data-action="slider" data-m="${m.id}" class="f-slider" />
      <div class="f-stake-foot">
        <span>${fmt(stake)} at risk → <b class="pos">${fmt(stake * oddsOf(o.prob))}</b> payout</span>
        <span>
          <button class="f-btn ghost" data-action="cancel">Cancel</button>
          <button class="f-btn" data-action="place" data-m="${m.id}" ${stake <= 0 ? "disabled" : ""}>${existing ? "Update" : "Place"} pick</button>
        </span>
      </div>
    </div>`;
  }

  function marketCard(m) {
    const pick = state.picks[m.id];
    const head = m.kind === "matches"
      ? `<div class="f-card-head"><span class="f-date">${m.date || ""}</span><span class="f-title">${flag(m.options[0].team)} ${m.options[0].label} <span class="vs">vs</span> ${flag(m.options[2].team)} ${m.options[2].label}</span></div>`
      : `<div class="f-card-head"><span class="f-title">${m.title}</span></div>`;
    return `<div class="f-card ${pick ? "has-pick" : ""}">
      ${head}
      <div class="f-opts ${m.kind === "champion" || m.kind === "groups" ? "grid" : ""}">${m.options.map((o) => optionChip(m, o)).join("")}</div>
      ${stakeRow(m)}
    </div>`;
  }

  function viewMarkets(kind) {
    const ms = MARKETS.filter((m) => m.kind === kind);
    if (!ms.length) return `<div class="empty">No ${kind} available yet — run <code>python scripts/export_web.py</code>.</div>`;
    return `<div class="f-grid">${ms.map(marketCard).join("")}</div>`;
  }

  function viewCard() {
    const picks = Object.values(state.picks);
    if (!picks.length) return `<div class="empty">No picks yet. Make some on the Matches, Champion, or Groups tabs — then come back to see your card.</div>`;
    const rows = picks.map((p) => {
      const m = MBY[p.marketId];
      const fading = p.optionKey !== modelPick(m);
      const status = p.status === "won"
        ? `<span class="tag YES">WON +🪙${fmt(p.stake * (p.odds - 1))}</span>`
        : p.status === "lost"
          ? `<span class="tag FADE">LOST −🪙${fmt(p.stake)}</span>`
          : `<span class="tag">pending</span>`;
      const last = pending(p)
        ? `<button class="f-btn ghost sm" data-action="remove" data-m="${p.marketId}">✕</button>`
        : "";
      return `<tr class="row-${p.status || "pending"}">
        <td>${m.title}</td>
        <td>${flag(p.team)} ${p.label} ${fading ? `<span class="tag FADE">fade 🔮</span>` : `<span class="tag YES">with 🔮</span>`}</td>
        <td>${p.odds.toFixed(2)}×</td>
        <td>${fmt(p.stake)}</td>
        <td class="pos">${fmt(p.stake * p.odds)}</td>
        <td>${status}</td>
        <td>${last}</td>
      </tr>`;
    }).join("");
    return `<table><thead><tr><th>Market</th><th>Your pick</th><th>Odds</th><th>Stake</th><th>To win</th><th>Status</th><th></th></tr></thead><tbody>${rows}</tbody></table>
      <div class="f-card-actions">
        <button class="f-btn" data-action="share">Share card</button>
        <button class="f-btn ghost" data-action="reset">Reset everything</button>
      </div>
      <p class="f-note">Picks settle as results come in. Taking the other side of the model pays more when your read is right.</p>`;
  }

  function accountBar() {
    if (!window.AUTH || !window.AUTH.available) return "";
    const u = window.AUTH.currentUser();
    return u
      ? `<div class="f-acct">☁️ Saved to your account — ${u.displayName || u.email} · <button class="f-link" data-action="signout">sign out</button></div>`
      : `<div class="f-acct"><button class="f-btn sm" data-action="signin">Sign in to save your card across devices</button></div>`;
  }

  function render() {
    const host = document.getElementById("play");
    if (!host) return;
    const body = sub === "card" ? viewCard() : viewMarkets(sub);
    host.innerHTML = `<div class="fantasy">
      <div class="f-hero"><div class="eyebrow">Paper pick'em</div><h2>Build a card</h2><p>Start with 1,000 credits. Take the model's side or price the upset.</p></div>
      ${accountBar()}${dashboard()}${subnav()}<div class="f-body">${body}</div>
    </div>`;
  }

  // ---- interactions ------------------------------------------------------
  function onClick(e) {
    const t = e.target.closest("[data-action]");
    if (!t) return;
    const a = t.dataset.action;
    if (a === "sub") { sub = t.dataset.sub; sel = null; return render(); }
    if (a === "pick") {
      const m = MBY[t.dataset.m];
      if (resultFor(m.id) != null) return;   // market settled -> locked
      const cur = state.picks[m.id];
      sel = { marketId: m.id, optionKey: t.dataset.o, stake: cur?.stake || Math.min(50, free() + (cur?.stake || 0)) };
      return render();
    }
    if (a === "stake") {
      const m = MBY[t.dataset.m]; const existing = state.picks[m.id]?.stake || 0; const cap = free() + existing;
      const v = t.dataset.v;
      sel.stake = v === "max" ? cap : v === "half" ? Math.floor(cap / 2) : Math.min(+v, cap);
      return render();
    }
    if (a === "place") {
      const m = MBY[t.dataset.m]; const o = m.options.find((x) => x.key === sel.optionKey);
      state.picks[m.id] = { marketId: m.id, optionKey: o.key, team: o.team, label: o.label,
        prob: o.prob, odds: oddsOf(o.prob), stake: Math.max(0, Math.round(sel.stake)) };
      sel = null; save(); return render();
    }
    if (a === "remove") { delete state.picks[t.dataset.m]; save(); return render(); }
    if (a === "cancel") { sel = null; return render(); }
    if (a === "reset") { if (confirm("Reset your card and bankroll?")) { state = fresh(); sel = null; save(); render(); } return; }
    if (a === "share") return share();
    if (a === "signin" && window.AUTH) return window.AUTH.signIn();
    if (a === "signout" && window.AUTH) return window.AUTH.signOut();
  }

  function onInput(e) {
    const t = e.target.closest('[data-action="slider"]');
    if (!t || !sel) return;
    sel.stake = +t.value;
    // light update of the foot without full re-render jank
    render();
  }

  // ---- share -------------------------------------------------------------
  function share() {
    const compact = Object.values(state.picks).map((p) => [p.marketId, p.optionKey, p.stake]);
    const code = btoa(unescape(encodeURIComponent(JSON.stringify(compact))));
    const url = location.origin + location.pathname + "#card=" + code;
    location.hash = "card=" + code;
    const clip = navigator.clipboard;
    if (clip && clip.writeText) {
      clip.writeText(url).then(() => toast("Card link copied to clipboard!"),
        () => toast("Card saved to the URL — copy it from your address bar."));
    } else {
      toast("Card saved to the URL — copy it from your address bar.");
    }
  }

  function importHash() {
    const h = location.hash;
    if (!h.startsWith("#card=")) return;
    try {
      const arr = JSON.parse(decodeURIComponent(escape(atob(h.slice(6)))));
      if (Array.isArray(arr) && confirm("Load this shared prediction card? It replaces your current one.")) {
        state = fresh();
        arr.forEach(([mid, ok, st]) => {
          const m = MBY[mid]; if (!m) return;
          const o = m.options.find((x) => x.key === ok); if (!o) return;
          state.picks[mid] = { marketId: mid, optionKey: ok, team: o.team, label: o.label, prob: o.prob, odds: oddsOf(o.prob), stake: st };
        });
        save();
      }
    } catch {}
    history.replaceState(null, "", location.pathname);
  }

  function toast(msg) {
    let el = document.getElementById("f-toast");
    if (!el) { el = document.createElement("div"); el.id = "f-toast"; document.body.appendChild(el); }
    el.textContent = msg; el.classList.add("show");
    setTimeout(() => el.classList.remove("show"), 2200);
  }

  // ---- init --------------------------------------------------------------
  function init(data) {
    DATA = { matches: data.matches || { fixtures: [] }, tournament: data.tournament || { teams: [] } };
    RESULTS = (data.results && data.results.markets) || {};
    buildMarkets();
    importHash();
    settle();                 // auto-settle pending picks against live results
    const host = document.getElementById("play");
    host.addEventListener("click", onClick);
    host.addEventListener("input", onInput);

    // Optional cloud sync: adopt the user's saved card on sign-in (or push local up).
    if (window.AUTH && window.AUTH.available) {
      window.AUTH.onChange(async (u) => {
        if (u) {
          const cloud = await window.AUTH.loadCard();
          if (cloud && cloud.picks && Object.keys(cloud.picks).length) {
            state.picks = cloud.picks;
            localStorage.setItem(KEY, JSON.stringify(state));
          } else {
            window.AUTH.saveCard(state.picks);
          }
          settle();
        }
        render();
      });
    }
    render();
  }

  return { init, render };
})();
