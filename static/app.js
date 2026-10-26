"use strict";

const SUIT_SYMBOL = { S: "\u2660", H: "\u2665", D: "\u2666", C: "\u2663" };
const RED = new Set(["H", "D"]);

let gameId = null;
let currentSteps = [];

// ---- setup phase ----

function buildColumnInputs() {
  const grid = document.getElementById("columnInputs");
  grid.innerHTML = "";
  for (let i = 0; i < 7; i++) {
    const label = document.createElement("label");
    label.className = "field";
    label.innerHTML = `
      <span>Column ${i + 1} (bottom-first, "?" = face-down)</span>
      <input type="text" data-col="${i}" placeholder="e.g. ? ? 5H 4S 3H" />
    `;
    grid.appendChild(label);
  }
}

const FOUNDATION_SUITS = [
  { suit: "S", label: "Spades \u2660" },
  { suit: "H", label: "Hearts \u2665" },
  { suit: "D", label: "Diamonds \u2666" },
  { suit: "C", label: "Clubs \u2663" },
];

function buildFoundationInputs() {
  const grid = document.getElementById("foundationInputs");
  grid.innerHTML = "";
  for (const { suit, label } of FOUNDATION_SUITS) {
    const field = document.createElement("label");
    field.className = "field";
    field.innerHTML = `
      <span>${label}</span>
      <input type="text" data-foundation="${suit}" placeholder="e.g. A${suit}" />
    `;
    grid.appendChild(field);
  }
}

function parseTokens(text) {
  // Keep "?" as-is; uppercase card codes.
  return text.trim().split(/\s+/).filter(Boolean)
    .map((t) => (t === "?" ? "?" : t.toUpperCase()));
}

function collectLayout() {
  const columns = [];
  for (let i = 0; i < 7; i++) {
    const input = document.querySelector(`input[data-col="${i}"]`);
    columns.push(parseTokens(input.value));
  }
  const foundations = {};
  for (const { suit } of FOUNDATION_SUITS) {
    const input = document.querySelector(`input[data-foundation="${suit}"]`);
    const codes = parseTokens(input.value);
    if (codes.length) foundations[suit] = codes;
  }
  const stock = parseTokens(document.getElementById("stockInput").value);
  const waste = parseTokens(document.getElementById("wasteInput").value);
  return { columns, foundations, stock, waste, drawCount: 3 };
}

async function startGame() {
  const errEl = document.getElementById("setupError");
  errEl.textContent = "";
  let layout;
  try {
    layout = collectLayout();
  } catch (e) {
    errEl.textContent = e.message;
    return;
  }
  const status = document.getElementById("setupError");
  status.textContent = "Analyzing the position\u2026";
  const res = await fetch("/api/new", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(layout),
  });
  const data = await res.json();
  if (!res.ok) {
    errEl.textContent = data.error || "Could not start.";
    return;
  }
  status.textContent = "";
  gameId = data.gameId;

  if (data.gameOver) {
    // No legal moves at all.
    enterSolvePhase(data.state);
    applyStepsResponse(data);
    return;
  }

  gameId = data.gameId;
  enterSolvePhase(data.state);
  applyStepsResponse(data);
}

function fillSample() {
  // A real mid-game position with 17 face-down cards and the full known stock.
  const columns = [
    ["10D"],
    ["?", "JS"],
    ["?", "?", "5D"],
    ["KC"],
    ["?", "?", "?", "?", "7S"],
    ["?", "?", "?", "?", "6C", "6H"],
    ["?", "?", "?", "?", "?", "?", "10C"],
  ];
  const foundations = { S: ["AS"], D: ["AD"], H: ["AH"] };
  const waste = [];
  const stock = [
    "5H", "5C", "QD", "4D", "10H", "QC", "3C", "9D", "2S", "JC", "7H", "QH",
    "4S", "9C", "KD", "6D", "3D", "9S", "QS", "8H", "9H", "JH", "3S", "10S",
  ];

  columns.forEach((col, i) => {
    document.querySelector(`input[data-col="${i}"]`).value = col.join(" ");
  });
  for (const { suit } of FOUNDATION_SUITS) {
    const input = document.querySelector(`input[data-foundation="${suit}"]`);
    input.value = (foundations[suit] || []).join(" ");
  }
  document.getElementById("stockInput").value = stock.join(" ");
  document.getElementById("wasteInput").value = waste.join(" ");
}

// ---- solve phase ----

function enterSolvePhase(state) {
  document.getElementById("setup").classList.add("hidden");
  document.getElementById("solve").classList.remove("hidden");
  renderBoard(state);
}

function cardEl(code, faceUp = true, mini = false) {
  const el = document.createElement("div");
  el.className = "card" + (mini ? " mini" : "");
  if (!faceUp) {
    el.classList.add("facedown");
    el.textContent = "";
    return el;
  }
  el.classList.add("faceup");
  if (code === "?" || code == null) {
    // Face-up but unknown: a just-flipped card awaiting reveal.
    el.classList.add("unknown");
    el.textContent = "?";
    return el;
  }
  const suit = code.slice(-1);
  const rank = code.slice(0, -1);
  el.classList.add(RED.has(suit) ? "red" : "black");
  el.textContent = `${rank}${SUIT_SYMBOL[suit]}`;
  return el;
}

function slotEl(label) {
  const el = document.createElement("div");
  el.className = "card slot";
  el.textContent = label || "";
  return el;
}

function renderBoard(state) {
  const board = document.getElementById("board");
  board.innerHTML = "";

  // top row: foundations + stock/waste
  const topRow = document.createElement("div");
  topRow.className = "top-row";

  const foundations = document.createElement("div");
  foundations.className = "foundations";
  for (const suit of ["S", "H", "D", "C"]) {
    const pile = state.foundations[suit];
    const wrap = document.createElement("div");
    wrap.className = "pile-wrap";
    if (pile.length) {
      wrap.appendChild(cardEl(pile[pile.length - 1], true, true));
    } else {
      wrap.appendChild(slotEl(SUIT_SYMBOL[suit]));
    }
    foundations.appendChild(wrap);
  }

  const stockWaste = document.createElement("div");
  stockWaste.className = "stock-waste";

  const stockWrap = document.createElement("div");
  stockWrap.className = "pile-wrap";
  const stockCard = state.stockCount > 0 ? cardEl("", false, true) : slotEl("\u21bb");
  stockWrap.appendChild(stockCard);
  stockWrap.appendChild(Object.assign(document.createElement("small"), { textContent: `stock ${state.stockCount}` }));
  stockWaste.appendChild(stockWrap);

  const wasteWrap = document.createElement("div");
  wasteWrap.className = "pile-wrap";
  const fan = state.wasteFan || [];
  if (fan.length) {
    // Fan the current draw horizontally; the last card is the playable top.
    const fanRow = document.createElement("div");
    fanRow.className = "waste-fan";
    fan.forEach((c, i) => {
      const card = cardEl(c.code, true, true);
      if (i === fan.length - 1) card.classList.add("waste-top-playable");
      fanRow.appendChild(card);
    });
    wasteWrap.appendChild(fanRow);
  } else {
    wasteWrap.appendChild(slotEl(""));
  }
  wasteWrap.appendChild(Object.assign(document.createElement("small"), { textContent: `waste ${state.waste.length}` }));
  stockWaste.appendChild(wasteWrap);

  topRow.appendChild(foundations);
  topRow.appendChild(stockWaste);
  board.appendChild(topRow);

  // tableau
  const tableau = document.createElement("div");
  tableau.className = "tableau";
  state.tableau.forEach((col, i) => {
    const colEl = document.createElement("div");
    const wrap = document.createElement("div");
    wrap.innerHTML = `<div class="col-label">Col ${i + 1}</div>`;
    colEl.className = "column";
    if (col.length === 0) {
      colEl.appendChild(slotEl("K"));
    } else {
      col.forEach((c) => colEl.appendChild(cardEl(c.code, c.faceUp)));
    }
    wrap.appendChild(colEl);
    tableau.appendChild(wrap);
  });
  board.appendChild(tableau);
}

async function loadSteps() {
  setBusy(true, "Computing next steps\u2026");
  try {
    const res = await fetch(`/api/steps/${gameId}?n=3`);
    const data = await res.json();
    applyStepsResponse(data);
  } finally {
    setBusy(false);
  }
}

function applyStepsResponse(data) {
  const status = document.getElementById("solveStatus");
  const info = document.getElementById("planInfo");
  if (data.state) renderBoard(data.state);
  updateUndoButton(data.canUndo);

  // Won.
  if (data.solved || (data.state && data.state.won)) {
    currentSteps = [];
    renderSteps([]);
    info.textContent = "";
    status.textContent = "\u{1f389} You won! All foundations complete.";
    status.className = "status win";
    return;
  }
  // Truly stuck: no legal moves.
  if (data.gameOver) {
    currentSteps = [];
    renderSteps([]);
    info.textContent = "";
    status.textContent = data.message || "No legal moves left. Game over.";
    status.className = "status gameover";
    return;
  }
  // Otherwise we have recommended steps.
  currentSteps = data.steps || [];
  const hidden = data.state ? data.state.faceDownCount : 0;
  const src = data.mc ? "Monte Carlo" : "greedy";
  info.textContent = (hidden
    ? `${hidden} face-down card${hidden === 1 ? "" : "s"} remaining`
    : "") + ` \u00b7 ${src}`;
  // Proven win: show the guarantee and skip any probability estimation.
  if (data.exactGuaranteed) {
    const mcInfo = document.getElementById("mcInfo");
    mcInfo.textContent = "\u2705 Guaranteed win \u2014 following the solved line.";
    mcInfo.className = "mc-info guaranteed";
  } else if (data.mc && typeof data.winProbability === "number") {
    // When the plan came from Monte Carlo, keep the win-probability readout fresh.
    renderMcResult(data);
  } else if (!data.gameOver && !data.solved) {
    // Greedy mode: fetch the CHEAP position win-probability in the background
    // so the readout appears without the cost of the per-move analysis.
    fetchWinProb();
  }
  status.textContent = data.revealNext
    ? "The next move flips a face-down card \u2014 accept it, then tell me what it is."
    : "";
  status.className = "status";
  renderSteps(currentSteps);
}

async function fetchWinProb() {
  if (!gameId) return;
  const mcInfo = document.getElementById("mcInfo");
  mcInfo.textContent = "Estimating win probability\u2026";
  mcInfo.className = "mc-info working";
  const gid = gameId;
  try {
    const res = await fetch(`/api/winprob/${gid}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (gid !== gameId) return;  // state moved on; drop stale result
    const data = await res.json();
    if (!res.ok) { mcInfo.textContent = ""; mcInfo.className = "mc-info"; return; }
    // Position estimate has no per-move breakdown; render just the probability.
    renderMcResult({ ...data, perMove: [] });
  } catch (e) {
    mcInfo.textContent = ""; mcInfo.className = "mc-info";
  }
}

function renderSteps(steps) {
  const list = document.getElementById("stepsList");
  list.innerHTML = "";
  if (steps.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No steps to show.";
    list.appendChild(li);
    return;
  }
  steps.forEach((step, i) => {
    const li = document.createElement("li");
    li.className = "step" + (i === 0 ? " pending-next" : "");
    const left = document.createElement("div");
    left.className = "desc";
    const badge = step.revealsHidden
      ? ' <span class="reveal-badge">flips a hidden card</span>'
      : "";
    left.innerHTML = `<span class="num">${i + 1}.</span>${step.description}${badge}`;
    li.appendChild(left);

    if (i === 0) {
      const actions = document.createElement("div");
      actions.className = "actions";
      const accept = document.createElement("button");
      accept.className = "accept";
      accept.textContent = "Accept";
      accept.onclick = () => acceptStep(step);
      const reject = document.createElement("button");
      reject.className = "reject";
      reject.textContent = "Reject";
      reject.onclick = () => rejectStep(step.move);
      actions.appendChild(accept);
      actions.appendChild(reject);
      li.appendChild(actions);
    }
    list.appendChild(li);
  });
}

// ---- busy indicator ----

let busyDepth = 0;

function setBusy(on, message) {
  busyDepth = Math.max(0, busyDepth + (on ? 1 : -1));
  const active = busyDepth > 0;
  document.body.classList.toggle("busy", active);
  // Disable in-flight-sensitive buttons while a request runs. "New game"
  // (resetBtn) stays enabled so the user can always bail out.
  ["undoBtn", "refreshStepsBtn", "analyzeBtn", "deepAnalyzeBtn"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.disabled = active;
  });
  document.querySelectorAll(".step .actions button").forEach((b) => { b.disabled = active; });
  if (active && message) {
    const status = document.getElementById("solveStatus");
    status.innerHTML = `<span class="spinner"></span>${message}`;
    status.className = "status working";
  }
}

async function acceptStep(step) {
  const move = step.move;
  // Skip the (possibly expensive) recompute for a plain DRAW: it only moves
  // known stock to the waste, reveals nothing, and doesn't change the planned
  // continuation — so the precomputed steps 2..n are still exactly right.
  // (Other move types can change the best line, and a reveal invalidates the
  // rest of the plan, so those always recompute.)
  const remaining = currentSteps.slice(1);
  const canSkip = move.kind === "draw" && !step.revealsHidden && remaining.length > 0;

  setBusy(true, "Applying the move\u2026");
  let data;
  try {
    const res = await fetch(`/api/accept/${gameId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ move, skipPlan: canSkip }),
    });
    data = await res.json();
    if (!res.ok) {
      document.getElementById("solveStatus").textContent = data.error || "Move rejected by server.";
      return;
    }
  } finally {
    setBusy(false);
  }
  renderBoard(data.state);

  // Fast path: reuse the precomputed continuation, no recompute needed.
  if (data.skippedPlan) {
    updateUndoButton(data.canUndo);
    currentSteps = remaining;
    renderSteps(currentSteps);
    const status = document.getElementById("solveStatus");
    status.textContent = "";
    status.className = "status";
    return;
  }

  // If this move flipped a face-down card, ask the user what it really is.
  // Keep asking until a valid card is entered (or the user cancels), so a
  // typo never corrupts the game — it just re-prompts.
  if (data.revealNeeded) {
    const col = data.revealNeeded.col;
    let errorHint = "";
    while (true) {
      const real = await promptRevealedCard(col, errorHint);
      if (!real) {
        document.getElementById("solveStatus").textContent =
          "Flip revealed a card \u2014 tell me what it is to continue. (You can also Undo.)";
        return;
      }
      setBusy(true, "Updating with the revealed card\u2026");
      let rdata, ok;
      try {
        const rres = await fetch(`/api/reveal/${gameId}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ col, card: real }),
        });
        ok = rres.ok;
        rdata = await rres.json();
      } finally {
        setBusy(false);
      }
      if (ok) {
        applyStepsResponse(rdata);
        return;
      }
      // Invalid card (duplicate, typo, or not a hidden card): re-prompt with
      // the reason, instead of leaving the game stuck.
      errorHint = rdata.error || "That card can't be here. Try again.";
    }
  }

  // No reveal: show the recomputed plan.
  applyStepsResponse(data);
}

function promptRevealedCard(col, errorHint) {
  const prefix = errorHint ? `\u26a0\ufe0f ${errorHint}\n\n` : "";
  const answer = window.prompt(
    `${prefix}Column ${col + 1}: you flipped a face-down card.\n\n` +
    `Enter the card you see (e.g. 7C, 10H, KS):`
  );
  if (answer === null) return null;
  const trimmed = answer.trim().toUpperCase();
  return trimmed === "" ? null : trimmed;
}

async function rejectStep(move) {
  setBusy(true, "Finding an alternative move\u2026");
  try {
    const res = await fetch(`/api/reject/${gameId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ move, n: 3 }),
    });
    const data = await res.json();
    if (!res.ok) {
      document.getElementById("solveStatus").textContent = data.error || "Could not find an alternative.";
      return;
    }
    applyStepsResponse(data);
  } finally {
    setBusy(false);
  }
}

function resetGame() {
  gameId = null;
  currentSteps = [];
  document.getElementById("solve").classList.add("hidden");
  document.getElementById("setup").classList.remove("hidden");
  document.getElementById("solveStatus").textContent = "";
  document.getElementById("mcInfo").textContent = "";
  document.getElementById("mcModeToggle").checked = false;
  updateUndoButton(false);
}

// ---- undo ----

async function undoMove() {
  if (!gameId) return;
  const res = await fetch(`/api/undo/${gameId}`, { method: "POST" });
  const data = await res.json();
  if (!res.ok) {
    document.getElementById("solveStatus").textContent = data.error || "Could not undo.";
    return;
  }
  applyStepsResponse(data);
  if (data.message === "Nothing to undo.") {
    document.getElementById("solveStatus").textContent = "Nothing to undo.";
  } else {
    document.getElementById("solveStatus").textContent = "Undid the last move.";
  }
}

function updateUndoButton(canUndo) {
  const btn = document.getElementById("undoBtn");
  if (btn) btn.disabled = !canUndo;
}

// ---- Monte Carlo ----

async function analyzeMonteCarlo(deep = false) {
  if (!gameId) return;
  const mcInfo = document.getElementById("mcInfo");
  mcInfo.textContent = deep
    ? "Running DEEP Monte Carlo analysis\u2026 (more samples, this can take ~15-20s)"
    : "Running Monte Carlo analysis\u2026 (this can take a few seconds)";
  mcInfo.className = "mc-info working";
  setBusy(true, deep ? "Deep analyzing\u2026" : "Analyzing\u2026");
  try {
    const res = await fetch(`/api/analyze/${gameId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ deep }),
    });
    const data = await res.json();
    if (!res.ok) {
      mcInfo.textContent = data.error || "Analysis failed.";
      mcInfo.className = "mc-info";
      return;
    }
    renderMcResult(data);
  } finally {
    setBusy(false);
  }
}

function fmtProb(rate, margin, exact) {
  const pct = Math.round((rate || 0) * 100);
  if (exact) return `${pct}%`;
  if (typeof margin === "number" && margin > 0) {
    return `${pct}% \u00b1 ${Math.round(margin * 100)}%`;
  }
  return `${pct}%`;
}

function renderMcResult(data) {
  const mcInfo = document.getElementById("mcInfo");
  const winProbability = data.winProbability;
  const margin = data.winMargin;
  const exact = data.exact;
  const samples = (typeof data.samples === "number") ? data.samples : 0;
  const perMove = data.perMove;
  const timedOut = data.timedOut;

  const deepTag = data.deep ? "deep, " : "";
  const head = exact
    ? `Win probability: ${fmtProb(winProbability, 0, true)} (exact \u2014 all layouts checked)`
    : `Estimated win probability: ${fmtProb(winProbability, margin, false)} ` +
      `(${deepTag}${samples} sampled layouts${timedOut ? ", time-limited" : ""})`;
  const lines = [head];
  if (perMove && perMove.length) {
    const top = perMove.slice(0, 4)
      .map((m) => `${fmtProb(m.winRate, m.margin, exact)} \u2014 ${m.description}`);
    lines.push("Best moves: " + top.join("  |  "));
  }
  mcInfo.textContent = lines.join("\n");
  const pct = Math.round((winProbability || 0) * 100);
  mcInfo.className = "mc-info" + (pct === 0 ? " zero" : "");
}

async function toggleMcMode(on) {
  if (!gameId) return;
  const mcInfo = document.getElementById("mcInfo");
  mcInfo.textContent = on
    ? "Monte Carlo now drives recommendations (slower, stronger)\u2026"
    : "";
  mcInfo.className = "mc-info working";
  const res = await fetch(`/api/mcmode/${gameId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ on }),
  });
  const data = await res.json();
  if (!res.ok) {
    mcInfo.textContent = data.error || "Could not change mode.";
    return;
  }
  applyStepsResponse(data);
  if (data.mc) {
    renderMcResult(data);
  } else {
    mcInfo.textContent = "";
    mcInfo.className = "mc-info";
  }
}

// ---- wire up ----

document.addEventListener("DOMContentLoaded", () => {
  buildColumnInputs();
  buildFoundationInputs();
  document.getElementById("startBtn").onclick = startGame;
  document.getElementById("fillSampleBtn").onclick = fillSample;
  document.getElementById("refreshStepsBtn").onclick = loadSteps;
  document.getElementById("resetBtn").onclick = resetGame;
  document.getElementById("analyzeBtn").onclick = () => analyzeMonteCarlo(false);
  document.getElementById("deepAnalyzeBtn").onclick = () => analyzeMonteCarlo(true);
  document.getElementById("mcModeToggle").onchange = (e) => toggleMcMode(e.target.checked);
  document.getElementById("undoBtn").onclick = undoMove;
});
