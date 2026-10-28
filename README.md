<<<<<<< HEAD
# klondike
Solitare draw 3 solver
=======
# Solitaire Solver

A Klondike solitaire solver with a web interface. You enter a **mid-game
position** (including face-down cards you can't see), and the solver tells you
whether it can be won and walks you through the solution three steps at a time.
If no solution exists, it says **Game over**.

## Variant / rules

- 7 tableau columns
- 4 foundations (build up by suit, A → K)
- Stock in the corner, **draw 3** at a time (recyclable)
- Only a **King** may move onto an empty column
- Tableau builds: alternating color, descending (e.g. red 6 on black 7)
- A column's face-up run moves **as a whole** — you cannot split it and move
  only the top part

## Handling hidden cards

Real positions have face-down cards (in the tableau) you can't see. Because the
game uses a full 52-card deck, the **identities** of the hidden cards are fixed
by elimination (they're whatever isn't visible), but their **order** within a
hidden stack isn't known. The solver:

1. Works out the set of hidden cards from everything you entered.
2. Tries arrangements of those cards into the face-down slots and looks for one
   that wins (optimistic: it assumes the buried cards fall favorably).
3. As you play and actually flip a face-down card, you tell the solver its real
   identity. If it matches the assumption, play continues; if not, the solver
   corrects the position and re-plans.

The stock is a **fixed cycling sequence**, so you enter it in full — only the
tableau face-down cards are genuinely unknown.

## Stack

- **Python + Flask** — game engine, solver, and JSON API
- **JavaScript** (vanilla) + HTML/CSS — web interface

## Project layout

```
app.py                 Flask app and API endpoints
solver/
  cards.py             Card model, parsing, stacking rules
  game.py              GameState, Move, layout construction
  engine.py            Legal move generation + move application (no run splitting)
  resolver.py          Mid-game parsing + hidden-card elimination
  solver.py            Exact best-first search (needs all cards known)
  strategy.py          Greedy strategy planner over the known, opaque state
  montecarlo.py        Monte Carlo determinization (play under uncertainty)
  storage.py           SQLite persistence of submitted positions
static/
  index.html           UI shell (setup + solve phases)
  styles.css           Styles / board rendering
  app.js               UI logic, API calls, accept/reject/reveal flow
check.py               Sanity checks for engine, resolver, and solver
requirements.txt
```

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Then open http://127.0.0.1:5001

## How to use

1. **Set up the position.**
   - **Columns.** One input per column, bottom-first. Type card codes for
     face-up cards and `?` for each face-down card. Rank + suit letter
     (`S H D C`), e.g. `AS`, `10H`, `KC`. Face-down `?` must come before the
     face-up cards in a column. Example: `? ? ? 5H 4S 3H`.
   - **Foundations.** The cards already moved to each foundation, ascending
     from the Ace. e.g. Diamonds `AD 2D 3D` if you're at 3♦.
   - **Stock.** The full fixed cycling sequence, in draw order.
   - **Waste.** The cards currently fanned out of the stock (top card last),
     if any. Optional.
   - Columns + foundations + stock + waste must total 52 unique cards. The
     `?` slots are filled from whatever is left over, so the counts must line
     up or you'll get a clear error.
2. **Solve.** The solver reports either a winning line (shown 3 steps at a
   time) or **Game over** if the position can't be won.
3. **Calibrate / play.** For the next step you can:
   - **Accept** — applies the move. If it flips a face-down card, you're asked
     what the real card is; enter it (or leave blank if the solver's guess was
     right) and the plan updates.
   - **Reject** — the solver finds a different line that avoids that move.

## Monte Carlo analysis

Because face-down cards are hidden, no strategy can guarantee a win — some
deals are simply unwinnable. The **Monte Carlo** analyzer gives the strongest
practical guidance for this uncertainty:

- It samples many concrete arrangements of the hidden cards (their identities
  are fixed by elimination; only their order varies), runs the exact solver on
  each, and reports an **estimated win probability** plus a per-move win-rate
  breakdown.
- Two ways to use it:
  - **Analyze (Monte Carlo)** button — on-demand analysis of the current
    position without changing the game.
  - **MC drives recommendations** toggle — makes Monte Carlo choose every
    recommended move (slower but stronger), for testing against the greedy
    planner.

Monte Carlo is expensive (samples x legal moves x a full solve each), so the
fast greedy planner remains the default for normal stepping. Sample count and
solve budget are tunable via the `/api/analyze` body (`samples`, `budget`,
`timeBudget`).

### Effort and confidence

Accuracy is balanced against compute time automatically, following the
accepted practice for determinized Monte Carlo (the "tens of determinizations"
range from the Klondike research literature, with diminishing returns beyond):

- **Few hidden cards** (tiny arrangement space): all arrangements are
  enumerated, giving an **exact** win probability (no sampling error).
- **Moderate** (up to ~14 hidden): the classic **40-80 sample** sweet spot.
- **Many** (15+ hidden): capped sampling plus a wall-clock budget; the estimate
  is necessarily rough, but the move *ranking* usually remains reliable.

Every non-exact estimate reports a **95% confidence interval** (Wilson score),
shown as e.g. `34% ± 9%`, so you can tell how much to trust the number. On
heavy positions a large margin is the honest signal that the probability is
noisy — it tightens as you reveal cards and the hidden count drops.

### Proven wins skip Monte Carlo

Once a position is a **guaranteed win** — either fully known, or few enough
hidden cards that every arrangement is enumerable and all win — the app follows
the exact solver's winning line instead of Monte Carlo (even if the MC toggle
is on). This is faster and certain: you get a proven move-by-move path to the
finish. A merely *sampled* 100% does NOT trigger this; only an exact,
enumerated guarantee does, so there's no risk of a lurking losing arrangement.

## Notes on the solver

The solver uses a weighted best-first search with state deduplication, wrapped
in an arrangement search over the hidden cards. It finds a winning line for
most solvable positions within its budget, but it is not guaranteed to find the
shortest solution. Tunable via body parameters on `/api/new`:
`budget` (search states) and `arrangements` (max hidden-card orderings tried).

## API

- `POST /api/new` — parse + solve a mid-game position. Body:
  `{columns, foundations, stock, waste, drawCount, engine}`. `engine` is one of
  `"greedy"` (instant greedy planner, default), `"fast_mc"` (Monte Carlo at the
  fast tier with a confidence short-circuit), or `"deep_mc"` (adaptive Monte
  Carlo: a fast pass that escalates to the deep tier only when the move is still
  ambiguous). Returns the solution's first 3 steps, or `gameOver: true`.
- `GET  /api/steps/<id>?n=3` — recompute the solution, return first `n` steps.
- `POST /api/accept/<id>` — apply an accepted move `{move}`; reports any
  `revealed` face-down cards.
- `POST /api/reveal/<id>` — tell the solver a flipped card's real identity
  `{col, card}`; re-plans if it differs from the assumption.
- `POST /api/reject/<id>` — reject a move `{move, n}`, return an alternative line.
- `POST /api/winprob/<id>` — cheap win-probability for the current position
  (one solve per sampled world, no per-move loop); used for the readout while
  stepping.
- `POST /api/analyze/<id>` — full per-move Monte Carlo comparison
  `{samples, budget, timeBudget}`; returns `winProbability` and a per-move
  win-rate breakdown. Skips the per-move loop when there's only one legal move.
- `POST /api/engine/<id>` — change the recommendation engine for a live game
  `{engine: "greedy" | "fast_mc" | "deep_mc"}`; recomputes and returns the
  current recommendation under the new engine.

### Saved positions (debugging)

Every position submitted to `/api/new` is saved to `games.db` (SQLite), including
ones that fail to parse — so you can inspect exactly what was entered when
something goes wrong. The raw input (columns, foundations, stock, waste,
draw_count), the solver verdict (`solved` = 1 solved / 0 game over / -1 input
error), the move count, the hidden-card assignment, and any error are stored.

- `GET  /api/games?limit=100` — list saved positions, newest first.
- `GET  /api/games/<id>` — full raw input + verdict for one position.
- `POST /api/games/<id>/replay` — reload a saved position and re-solve it.
- `POST /api/games/<id>/note` — attach a free-form note `{note}`.

Inspect the DB directly if you prefer:

```bash
sqlite3 games.db "SELECT id, created_iso, solved, total_moves, error FROM games ORDER BY created_at DESC;"
```

## Sanity checks

```bash
.venv/bin/python check.py
```
>>>>>>> 5a29921 (Monte Carlo solver with uncertainty handling and play-quality fixes)
