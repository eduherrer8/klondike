"""Empirical win-rate measurement for the MC-driven player on random deals.

Deals standard 52-card draw-3 Klondike games (seeded shuffle), plays each to
completion using the Monte Carlo planner (revealing true cards as they flip,
exactly like a real player would), and reports the fraction WON with a 95%
Wilson confidence interval.

Progress is written INCREMENTALLY to a results file (default: winrate.log) and
flushed after every game, so you can tail it while it runs and a crash on one
game never loses the games already finished. Each game is wrapped in try/except
so one bad deal can't abort the whole batch.

Usage:
    .venv-pypy/bin/pypy _winrate_batch.py [N] [SAMPLES] [SOLVE_BUDGET] [TIME_S] [OUTFILE] [START_SEED]

Defaults: N=1000 samples=60 solve_budget=40000 time_s=3.0 outfile=winrate.log start_seed=0

START_SEED lets you split work across machines without overlap: each PC plays
seeds START_SEED .. START_SEED+N-1. A given seed always produces the SAME deal,
so distinct seed ranges = distinct games that merge cleanly (just sum the
WON counts and the total games across machines).
"""
import sys
import time
import random
from collections import Counter

from solver.cards import full_deck, UNKNOWN
from solver.game import GameState, Move
from solver.engine import apply_move
from solver.montecarlo import mc_plan_steps
from solver.strategy import flips_hidden_card


def deal(seed):
    rng = random.Random(seed)
    deck = full_deck()
    rng.shuffle(deck)
    gs = GameState(draw_count=3)
    idx = 0
    truth = {}
    for n in range(1, 8):
        col = []
        for i in range(n):
            c = deck[idx + i]
            up = (i == n - 1)
            col.append((c, up))
        gs.tableau[n - 1] = col
        idx += n
    gs.stock = deck[idx:][::-1]
    for ci, col in enumerate(gs.tableau):
        for pi, (c, up) in enumerate(col):
            if not up:
                truth[(ci, pi)] = c
                col[pi] = (UNKNOWN, False)
    return gs, truth


def autoplay(seed, samples, solve_budget, time_s, max_steps=1000):
    st, truth = deal(seed)
    init_wp = None
    for step in range(max_steps):
        if st.is_won():
            return {'outcome': 'WON', 'steps': step, 'init_wp': init_wp}
        p = mc_plan_steps(st, n=1, samples=samples, solve_budget=solve_budget,
                          time_budget_s=time_s, settle=True,
                          rng=random.Random(seed * 100 + step))
        if init_wp is None:
            # Planner's estimate of the fresh deal's winnability, captured
            # BEFORE any move: our best proxy for "was this deal lucky?".
            init_wp = p.get('winProbability')
        if p['gameOver']:
            oc = 'STUCK' if p.get('stuck') else 'GAMEOVER'
            return {'outcome': oc, 'steps': step, 'init_wp': init_wp}
        if not p['steps']:
            return {'outcome': 'NOSTEPS', 'steps': step, 'init_wp': init_wp}
        mv = Move(**p['steps'][0]['move'])
        reveal_col = mv.src if flips_hidden_card(st, mv) else None
        st = apply_move(st, mv)
        if reveal_col is not None:
            col = st.tableau[reveal_col]
            pos = len(col) - 1
            real = truth.get((reveal_col, pos))
            if real is None:
                return {'outcome': 'REVEAL_MISS', 'steps': step,
                        'init_wp': init_wp}
            col[pos] = (real, True)
    return {'outcome': 'MAXSTEPS', 'steps': max_steps, 'init_wp': init_wp}


def wilson(wins, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 40_000
    time_s = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
    outfile = sys.argv[5] if len(sys.argv) > 5 else "winrate.log"
    start_seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0

    out = open(outfile, "w", buffering=1)  # line-buffered

    def emit(line):
        print(line)
        out.write(line + "\n")
        out.flush()

    started = time.time()
    end_seed = start_seed + n
    header = (f"# winrate batch | games={n} seeds={start_seed}..{end_seed - 1} "
              f"samples={samples} solve_budget={budget} time_budget_s={time_s} | "
              f"start={time.strftime('%Y-%m-%d %H:%M:%S')}")
    emit(header)

    res = Counter()
    wins = 0
    won_wps = []   # initial win-prob of games we WON
    lost_wps = []  # initial win-prob of games we LOST
    for i, seed in enumerate(range(start_seed, end_seed)):
        t0 = time.time()
        steps = 0
        init_wp = None
        try:
            r = autoplay(seed, samples, budget, time_s)
            outcome = r['outcome']
            steps = r['steps']
            init_wp = r['init_wp']
        except Exception as e:  # never let one game kill the batch
            outcome = 'ERROR'
            emit(f"# seed={seed} raised {type(e).__name__}: {e}")
        res[outcome] += 1
        if outcome == 'WON':
            wins += 1
            if init_wp is not None:
                won_wps.append(init_wp)
        elif outcome in ('STUCK', 'GAMEOVER', 'NOSTEPS', 'MAXSTEPS'):
            if init_wp is not None:
                lost_wps.append(init_wp)
        done = i + 1
        p, lo, hi = wilson(wins, done)
        elapsed = time.time() - started
        rate = done / elapsed if elapsed else 0.0
        eta_min = (n - done) / rate / 60 if rate else 0.0
        wp_str = f"{init_wp:.2f}" if init_wp is not None else " n/a"
        emit(f"seed={seed:5d} -> {outcome:10s} | "
             f"winrate={p:6.2%} (95% CI {lo:.2%}-{hi:.2%}) n={done} | "
             f"initWP={wp_str} steps={steps:3d} | "
             f"game={time.time()-t0:5.1f}s eta={eta_min:6.1f}min")

    p, lo, hi = wilson(wins, n)

    def avg(xs):
        return sum(xs) / len(xs) if xs else float('nan')

    emit("-" * 72)
    emit(f"# WIN RATE: {wins}/{n} = {p:.2%}  (95% Wilson CI {lo:.2%} - {hi:.2%})")
    emit("#")
    emit("# OUTCOME BREAKDOWN:")
    labels = {
        'WON':        'won                                   ',
        'STUCK':      'lost - STUCK (only stock-cycle left)  ',
        'GAMEOVER':   'lost - GAMEOVER (no legal moves)      ',
        'NOSTEPS':    'lost - NOSTEPS (planner found no move)',
        'MAXSTEPS':   'lost - MAXSTEPS (hit step cap)        ',
        'REVEAL_MISS':'bug  - REVEAL_MISS (reveal mismatch)  ',
        'ERROR':      'bug  - ERROR (exception raised)       ',
    }
    for key in ('WON', 'STUCK', 'GAMEOVER', 'NOSTEPS', 'MAXSTEPS',
                'REVEAL_MISS', 'ERROR'):
        c = res.get(key, 0)
        if c:
            emit(f"#   {labels[key]} : {c:4d}  ({c / n:6.2%})")
    emit("#")
    emit("# LUCK vs PLAY (initial win-prob = planner's estimate of the fresh deal):")
    emit(f"#   avg initWP of WON  games : {avg(won_wps):.3f}  (n={len(won_wps)})")
    emit(f"#   avg initWP of LOST games : {avg(lost_wps):.3f}  (n={len(lost_wps)})")
    emit("#   -> LOST games with HIGH initWP mean the planner lost winnable deals")
    emit("#      (bad play); LOST games with LOW initWP mean the deals were")
    emit("#      likely unwinnable to begin with (bad luck).")
    # Count lost games the planner itself thought were very winnable (>=0.8):
    blown = sum(1 for w in lost_wps if w is not None and w >= 0.8)
    emit(f"#   lost games with initWP >= 0.80 (likely 'blown' winnable deals): {blown}")
    emit("#")
    emit(f"# total wall time: {(time.time()-started)/60:.1f} min")
    out.close()


if __name__ == "__main__":
    main()
