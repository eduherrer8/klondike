import random
from collections import Counter
from solver.cards import full_deck, UNKNOWN, Card
from solver.game import GameState, Move
from solver.engine import apply_move, legal_moves
from solver.montecarlo import mc_plan_steps
from solver.strategy import flips_hidden_card

def deal(seed):
    rng = random.Random(seed); deck = full_deck(); rng.shuffle(deck)
    gs = GameState(draw_count=3); idx = 0
    truth = {}
    for n in range(1, 8):
        col = []
        for i in range(n):
            c = deck[idx+i]; up = (i == n-1); col.append((c, up))
        gs.tableau[n-1] = col; idx += n
    gs.stock = deck[idx:][::-1]
    for ci, col in enumerate(gs.tableau):
        for pi, (c, up) in enumerate(col):
            if not up:
                truth[(ci, pi)] = c
                col[pi] = (UNKNOWN, False)
    return gs, truth

def autoplay(seed, max_steps=800):
    st, truth = deal(seed)
    last_wp = None
    for step in range(max_steps):
        if st.is_won():
            return 'WON', step, last_wp
        p = mc_plan_steps(st, n=1, samples=60, solve_budget=40_000,
                          time_budget_s=3.0, settle=True, rng=random.Random(seed*100+step))
        last_wp = p.get('winProbability')
        if p['gameOver']:
            return ('STUCK' if p.get('stuck') else 'GAMEOVER'), step, last_wp
        if not p['steps']:
            return 'NOSTEPS', step, last_wp
        s = p['steps'][0]
        mv = Move(**s['move'])
        reveal_col = mv.src if flips_hidden_card(st, mv) else None
        st = apply_move(st, mv)
        if reveal_col is not None:
            col = st.tableau[reveal_col]; pos = len(col)-1
            real = truth.get((reveal_col, pos))
            if real is None:
                return 'REVEAL_MISS', step, last_wp
            col[pos] = (real, True)
    return 'MAXSTEPS', max_steps, last_wp

res = Counter()
for seed in range(12):
    outcome, steps, wp = autoplay(seed)
    res[outcome] += 1
    print(f"seed={seed:2d} -> {outcome:9s} after {steps} steps (last winProb={wp})")
print(dict(res))
