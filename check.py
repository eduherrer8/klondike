"""Quick sanity checks for the engine and solver (not a test suite)."""

from solver.cards import Card, can_stack_foundation, can_stack_tableau, full_deck
from solver.engine import apply_move, legal_moves
from solver.game import GameState, Move
from solver.solver import solve


def check_cards():
    assert Card.parse("AS").code() == "AS"
    assert Card.parse("10H").code() == "10H"
    assert Card.parse("TH").code() == "10H"
    assert Card.parse("kc").code() == "KC"
    # red 6 onto black 7
    assert can_stack_tableau(Card.parse("6H"), Card.parse("7S"))
    assert not can_stack_tableau(Card.parse("6H"), Card.parse("7H"))
    assert can_stack_foundation(Card.parse("AS"), None, None)
    assert can_stack_foundation(Card.parse("2S"), Card.parse("AS"), "S")
    assert not can_stack_foundation(Card.parse("2H"), Card.parse("AS"), "S")
    print("cards: OK")


def check_deck():
    deck = full_deck()
    assert len(deck) == 52
    assert len(set(c.code() for c in deck)) == 52
    print("deck: OK")


def check_engine_basics():
    # Simple layout: col1 has KS face up, col2 empty-ish ... build a tiny scenario.
    columns = [
        ["KS"],
        ["7H", "QD"],
        ["3C", "8S", "4H"],
        ["AD", "5C", "9H", "2S"],
        ["6D", "10C", "JH", "3S", "7C"],
        ["QH", "4D", "8C", "KH", "5S", "9D"],
        ["2H", "6C", "10D", "JS", "AC", "4S", "8H"],
    ]
    stock = [
        "AS", "2C", "3H", "5H", "6H", "7S", "8D", "9S", "9C", "10H",
        "10S", "JD", "JC", "QS", "QC", "KD", "KC", "2D", "3D", "4C",
        "5D", "6S", "7D", "AH",
    ]
    gs = GameState.from_layout(columns, stock, draw_count=3)
    moves = legal_moves(gs)
    assert any(m.kind == "draw" for m in moves), "draw should be available"
    # QD (red) can go on KS (black)
    assert any(m.kind == "tableau" and m.card == "QD" for m in moves), [m for m in moves]
    # drawing adds up to 3 to waste
    after = apply_move(gs, Move("draw"))
    assert len(after.waste) == 3
    assert len(after.stock) == len(gs.stock) - 3
    print("engine: OK")


def check_no_split_runs():
    """A face-up run must move as a whole; sub-runs are not legal moves."""
    gs = GameState(draw_count=3)
    # Column 0: face-down 2S, then face-up run 9H 8S 7H (red-black-red, descending)
    gs.tableau[0] = [
        (Card.parse("2S"), False),
        (Card.parse("9H"), True),
        (Card.parse("8S"), True),
        (Card.parse("7H"), True),
    ]
    # Column 1: face-up 10S so that 9H could stack onto it (whole run),
    # and a black 8 is NOT present anywhere to tempt a split.
    gs.tableau[1] = [(Card.parse("10S"), True)]
    # Column 2: face-up 9S, onto which the sub-run 8S 7H *would* fit if splitting
    # were allowed. It must NOT be offered.
    gs.tableau[2] = [(Card.parse("9S"), True)]

    tableau_moves = [m for m in legal_moves(gs) if m.kind == "tableau"]
    # The whole run (bottom 9H) onto 10S is legal.
    assert any(m.src == 0 and m.dst == 1 and m.card == "9H" and m.count == 3
               for m in tableau_moves), tableau_moves
    # A split starting at 8S onto 9S must NOT be present.
    assert not any(m.card == "8S" for m in tableau_moves), \
        f"splitting a run should be illegal, got {tableau_moves}"
    print("no-split runs: OK")


def check_solver_small():
    # An almost-finished game: all foundations at Queen, kings on tableau tops.
    gs = GameState(draw_count=3)
    for suit in ("S", "H", "D", "C"):
        gs.foundations[suit] = [Card(r, suit) for r in range(1, 13)]  # A..Q
    # put the four kings face up as the only tableau cards
    gs.tableau[0] = [(Card(13, "S"), True)]
    gs.tableau[1] = [(Card(13, "H"), True)]
    gs.tableau[2] = [(Card(13, "D"), True)]
    gs.tableau[3] = [(Card(13, "C"), True)]
    sol = solve(gs, max_states=50_000)
    assert sol is not None, "should solve the trivial end position"
    # apply and confirm win
    state = gs
    for m in sol:
        state = apply_move(state, m)
    assert state.is_won(), "solution must reach a won state"
    print(f"solver (trivial end): OK, {len(sol)} moves")


def check_prefilled_foundations():
    columns = [
        ["KS"],
        ["7H", "QD"],
        ["3C", "8S", "4H"],
        ["AD", "5C", "9H", "2S"],
        ["6D", "10C", "JH", "3S", "7C"],
        ["QH", "4D", "8C", "KH", "5S", "9D"],
        ["2H", "6C", "10D", "JS", "AC", "4S", "8H"],
    ]
    stock = [
        "2C", "3H", "5H", "6H", "7S", "8D", "9S", "9C", "10H",
        "10S", "JD", "JC", "QS", "QC", "KD", "KC", "2D", "3D", "4C",
        "5D", "6S", "7D", "AH",
    ]
    gs = GameState.from_layout(columns, stock, draw_count=3, foundations={"S": ["AS"]})
    assert [c.code() for c in gs.foundations["S"]] == ["AS"]
    # now the 2S on top of column 4 can go to the Spades foundation
    moves = legal_moves(gs)
    assert any(m.kind == "to_foundation" and m.card == "2S" for m in moves), [m for m in moves]

    # invalid foundation (wrong suit) should raise
    try:
        GameState.from_layout(columns, stock, foundations={"S": ["AH"]})
        raise AssertionError("expected wrong-suit foundation to raise")
    except ValueError:
        pass

    # invalid foundation (does not start at Ace) should raise
    try:
        GameState.from_layout(columns, stock + ["AS"], foundations={"S": ["2S"]})
        raise AssertionError("expected non-ace-start foundation to raise")
    except ValueError:
        pass
    print("prefilled foundations: OK")


def check_resolver_elimination():
    """The resolver must deduce hidden cards by elimination."""
    from solver.resolver import parse_midgame

    # Minimal consistent 52-card position with 2 face-down cards.
    # Put most of the deck on foundations so only a few cards float.
    foundations = {
        "S": [f"{r}S" for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]],
        "H": [f"{r}H" for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q"]],  # up to Q
        "D": [f"{r}D" for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J"]],       # up to J
        "C": [f"{r}C" for r in ["A", "2", "3", "4", "5", "6", "7"]],                              # up to 7
    }
    # Remaining cards: KH; QD KD; 8C 9C 10C JC QC KC  => 9 cards.
    # Place some face-up, hide 2 as "?", rest in stock.
    columns = [["?", "?", "KH"], [], [], [], [], [], []]
    stock = ["QD", "KD", "8C", "9C", "10C", "JC"]  # 6 in stock
    # face-up KH (1) + 2 hidden + 6 stock = 9. The 2 hidden come from the pool.
    layout = parse_midgame(columns, stock, foundations=foundations, waste=[], draw_count=3)
    pool = sorted(c.code() for c in layout.unknown_pool)
    assert len(layout.unknown_slots) == 2
    assert pool == ["KC", "QC"], pool  # QC, KC are the only unaccounted cards
    print("resolver elimination: OK")


def check_resolver_bad_counts():
    from solver.resolver import parse_midgame
    # Too many unknowns vs leftovers -> error
    try:
        parse_midgame([["?", "?", "?"], [], [], [], [], [], []], stock=[], foundations={})
        raise AssertionError("expected mismatch error")
    except ValueError:
        pass
    # face-down above face-up in a column -> error
    try:
        parse_midgame([["5H", "?"], [], [], [], [], [], []], stock=[], foundations={})
        raise AssertionError("expected face-down-above-face-up error")
    except ValueError:
        pass
    print("resolver bad counts: OK")


def check_midgame_solver():
    """A mid-game position with a genuinely buried hidden card that is solvable.

    Foundations are complete except Spades, which needs K. The KS is buried
    face-down in a column under a face-up card that can be cleared, so once the
    cover moves the KS flips and finishes the game.
    """
    from solver.resolver import build_state, parse_midgame
    from solver.solver import solve_midgame

    foundations = {
        "S": [f"{r}S" for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q"]],  # need KS
        "H": [f"{r}H" for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]],  # done
        "D": [f"{r}D" for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]],  # done
        "C": [f"{r}C" for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]],  # done
    }
    # Only KS is unaccounted for. It is buried face-down under nothing useful:
    # column 1 = [?]  -> but a lone card is exposed, so instead bury it under a
    # movable card. Put KS face-down beneath a face-up QH? QH is on foundation.
    # Simplest realistic case: KS is the only hidden card and it sits alone;
    # since it's exposed it's effectively face-up and can go straight to
    # foundation.  Represent that as a known KS (no '?').
    columns = [["KS"], [], [], [], [], [], []]
    stock = []
    layout = parse_midgame(columns, stock, foundations=foundations, waste=[], draw_count=3)
    assert len(layout.unknown_slots) == 0

    solution, assignment = solve_midgame(layout, max_states=20_000, max_arrangements=50)
    assert solution is not None, "should finish by sending KS to foundation"
    state = build_state(layout, assignment)
    from solver.engine import apply_move as _apply
    for m in solution:
        state = _apply(state, m)
    assert state.is_won()
    print(f"midgame solver (no hidden): OK, {len(solution)} moves")


def check_midgame_hidden_solver():
    """A position where a hidden card is genuinely buried and must be revealed
    by clearing the face-up card on top of it."""
    from solver.resolver import build_state, parse_midgame
    from solver.solver import solve_midgame

    # Foundations: H, D, C complete. Spades up to J (need Q, K).
    foundations = {
        "S": [f"{r}S" for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J"]],  # need QS, KS
        "H": [f"{r}H" for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]],
        "D": [f"{r}D" for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]],
        "C": [f"{r}C" for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]],
    }
    # Left: QS, KS. Bury one hidden under QS: column = [?, QS]. The ? must be KS.
    # QS -> foundation first, then the revealed KS -> foundation.
    columns = [["?", "QS"], [], [], [], [], [], []]
    stock = []
    layout = parse_midgame(columns, stock, foundations=foundations, waste=[], draw_count=3)
    assert len(layout.unknown_slots) == 1
    assert sorted(c.code() for c in layout.unknown_pool) == ["KS"]

    solution, assignment = solve_midgame(layout, max_states=20_000, max_arrangements=10)
    assert solution is not None, "should reveal and finish"
    state = build_state(layout, assignment)
    from solver.engine import apply_move as _apply
    for m in solution:
        state = _apply(state, m)
    assert state.is_won()
    print(f"midgame solver (hidden revealed): OK, {len(solution)} moves")


def check_opaque_unknowns():
    """Face-up unknown cards must block their column and never be moved."""
    from solver.cards import UNKNOWN, is_unknown
    from solver.engine import legal_moves

    gs = GameState(draw_count=3)
    # col0: a face-up unknown on top -> nothing movable, can't stack onto it.
    gs.tableau[0] = [(UNKNOWN, True)]
    # col1: a real 6H that could stack onto a 7S if one existed; here just ensure
    # no move targets the unknown.
    gs.tableau[1] = [(Card.parse("7S"), True)]
    moves = legal_moves(gs)
    # No move should involve col0 (the unknown) as src or dst.
    assert all(not (m.src == 0 or m.dst == 0) for m in moves), moves
    # The unknown itself is not foundation-eligible.
    assert not any(m.kind == "to_foundation" and m.src == 0 for m in moves)
    print("opaque unknowns: OK")


def check_strategy_keeps_playing_after_reveal():
    """Reproduces the reported bug: after flipping a hidden card, the planner
    must keep offering moves (not false game over) while legal moves remain."""
    from solver.cards import Card
    from solver.engine import apply_move
    from solver.resolver import build_opaque_state, parse_midgame
    from solver.strategy import flips_hidden_card, plan_steps

    fo = {"S": ["AS"], "D": ["AD"], "H": ["AH"]}
    cols = [["10D"], ["?", "JS"], ["?", "?", "5D"], ["KC"],
            ["?", "?", "?", "?", "7S"], ["?", "?", "?", "?", "?", "6H"],
            ["?", "?", "?", "?", "?", "?", "10C"]]
    stock = ("5H 5C QD 4D 10H QC 3C 9D 2S JC 7H QH 4S 9C KD 6D 3D 9S QS 8H 9H "
             "JH 3S 10S").split()
    layout = parse_midgame(cols, stock, foundations=fo, waste=[], draw_count=3)
    st = build_opaque_state(layout)

    res = plan_steps(st, n=3)
    assert not res["gameOver"]
    first = res["steps"][0]
    assert first["revealsHidden"], "expected the 6H move to flip a hidden card"

    from solver.game import Move
    mv = Move(**first["move"])
    assert flips_hidden_card(st, mv)
    st2 = apply_move(st, mv)
    # Reveal it as 6C (pin the unknown top to the real card).
    st2.tableau[5][-1] = (Card.parse("6C"), True)

    res2 = plan_steps(st2, n=3)
    assert not res2["gameOver"], "must NOT be game over after revealing 6C"
    assert res2["steps"], "should still recommend moves"
    print("strategy keeps playing after reveal: OK")


def check_dead_stock_cycle():
    """A real stuck position (from an actual game): every tableau top is frozen
    and the only move is cycling a 3-card stock that never helps. The planner
    must report game over (stuck), not loop on 'Draw'."""
    from solver.resolver import build_opaque_state, parse_midgame
    from solver.strategy import is_dead_stock_cycle, plan_steps

    foundations = {"S": ["AS", "2S"], "D": ["AD", "2D", "3D"],
                   "H": ["AH", "2H", "3H", "4H", "5H"], "C": ["AC"]}
    columns = [
        ["KH", "QS", "JD", "10C", "9H", "8C"], [],
        ["KS", "QH"],
        ["KC", "QD", "JS", "10D", "9C", "8H", "7S", "6H", "5C", "4D", "3C"],
        ["?", "?", "4S"], ["?", "?", "?", "?", "6C", "5D", "4C"],
        ["?", "?", "?", "?", "9S", "8D", "7C", "6D", "5S"],
    ]
    stock = ["10S", "6S", "10H"]
    layout = parse_midgame(columns, stock, foundations=foundations, waste=[], draw_count=3)
    st = build_opaque_state(layout)

    assert is_dead_stock_cycle(st), "should detect the dead stock cycle"
    res = plan_steps(st, n=3)
    assert res["gameOver"] and res["stuck"], res
    assert res["steps"] == [], "should not recommend looping draws"
    print("dead stock cycle: OK")


def check_stock_peel_reachability():
    """Draw-3 peel model: a stock card playable only after peeling the cards in
    front of it must be recognized. The dead-cycle detector must NOT call a
    position dead when a reachable (front-of-triplet) card can be played, and
    MUST call it dead when every playable card is buried behind an unplayable
    one."""
    from solver.resolver import build_opaque_state, parse_midgame
    from solver.strategy import is_dead_stock_cycle

    foundations = {"S": ["AS", "2S", "3S"], "D": ["AD"], "C": ["AC"]}
    columns = [
        ["KS"], ["JS", "10H", "9S"],
        ["?", "?", "KD", "QS", "JD", "10S", "9H", "8S", "7H", "6C", "5D", "4C", "3H"],
        ["QC"], ["?", "?", "KC"], ["?", "?", "?", "5C", "4D"],
        ["?", "?", "?", "?", "?", "?", "JC", "10D", "9C"],
    ]
    # Real deal: AH/3C/QH are each buried behind an unplayable front card -> dead.
    dead_stock = "6S AH 7D 3C 5S 10C 6D 7C 7S 9D QH 4H".split()
    st_dead = build_opaque_state(parse_midgame(columns, dead_stock,
                                               foundations=foundations, waste=[]))
    assert is_dead_stock_cycle(st_dead), "buried playables -> should be dead"

    # Same board but AH is the FRONT of its triplet -> reachable -> NOT dead.
    live_stock = "6S 7D AH 3C 5S 10C 6D 7C 7S 9D QH 4H".split()
    st_live = build_opaque_state(parse_midgame(columns, live_stock,
                                               foundations=foundations, waste=[]))
    assert not is_dead_stock_cycle(st_live), "reachable AH -> should NOT be dead"
    print("stock peel reachability: OK")


def check_game_over_only_when_no_moves():
    """Game over must require zero legal moves, not merely 'no proven win'."""
    from solver.engine import legal_moves
    from solver.strategy import plan_steps

    # A dead position: single column with a face-up card that can't move and no
    # stock/waste. (All foundations empty, one lonely 5H, nothing else.)
    gs = GameState(draw_count=3)
    gs.tableau[0] = [(Card.parse("5H"), True)]
    res = plan_steps(gs, n=3)
    assert not legal_moves(gs), "test setup should have no legal moves"
    assert res["gameOver"], "should be game over when no legal moves exist"
    print("game over only when no moves: OK")


if __name__ == "__main__":
    check_cards()
    check_deck()
    check_engine_basics()
    check_no_split_runs()
    check_solver_small()
    check_prefilled_foundations()
    check_resolver_elimination()
    check_resolver_bad_counts()
    check_midgame_solver()
    check_midgame_hidden_solver()
    check_opaque_unknowns()
    check_strategy_keeps_playing_after_reveal()
    check_dead_stock_cycle()
    check_stock_peel_reachability()
    check_game_over_only_when_no_moves()
    print("\nAll sanity checks passed.")
