"""Flask app: Klondike solitaire solver with step-by-step calibration.

Supports mid-game positions with face-down (unknown) cards entered as "?".
Face-down cards are treated as OPAQUE (never guessed). A strategy planner
recommends the strongest moves using only known cards and the known stock,
three steps at a time, and asks the user to reveal a card's identity whenever a
move flips one. "Game over" means there are no legal moves at all.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from flask import Flask, jsonify, request, send_from_directory

from solver.cards import Card, is_unknown
from solver.engine import apply_move, legal_moves
from solver.game import GameState, Move
from solver.resolver import MidGameLayout, build_opaque_state, parse_midgame
from solver.strategy import flips_hidden_card, plan_steps
from solver import storage

app = Flask(__name__, static_folder="static", static_url_path="")

storage.init_db()


@dataclass
class Game:
    """A live game: the current opaque state (face-down cards are UNKNOWN, not
    guessed) plus the originating layout."""
    state: GameState
    layout: MidGameLayout


# In-memory store keyed by game id. Fine for a local single-user tool.
GAMES: dict[str, Game] = {}


# ---- serialization ----

def _cell(card, up) -> dict:
    """Serialize one tableau cell. Face-down -> hidden. Face-up unknown (a
    just-flipped '?') -> face-up but code '?' so the UI can prompt."""
    if not up:
        return {"code": None, "faceUp": False, "red": False, "unknown": True}
    if is_unknown(card):
        return {"code": "?", "faceUp": True, "red": False, "unknown": True}
    return {"code": card.code(), "faceUp": True, "red": card.is_red, "unknown": False}


def serialize_state(state: GameState) -> dict:
    # The current visible fan = the last `fan_size` cards of the waste. The
    # last one is the playable top. (If fan_size is 0 but the waste is
    # non-empty — e.g. a freshly loaded position — fall back to showing up to
    # the last 3 so the top is still visible.)
    fan_n = state.fan_size if state.fan_size else min(len(state.waste), state.draw_count)
    fan = state.waste[-fan_n:] if fan_n else []
    return {
        "tableau": [[_cell(c, up) for c, up in col] for col in state.tableau],
        "foundations": {
            suit: [c.code() for c in pile] for suit, pile in state.foundations.items()
        },
        "stockCount": len(state.stock),
        "waste": [{"code": c.code(), "red": c.is_red} for c in state.waste],
        "wasteTop": state.waste[-1].code() if state.waste else None,
        "wasteFan": [{"code": c.code(), "red": c.is_red} for c in fan],
        "won": state.is_won(),
        "faceDownCount": sum(1 for col in state.tableau for _, up in col if not up),
        "openUnknown": _open_unknown_column(state),
    }


def _open_unknown_column(state: GameState) -> int | None:
    """Index of a column whose top card is a face-up unknown awaiting reveal,
    or None. (At most one at a time in normal play.)"""
    for i, col in enumerate(state.tableau):
        if col and col[-1][1] and is_unknown(col[-1][0]):
            return i
    return None


def move_from_payload(data: dict) -> Move:
    return Move(
        kind=data["kind"],
        src=data.get("src"),
        dst=data.get("dst"),
        count=data.get("count", 1),
        card=data.get("card"),
    )


# ---- routes ----

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/new", methods=["POST"])
def new_game():
    """Create and solve a mid-game position.

    Expected JSON:
      {
        "columns":     [[...tokens...], x7],  # bottom-first; "?" = face-down
        "foundations": {"S": ["AS"], ...},    # current pile per suit
        "stock":       [...codes...],         # fixed cycling order, front first
        "waste":       [...codes...],         # optional, currently fanned
        "drawCount":   3
      }
    """
    data = request.get_json(force=True)
    columns = data.get("columns")
    foundations = data.get("foundations") or {}
    stock = data.get("stock", [])
    waste = data.get("waste") or []
    draw_count = int(data.get("drawCount", 3))

    # One id for both the in-memory game and its saved record.
    game_id = uuid.uuid4().hex

    def _save(solved, total_moves=None, assignment=None, error=None):
        try:
            storage.save_game(
                game_id,
                columns=columns if isinstance(columns, list) else [],
                foundations=foundations, stock=stock, waste=waste,
                draw_count=draw_count, solved=solved,
                total_moves=total_moves, assignment=assignment, error=error,
            )
        except Exception:  # persistence must never break the solve flow
            app.logger.exception("failed to save game %s", game_id)

    if not isinstance(columns, list) or len(columns) != 7:
        msg = "columns must be a list of 7 columns"
        _save(solved=-1, error=msg)
        return jsonify({"error": msg}), 400

    try:
        layout = parse_midgame(columns, stock, foundations=foundations,
                               waste=waste, draw_count=draw_count)
    except ValueError as exc:
        _save(solved=-1, error=str(exc))
        return jsonify({"error": str(exc)}), 400

    # Build the opaque state (face-down cards stay unknown, never guessed).
    state = build_opaque_state(layout)
    plan = plan_steps(state, n=3)

    # solved=2 here means "accepted & planning" (distinct from the old exact
    # solver's 1=won-proof). gameOver is only true when no legal moves exist.
    _save(solved=0 if plan["gameOver"] else 2)
    GAMES[game_id] = Game(state=state, layout=layout)
    return jsonify(_plan_response(game_id, state, plan))


def _plan_response(game_id, state, plan):
    stuck = plan.get("stuck", False)
    if plan["solved"]:
        message = "You won! All foundations complete."
    elif stuck:
        message = ("Stuck: the only move left is cycling the stock, and it never "
                   "exposes a playable card. Game over.")
    elif plan["gameOver"]:
        message = "No legal moves left. Game over."
    else:
        message = ""
    return {
        "gameId": game_id,
        "solved": plan["solved"],       # True only if already won
        "gameOver": plan["gameOver"],   # True if no moves or dead stock cycle
        "stuck": stuck,                 # dead-stock-cycle variant of game over
        "revealNext": plan["revealNext"],
        "steps": plan["steps"],
        "state": serialize_state(state),
        "message": message,
    }


@app.route("/api/steps/<game_id>", methods=["GET"])
def get_steps(game_id: str):
    game = GAMES.get(game_id)
    if game is None:
        return jsonify({"error": "unknown game"}), 404
    n = int(request.args.get("n", 3))
    plan = plan_steps(game.state, n=n)
    return jsonify(_plan_response(game_id, game.state, plan))


@app.route("/api/accept/<game_id>", methods=["POST"])
def accept_step(game_id: str):
    """Apply one accepted move. If it flips a face-down card, the response says
    so (revealNeeded) and the UI should prompt for the real card before the
    next plan is meaningful."""
    game = GAMES.get(game_id)
    if game is None:
        return jsonify({"error": "unknown game"}), 404
    data = request.get_json(force=True)
    move = move_from_payload(data["move"])

    if not _move_is_legal(game.state, move):
        return jsonify({"error": "move is not legal in current state"}), 400

    reveal_col = move.src if flips_hidden_card(game.state, move) else None
    game.state = apply_move(game.state, move)

    # If a hidden card was exposed, don't plan yet; ask for its identity.
    if reveal_col is not None:
        return jsonify({
            "state": serialize_state(game.state),
            "revealNeeded": {"col": reveal_col},
            "steps": [],
            "gameOver": False,
            "solved": False,
        })

    plan = plan_steps(game.state, n=3)
    resp = _plan_response(game_id, game.state, plan)
    resp["revealNeeded"] = None
    return jsonify(resp)


@app.route("/api/reveal/<game_id>", methods=["POST"])
def reveal_card(game_id: str):
    """Tell the solver the real identity of the face-down card just flipped.

    The flipped card is the (face-up, unknown) top of `col`. We pin it to the
    real card by swapping identities with wherever that card currently sits as
    an unknown placeholder, keeping the 52-card set consistent, then re-plan.
    """
    game = GAMES.get(game_id)
    if game is None:
        return jsonify({"error": "unknown game"}), 404
    data = request.get_json(force=True)
    col = int(data["col"])
    real_code = data["card"]

    try:
        real = Card.parse(real_code)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    column = game.state.tableau[col]
    if not column:
        return jsonify({"error": "column is empty"}), 400
    top_card, top_up = column[-1]
    if not top_up or not is_unknown(top_card):
        return jsonify({"error": "top card is not an unrevealed card"}), 400

    # Validate the revealed card against the set of cards still genuinely
    # unknown. That set is exactly: the full deck minus every KNOWN card
    # (face-up tableau, waste, foundations, stock). A valid reveal must be one
    # of those — this catches both duplicates of visible cards and typos that
    # name a card which is actually known to be elsewhere.
    remaining = _remaining_unknown_cards(game.state)
    if real not in remaining:
        if _card_known_elsewhere(game.state, real):
            msg = (f"{real.code()} is already visible elsewhere on the board, "
                   f"so it can't be the card you just flipped. Check the card "
                   f"and try again.")
        else:
            msg = (f"{real.code()} can't be here — it's not one of the "
                   f"face-down cards left to reveal. Check the card and try "
                   f"again.")
        return jsonify({"error": msg, "revealRetry": {"col": col}}), 400

    # Pin the revealed identity onto the top of the column.
    game.state.tableau[col][-1] = (real, True)

    plan = plan_steps(game.state, n=3)
    resp = _plan_response(game_id, game.state, plan)
    resp["revealNeeded"] = None
    return jsonify(resp)


@app.route("/api/reject/<game_id>", methods=["POST"])
def reject_step(game_id: str):
    """Reject the proposed next move; recommend the next-best line that starts
    with a different move."""
    game = GAMES.get(game_id)
    if game is None:
        return jsonify({"error": "unknown game"}), 404
    data = request.get_json(force=True)
    rejected = move_from_payload(data["move"])
    n = int(data.get("n", 3))

    moves = [m for m in legal_moves(game.state) if not _same_move(m, rejected)]
    if not moves:
        plan = {"solved": False, "gameOver": not game.state.is_won(),
                "steps": [], "revealNext": False}
        resp = _plan_response(game_id, game.state, plan)
        resp["message"] = "No alternative move available."
        return jsonify(resp)

    # Plan a line for each alternative first move; recommend the best-scoring.
    from solver.strategy import apply_move as _am  # same engine apply
    from solver.strategy import flips_hidden_card as _fh
    from solver.strategy import plan as _plan, state_score

    best_line = None
    best_val = float("-inf")
    for first in moves:
        if _fh(game.state, first):
            line = [first]
            val = state_score(game.state) + 12.0
        else:
            nxt = _am(game.state, first)
            rest = _plan(nxt, max_depth=7)
            line = [first] + rest
            val = state_score(nxt)
        if val > best_val:
            best_val = val
            best_line = line

    steps = []
    sim = game.state
    reveal_next = False
    for i, mv in enumerate(best_line[:n]):
        reveal = flips_hidden_card(sim, mv)
        steps.append({
            "index": i, "kind": mv.kind, "description": mv.describe(),
            "revealsHidden": reveal,
            "move": {"kind": mv.kind, "src": mv.src, "dst": mv.dst,
                     "count": mv.count, "card": mv.card},
        })
        if i == 0:
            reveal_next = reveal
        if reveal:
            break
        sim = apply_move(sim, mv)

    plan = {"solved": False, "gameOver": False, "steps": steps,
            "revealNext": reveal_next}
    return jsonify(_plan_response(game_id, game.state, plan))


@app.route("/api/games", methods=["GET"])
def list_saved_games():
    """List saved positions, newest first, for debugging."""
    limit = int(request.args.get("limit", 100))
    return jsonify({"games": storage.list_games(limit=limit)})


@app.route("/api/games/<game_id>", methods=["GET"])
def get_saved_game(game_id: str):
    """Fetch a saved position's full raw input + verdict."""
    rec = storage.get_game(game_id)
    if rec is None:
        return jsonify({"error": "unknown game"}), 404
    return jsonify(rec)


@app.route("/api/games/<game_id>/replay", methods=["POST"])
def replay_saved_game(game_id: str):
    """Re-load a saved position back into memory and re-solve it, so you can
    step through it again. Returns the same shape as /api/new."""
    rec = storage.get_game(game_id)
    if rec is None:
        return jsonify({"error": "unknown game"}), 404

    try:
        layout = parse_midgame(
            rec["columns"], rec["stock"], foundations=rec["foundations"],
            waste=rec["waste"], draw_count=rec["draw_count"],
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    state = build_opaque_state(layout)
    plan = plan_steps(state, n=3)
    GAMES[game_id] = Game(state=state, layout=layout)
    return jsonify(_plan_response(game_id, state, plan))


@app.route("/api/games/<game_id>/note", methods=["POST"])
def set_game_note(game_id: str):
    """Attach a free-form note to a saved position (e.g. 'KH reveal bug')."""
    data = request.get_json(force=True)
    note = str(data.get("note", ""))
    if not storage.set_note(game_id, note):
        return jsonify({"error": "unknown game"}), 404
    return jsonify({"ok": True})


# ---- helpers ----

def _card_known_elsewhere(state: GameState, card: Card) -> bool:
    """Is `card` already visible as a known card somewhere (tableau face-up,
    waste, or a foundation)? Face-down/unknown placeholders don't count."""
    for col in state.tableau:
        for c, up in col:
            if up and not is_unknown(c) and c == card:
                return True
    if any(c == card for c in state.waste):
        return True
    for pile in state.foundations.values():
        if card in pile:
            return True
    return False


def _remaining_unknown_cards(state: GameState) -> set[Card]:
    """The set of cards whose identity is still genuinely unknown: the full
    52-card deck minus every card with a KNOWN location (face-up tableau,
    waste, foundations, stock). Face-down/unknown tableau placeholders are, by
    definition, the remaining unknowns — so they are NOT subtracted."""
    from solver.cards import full_deck

    known: set[Card] = set()
    for col in state.tableau:
        for c, up in col:
            if not is_unknown(c):
                known.add(c)  # a concrete card has a known location
    known.update(state.waste)
    known.update(state.stock)
    for pile in state.foundations.values():
        known.update(pile)
    return {c for c in full_deck() if c not in known}


def _same_move(a: Move, b: Move) -> bool:
    return (a.kind == b.kind and a.src == b.src and a.dst == b.dst
            and a.count == b.count and a.card == b.card)


def _move_is_legal(state: GameState, move: Move) -> bool:
    return any(_same_move(move, m) for m in legal_moves(state))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
