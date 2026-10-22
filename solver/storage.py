"""SQLite persistence for submitted game positions.

Every position sent to /api/new is saved here so it can be replayed and
debugged later. We store the raw input exactly as submitted (columns,
foundations, stock, waste, draw_count) plus the solver's verdict.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "games.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id            TEXT PRIMARY KEY,
    created_at    REAL NOT NULL,
    created_iso   TEXT NOT NULL,
    columns_json  TEXT NOT NULL,
    foundations_json TEXT NOT NULL,
    stock_json    TEXT NOT NULL,
    waste_json    TEXT NOT NULL,
    draw_count    INTEGER NOT NULL,
    solved        INTEGER NOT NULL,   -- 1 solved, 0 game over, -1 input error
    total_moves   INTEGER,            -- NULL unless solved
    assignment_json TEXT,             -- hidden-card guesses, NULL unless solved
    error         TEXT,               -- parse/validation error, if any
    note          TEXT                -- free-form label you can set later
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def save_game(
    game_id: str,
    *,
    columns: list,
    foundations: dict,
    stock: list,
    waste: list,
    draw_count: int,
    solved: int,
    total_moves: int | None = None,
    assignment: dict | None = None,
    error: str | None = None,
) -> None:
    """Insert (or replace) a submitted position."""
    now = time.time()
    created_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO games (
                id, created_at, created_iso, columns_json, foundations_json,
                stock_json, waste_json, draw_count, solved, total_moves,
                assignment_json, error, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                COALESCE((SELECT note FROM games WHERE id = ?), NULL))
            """,
            (
                game_id, now, created_iso,
                json.dumps(columns), json.dumps(foundations),
                json.dumps(stock), json.dumps(waste), draw_count,
                solved, total_moves,
                json.dumps(assignment) if assignment is not None else None,
                error, game_id,
            ),
        )


def list_games(limit: int = 100) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, created_iso, draw_count, solved, total_moves, error, note
            FROM games ORDER BY created_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_game(game_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    # Decode the JSON columns for convenience.
    d["columns"] = json.loads(d.pop("columns_json"))
    d["foundations"] = json.loads(d.pop("foundations_json"))
    d["stock"] = json.loads(d.pop("stock_json"))
    d["waste"] = json.loads(d.pop("waste_json"))
    if d.get("assignment_json"):
        d["assignment"] = json.loads(d.pop("assignment_json"))
    else:
        d.pop("assignment_json", None)
        d["assignment"] = None
    return d


def set_note(game_id: str, note: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("UPDATE games SET note = ? WHERE id = ?", (note, game_id))
    return cur.rowcount > 0
