"""SQLite persistence for personalised voice commands.

Only validated, user-confirmed mappings are stored. The action is stored as an intent name
plus validated parameters (never free text, never code) and is re-validated on every load.
Raw audio is never stored. Embeddings are cached per phrase together with the encoder
version that produced them and are recomputed when the encoder changes.
"""
from __future__ import annotations

import json
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from ..nlu.taxonomy import TAXONOMY_VERSION
from ..schemas_v2 import VehicleAction

_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS personal_commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    trigger TEXT NOT NULL,
    trigger_norm TEXT NOT NULL,
    intent TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    user_confirmed INTEGER NOT NULL CHECK (user_confirmed IN (0, 1)),
    taxonomy_version TEXT NOT NULL,
    embedding_model TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (profile_id, trigger_norm)
);
CREATE TABLE IF NOT EXISTS command_phrases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    command_id INTEGER NOT NULL REFERENCES personal_commands(id) ON DELETE CASCADE,
    phrase TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('trigger', 'alias')),
    embedding BLOB,
    embedding_model TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    command_id INTEGER NOT NULL,
    old_json TEXT NOT NULL,
    new_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def normalize_phrase(text: str) -> str:
    t = unicodedata.normalize("NFC", text).casefold()
    t = "".join(ch if (ch.isalnum() or ch.isspace() or unicodedata.category(ch).startswith("M")) else " " for ch in t)
    return " ".join(t.split())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class PersonalCommand:
    id: int
    profile_id: str
    trigger: str
    action: VehicleAction
    user_confirmed: bool
    created_at: str
    updated_at: str
    embedding_model: str | None
    phrases: list[str]


class PersonalCommandStore:
    def __init__(self, path: str | Path, store_correction_history: bool = False):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(_SCHEMA)
        self.store_correction_history = store_correction_history

    def close(self) -> None:
        self.db.close()

    # ---- writes -----------------------------------------------------------
    def add_command(self, profile_id: str, trigger: str, action: VehicleAction, embedding: np.ndarray | None,
                    embedding_model: str | None) -> int:
        VehicleAction.model_validate(action.model_dump())  # re-validate before persisting
        now = _now()
        with self.db:
            cur = self.db.execute(
                "INSERT INTO personal_commands (profile_id, trigger, trigger_norm, intent, parameters_json, user_confirmed,"
                " taxonomy_version, embedding_model, created_at, updated_at) VALUES (?,?,?,?,?,1,?,?,?,?)",
                (profile_id, trigger.strip(), normalize_phrase(trigger), action.intent.value,
                 action.parameters.model_dump_json(exclude_none=True), TAXONOMY_VERSION, embedding_model, now, now))
            cid = cur.lastrowid
            self._insert_phrase(cid, trigger.strip(), "trigger", embedding, embedding_model)
        return cid

    def add_alias(self, profile_id: str, command_id: int, phrase: str, embedding: np.ndarray | None,
                  embedding_model: str | None) -> None:
        self._require(profile_id, command_id)
        with self.db:
            self._insert_phrase(command_id, phrase.strip(), "alias", embedding, embedding_model)
            self.db.execute("UPDATE personal_commands SET updated_at=? WHERE id=?", (_now(), command_id))

    def update_action(self, profile_id: str, command_id: int, action: VehicleAction) -> None:
        old = self._require(profile_id, command_id)
        VehicleAction.model_validate(action.model_dump())
        with self.db:
            self.db.execute("UPDATE personal_commands SET intent=?, parameters_json=?, updated_at=? WHERE id=?",
                            (action.intent.value, action.parameters.model_dump_json(exclude_none=True), _now(), command_id))
            if self.store_correction_history:  # opt-in audit trail (user consent setting)
                self.db.execute("INSERT INTO corrections (profile_id, command_id, old_json, new_json, created_at) VALUES (?,?,?,?,?)",
                                (profile_id, command_id, old.action.model_dump_json(), action.model_dump_json(), _now()))

    def delete_command(self, profile_id: str, command_id: int) -> bool:
        with self.db:
            cur = self.db.execute("DELETE FROM personal_commands WHERE id=? AND profile_id=?", (command_id, profile_id))
        return cur.rowcount == 1

    def _insert_phrase(self, cid, phrase, kind, emb, model):
        blob = np.asarray(emb, np.float32).tobytes() if emb is not None else None
        self.db.execute("INSERT INTO command_phrases (command_id, phrase, kind, embedding, embedding_model, created_at)"
                        " VALUES (?,?,?,?,?,?)", (cid, phrase, kind, blob, model, _now()))

    # ---- reads ------------------------------------------------------------
    def _require(self, profile_id: str, command_id: int) -> PersonalCommand:
        c = self.get(profile_id, command_id)
        if c is None:
            raise KeyError(f"no personal command {command_id} for profile {profile_id}")
        return c

    def _row_to_command(self, row) -> PersonalCommand | None:
        cid, profile, trigger, intent, params, confirmed, created, updated, model = row
        try:
            action = VehicleAction(intent=intent, parameters=json.loads(params))
        except (ValidationError, ValueError):
            return None  # a stored mapping that no longer validates is never executed
        phrases = [p for (p,) in self.db.execute("SELECT phrase FROM command_phrases WHERE command_id=? ORDER BY id", (cid,))]
        return PersonalCommand(cid, profile, trigger, action, bool(confirmed), created, updated, model, phrases)

    _COLS = "id, profile_id, trigger, intent, parameters_json, user_confirmed, created_at, updated_at, embedding_model"

    def get(self, profile_id: str, command_id: int) -> PersonalCommand | None:
        row = self.db.execute(f"SELECT {self._COLS} FROM personal_commands WHERE id=? AND profile_id=?",
                              (command_id, profile_id)).fetchone()
        return self._row_to_command(row) if row else None

    def find_by_trigger(self, profile_id: str, phrase: str) -> PersonalCommand | None:
        norm = normalize_phrase(phrase)
        row = self.db.execute(f"SELECT {self._COLS} FROM personal_commands WHERE profile_id=? AND trigger_norm=?",
                              (profile_id, norm)).fetchone()
        if row:
            return self._row_to_command(row)
        for c in self.list_commands(profile_id):  # aliases
            if norm in {normalize_phrase(p) for p in c.phrases}:
                return c
        return None

    def list_commands(self, profile_id: str) -> list[PersonalCommand]:
        rows = self.db.execute(f"SELECT {self._COLS} FROM personal_commands WHERE profile_id=? AND user_confirmed=1 ORDER BY id",
                               (profile_id,)).fetchall()
        return [c for c in map(self._row_to_command, rows) if c is not None]

    def phrase_index(self, profile_id: str, embedding_model: str, embed_fn) -> tuple[list[tuple[int, str]], np.ndarray]:
        """(command_id, phrase) pairs and their embeddings; stale/missing embeddings are recomputed."""
        valid = {c.id for c in self.list_commands(profile_id)}
        rows = self.db.execute(
            "SELECT p.id, p.command_id, p.phrase, p.embedding, p.embedding_model FROM command_phrases p"
            " JOIN personal_commands c ON c.id = p.command_id WHERE c.profile_id=? ORDER BY p.id", (profile_id,)).fetchall()
        rows = [r for r in rows if r[1] in valid]
        stale = [r for r in rows if r[3] is None or r[4] != embedding_model]
        if stale:
            new = embed_fn([r[2] for r in stale])
            with self.db:
                for r, e in zip(stale, new):
                    self.db.execute("UPDATE command_phrases SET embedding=?, embedding_model=? WHERE id=?",
                                    (np.asarray(e, np.float32).tobytes(), embedding_model, r[0]))
            fresh = {r[0]: e for r, e in zip(stale, new)}
        else:
            fresh = {}
        pairs = [(r[1], r[2]) for r in rows]
        embs = [fresh[r[0]] if r[0] in fresh else np.frombuffer(r[3], np.float32) for r in rows]
        return pairs, (np.stack(embs) if embs else np.zeros((0, 1), np.float32))

    def correction_count(self, profile_id: str) -> int:
        return self.db.execute("SELECT COUNT(*) FROM corrections WHERE profile_id=?", (profile_id,)).fetchone()[0]
