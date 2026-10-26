"""Local on-device database (SQLite).

Everything the patient does is written here first, so the app works offline and feels
instant. Writes that the clinic needs go into an outbox table; the mock service bus
(servicebus.py) delivers them later, which is where the client's ~2 minute delay lives.

Every record keeps a timestamp, the input method and the original words (story 7).
"""

from __future__ import annotations

import json
import random
import sqlite3
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY,
    metric TEXT NOT NULL,
    systolic INTEGER, diastolic INTEGER,
    value REAL, unit TEXT NOT NULL,
    canonical_value REAL, canonical_unit TEXT,
    measured_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    confirmed_at TEXT,
    input_method TEXT NOT NULL,
    source_text TEXT,
    lang TEXT,
    sync_status TEXT NOT NULL DEFAULT 'pending_sync'
);
CREATE TABLE IF NOT EXISTS slots (
    id INTEGER PRIMARY KEY,
    provider TEXT NOT NULL,
    start_at TEXT NOT NULL,
    booked INTEGER NOT NULL DEFAULT 0,
    UNIQUE(provider, start_at)
);
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY,
    slot_id INTEGER NOT NULL,
    provider TEXT NOT NULL,
    start_at TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    confirmed_by_patient_at TEXT,
    synced_at TEXT,
    input_method TEXT NOT NULL,
    source_text TEXT,
    lang TEXT
);
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    ref_id INTEGER NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    deliver_after TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    delivered_at TEXT
);
CREATE TABLE IF NOT EXISTS plan (
    lang TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    content TEXT NOT NULL,
    synced_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY,
    at TEXT NOT NULL,
    lang TEXT,
    input_method TEXT,
    transcript TEXT,
    route TEXT,
    reply TEXT,
    timings TEXT
);
"""

PROVIDERS = ["patel", "ruiz"]


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    # --- helpers --------------------------------------------------------------
    def _all(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def _one(self, sql: str, args=()) -> dict | None:
        with self.lock:
            r = self.conn.execute(sql, args).fetchone()
            return dict(r) if r else None

    def _exec(self, sql: str, args=()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur.lastrowid

    # --- seeding --------------------------------------------------------------
    def seed(self, now: datetime, plan_dir: Path, force: bool = False) -> None:
        """Demo data: 30 days of readings, four weeks of clinic slots, the plan."""
        if force:
            with self.lock:
                for table in ("measurements", "slots", "appointments", "outbox", "plan", "turns"):
                    self.conn.execute(f"DELETE FROM {table}")
                self.conn.commit()
        if not self._one("SELECT id FROM measurements LIMIT 1"):
            self._seed_measurements(now)
        latest = self._one("SELECT MAX(start_at) AS m FROM slots")
        if not latest or not latest["m"] or latest["m"] < _iso(now + timedelta(days=14)):
            self._seed_slots(now)
        for lang in ("en", "es"):
            if not self._one("SELECT lang FROM plan WHERE lang=?", (lang,)):
                content = (plan_dir / f"plan_{lang}.md").read_text(encoding="utf-8")
                self._exec("INSERT INTO plan(lang, version, content, synced_at) VALUES (?,?,?,?)",
                           (lang, 1, content, _iso(now)))

    def _seed_measurements(self, now: datetime) -> None:
        rng = random.Random(7)
        rows = []
        for d in range(30, 0, -1):
            day = (now - timedelta(days=d)).replace(hour=7, minute=40, second=0, microsecond=0)
            sys_ = int(rng.gauss(138 - (30 - d) * 0.15, 5))
            dia = int(rng.gauss(87 - (30 - d) * 0.08, 3))
            rows.append(("blood_pressure", sys_, dia, None, "mmHg", None, "mmHg", day))
            rows.append(("heart_rate", None, None, float(int(rng.gauss(74, 4))), "bpm", None, "bpm",
                         day + timedelta(minutes=1)))
            if day.weekday() in (0, 2, 4):
                g = float(int(rng.gauss(116, 7)))
                rows.append(("glucose", None, None, g, "mg/dL", g, "mg/dL", day + timedelta(minutes=5)))
            if day.weekday() == 0 or d % 7 == 3:
                lb = round(184 - (30 - d) * 0.17 + rng.gauss(0, 0.6), 1)
                rows.append(("weight", None, None, lb, "lb", round(lb / 2.20462, 2), "kg",
                             day + timedelta(minutes=10)))
        with self.lock:
            for metric, s, di, v, unit, cv, cu, at in rows:
                self.conn.execute(
                    """INSERT INTO measurements(metric, systolic, diastolic, value, unit,
                       canonical_value, canonical_unit, measured_at, recorded_at, confirmed_at,
                       input_method, source_text, lang, sync_status)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (metric, s, di, v, unit, cv if cv is not None else v, cu,
                     _iso(at), _iso(at), _iso(at), "device_import", None, "en", "synced"))
            self.conn.commit()

    def _seed_slots(self, now: datetime) -> None:
        """Weekday slots 9:00-16:30, about 60% already booked.

        The first upcoming Tuesday afternoon is fully booked so the "next closest
        options" behavior from story 2 shows up in the demo.
        """
        today = now.date()
        first_tuesday = today + timedelta(days=((1 - today.weekday()) % 7) or 7)
        with self.lock:
            for d in range(0, 29):
                day = today + timedelta(days=d)
                if day.weekday() >= 5:
                    continue
                for pi, provider in enumerate(PROVIDERS):
                    for minutes in range(9 * 60, 17 * 60, 30):
                        start = datetime.combine(day, datetime.min.time()) + timedelta(minutes=minutes)
                        rng = random.Random(day.toordinal() * 1000 + minutes * 3 + pi)
                        booked = rng.random() < 0.6
                        if day == first_tuesday and minutes >= 12 * 60:
                            booked = True
                        self.conn.execute(
                            "INSERT OR IGNORE INTO slots(provider, start_at, booked) VALUES (?,?,?)",
                            (provider, _iso(start), int(booked)))
            self.conn.commit()

    # --- measurements ---------------------------------------------------------
    def add_measurement(self, reading, input_method: str, source_text: str, lang: str,
                        now: datetime) -> int:
        return self._exec(
            """INSERT INTO measurements(metric, systolic, diastolic, value, unit,
               canonical_value, canonical_unit, measured_at, recorded_at, confirmed_at,
               input_method, source_text, lang, sync_status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?, 'pending_sync')""",
            (reading.metric, reading.systolic, reading.diastolic, reading.value, reading.unit,
             reading.canonical_value,
             {"blood_pressure": "mmHg", "weight": "kg", "glucose": "mg/dL", "heart_rate": "bpm"}[reading.metric],
             _iso(reading.measured_at or now), _iso(now), _iso(now), input_method, source_text, lang))

    def measurements(self, metric: str | None = None, since: datetime | None = None,
                     until: datetime | None = None) -> list[dict]:
        sql, args = "SELECT * FROM measurements WHERE 1=1", []
        if metric:
            sql += " AND metric=?"
            args.append(metric)
        if since:
            sql += " AND measured_at>=?"
            args.append(_iso(since))
        if until:
            sql += " AND measured_at<?"
            args.append(_iso(until))
        return self._all(sql + " ORDER BY measured_at", args)

    # --- slots and appointments ----------------------------------------------
    def free_slots(self, start: datetime, end: datetime, provider: str | None = None) -> list[dict]:
        sql = "SELECT * FROM slots WHERE booked=0 AND start_at>=? AND start_at<?"
        args: list = [_iso(start), _iso(end)]
        if provider:
            sql += " AND provider=?"
            args.append(provider)
        return self._all(sql + " ORDER BY start_at, provider", args)

    def slot(self, slot_id: int) -> dict | None:
        return self._one("SELECT * FROM slots WHERE id=?", (slot_id,))

    def book(self, slot_id: int, input_method: str, source_text: str, lang: str,
             now: datetime) -> int | None:
        """Books locally. Returns the appointment id, or None if the slot is gone."""
        with self.lock:
            s = self.conn.execute("SELECT * FROM slots WHERE id=?", (slot_id,)).fetchone()
            if not s or s["booked"]:
                return None
            self.conn.execute("UPDATE slots SET booked=1 WHERE id=?", (slot_id,))
            cur = self.conn.execute(
                """INSERT INTO appointments(slot_id, provider, start_at, status, created_at,
                   confirmed_by_patient_at, input_method, source_text, lang)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (slot_id, s["provider"], s["start_at"], "pending_sync", _iso(now), _iso(now),
                 input_method, source_text, lang))
            self.conn.commit()
            return cur.lastrowid

    def appointments(self, upcoming_from: datetime | None = None) -> list[dict]:
        if upcoming_from:
            return self._all(
                "SELECT * FROM appointments WHERE start_at>=? AND status!='conflict' ORDER BY start_at",
                (_iso(upcoming_from),))
        return self._all("SELECT * FROM appointments ORDER BY start_at")

    def set_appointment_status(self, appt_id: int, status: str, now: datetime) -> None:
        self._exec("UPDATE appointments SET status=?, synced_at=? WHERE id=?", (status, _iso(now), appt_id))
        if status == "conflict":
            self._exec("""UPDATE slots SET booked=1 WHERE id=(SELECT slot_id FROM appointments WHERE id=?)""",
                       (appt_id,))

    def set_measurement_synced(self, m_id: int) -> None:
        self._exec("UPDATE measurements SET sync_status='synced' WHERE id=?", (m_id,))

    # --- outbox ---------------------------------------------------------------
    def enqueue(self, kind: str, ref_id: int, payload: dict, now: datetime, delay_s: float) -> int:
        return self._exec(
            "INSERT INTO outbox(kind, ref_id, payload, created_at, deliver_after) VALUES (?,?,?,?,?)",
            (kind, ref_id, json.dumps(payload), _iso(now), _iso(now + timedelta(seconds=delay_s))))

    def due_outbox(self, now: datetime) -> list[dict]:
        return self._all("SELECT * FROM outbox WHERE status='queued' AND deliver_after<=? ORDER BY id",
                         (_iso(now),))

    def outbox(self) -> list[dict]:
        return self._all("SELECT * FROM outbox ORDER BY id DESC LIMIT 50")

    def mark_delivered(self, item_id: int, status: str, now: datetime) -> None:
        self._exec("UPDATE outbox SET status=?, delivered_at=? WHERE id=?", (status, _iso(now), item_id))

    # --- plan -----------------------------------------------------------------
    def plan(self, lang: str) -> dict | None:
        return self._one("SELECT * FROM plan WHERE lang=?", (lang,))

    def update_plan(self, lang: str, content: str, now: datetime) -> int:
        """Only the sync job calls this. The assistant has no path to change the plan."""
        current = self.plan(lang)
        if current and current["content"] == content:
            self._exec("UPDATE plan SET synced_at=? WHERE lang=?", (_iso(now), lang))
            return current["version"]
        version = (current["version"] + 1) if current else 1
        self._exec("INSERT OR REPLACE INTO plan(lang, version, content, synced_at) VALUES (?,?,?,?)",
                   (lang, version, content, _iso(now)))
        return version

    # --- latency log ----------------------------------------------------------
    def log_turn(self, lang, input_method, transcript, route, reply, timings, now) -> int:
        return self._exec("""INSERT INTO turns(at, lang, input_method, transcript, route, reply, timings)
                             VALUES (?,?,?,?,?,?,?)""",
                          (_iso(now), lang, input_method, transcript, route, reply, json.dumps(timings)))

    def add_turn_timing(self, turn_id: int, key: str, value: int) -> None:
        with self.lock:
            row = self.conn.execute("SELECT timings FROM turns WHERE id=?", (turn_id,)).fetchone()
            if not row:
                return
            timings = json.loads(row["timings"])
            timings[key] = value
            self.conn.execute("UPDATE turns SET timings=? WHERE id=?", (json.dumps(timings), turn_id))
            self.conn.commit()

    def turns(self) -> list[dict]:
        return self._all("SELECT * FROM turns ORDER BY id")
