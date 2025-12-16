from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any


def _pick_first_present(cols: set[str], names: list[str]) -> str | None:
    for n in names:
        if n in cols:
            return n
    return None


def main(argv: list[str]) -> int:
    db_path = Path(argv[1]) if len(argv) > 1 else Path("home-assistant_v2.db")
    entity_id = argv[2] if len(argv) > 2 else "sensor.pstryk_home_forward_active_energy_total_cost"
    limit = int(argv[3]) if len(argv) > 3 else 20

    t0 = time.time()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=0.2)
    conn.row_factory = sqlite3.Row
    print("db:", db_path.resolve())
    print("entity:", entity_id)
    print("opened in", round(time.time() - t0, 3), "s")

    states_cols_rows = conn.execute("PRAGMA table_info(states)").fetchall()
    states_cols = {r["name"] for r in states_cols_rows}
    print("states columns:", sorted(states_cols))

    entity_filter_sql: str
    entity_params: tuple[Any, ...]

    # Prefer filtering by metadata_id (typically indexed) to avoid scanning huge states.
    has_states_meta = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='states_meta'"
    ).fetchone() is not None

    if "metadata_id" in states_cols and has_states_meta:
        meta_cols_rows = conn.execute("PRAGMA table_info(states_meta)").fetchall()
        meta_cols = {r["name"] for r in meta_cols_rows}
        print("states_meta columns:", sorted(meta_cols))
        if not {"metadata_id", "entity_id"} <= meta_cols:
            raise SystemExit("Unsupported states_meta schema")

        # Fast path: resolve metadata_id once (indexed) then query states by metadata_id.
        mid_row = conn.execute(
            "SELECT metadata_id FROM states_meta WHERE entity_id = ? LIMIT 1",
            (entity_id,),
        ).fetchone()
        if mid_row is None:
            print("No states_meta row for entity; nothing to inspect")
            return 0
        metadata_id = mid_row["metadata_id"]
        print("metadata_id:", metadata_id)
        entity_filter_sql = "s.metadata_id = ?"
        entity_params = (metadata_id,)
    elif "entity_id" in states_cols:
        entity_filter_sql = "s.entity_id = ?"
        entity_params = (entity_id,)
    else:
        raise SystemExit("Unsupported states schema: no entity_id or states_meta mapping")

    ts_col = _pick_first_present(states_cols, ["last_updated_ts", "last_changed_ts", "last_updated", "last_changed"])
    print("ts_col:", ts_col)

    payload_mode: str
    sa_pk: str | None = None
    sa_payload_col: str | None = None
    has_state_attributes = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='state_attributes'"
    ).fetchone() is not None

    if "attributes_id" in states_cols and has_state_attributes:
        sa_cols_rows = conn.execute("PRAGMA table_info(state_attributes)").fetchall()
        sa_cols = {r["name"] for r in sa_cols_rows}
        sa_pk = _pick_first_present(sa_cols, ["attributes_id", "id"])
        sa_payload_col = _pick_first_present(sa_cols, ["shared_attrs", "attributes"])
        print("state_attributes pk:", sa_pk, "payload:", sa_payload_col)
        if not (sa_pk and sa_payload_col):
            raise SystemExit("Unsupported state_attributes schema")
        payload_mode = "hybrid"
    elif "attributes" in states_cols:
        payload_mode = "states.attributes"
    elif "shared_attrs" in states_cols:
        payload_mode = "states.shared_attrs"
    else:
        raise SystemExit("Cannot inspect last_reset: no usable attributes columns")

    print("payload_mode:", payload_mode)
    if not ts_col:
        raise SystemExit("Cannot inspect last_reset: missing timestamp column")

    # Fetch only a small sample of matching states first (fast on large DBs).
    select_bits = [f"s.{ts_col} AS ts", "s.state AS state"]
    if payload_mode in {"hybrid", "states.attributes"} and "attributes" in states_cols:
        select_bits.append("s.attributes AS payload")
    elif payload_mode == "states.shared_attrs" and "shared_attrs" in states_cols:
        select_bits.append("s.shared_attrs AS payload")
    else:
        select_bits.append("s.attributes_id AS attributes_id")

    if payload_mode == "hybrid":
        select_bits.append("s.attributes_id AS attributes_id")

    q = (
        f"SELECT {', '.join(select_bits)} FROM states s "
        f"WHERE {entity_filter_sql} LIMIT {limit}"
    )

    q_start = time.time()
    conn.set_progress_handler(lambda: 1 if (time.time() - q_start) > 12.0 else 0, 20000)
    try:
        rows = conn.execute(q, entity_params).fetchall()
    except sqlite3.OperationalError as exc:  # noqa: BLE001
        raise SystemExit(f"Query failed/interrupted: {exc}")

    def lookup_payload(attributes_id: Any) -> Any:
        if attributes_id is None:
            return None
        try:
            row = conn.execute(
                f"SELECT {sa_payload_col} AS payload FROM state_attributes WHERE {sa_pk} = ?",
                (attributes_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        return None if row is None else row["payload"]

    def decode_payload(payload: Any) -> str | None:
        if payload is None:
            return None
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        if isinstance(payload, (bytes, bytearray)):
            try:
                payload = payload.decode("utf-8")
            except Exception:  # noqa: BLE001
                return None
        if not isinstance(payload, str):
            return None
        s = payload.strip()
        return s if s else None

    def extract_last_reset(payload_text: str | None) -> tuple[Any, str | None]:
        if not payload_text:
            return None, None
        try:
            obj = json.loads(payload_text)
        except Exception as exc:  # noqa: BLE001
            return f"<json error: {type(exc).__name__}>", None
        if not isinstance(obj, dict):
            return f"<json type: {type(obj).__name__}>", None
        lr = obj.get("last_reset")
        return lr, type(lr).__name__ if lr is not None else None

    print("\nlatest rows:")
    for r in rows:
        raw_payload: Any
        if payload_mode == "states.attributes":
            raw_payload = r["payload"]
        elif payload_mode == "states.shared_attrs":
            raw_payload = r["payload"]
        else:
            # Hybrid: use states.attributes if present/non-null, else fall back to state_attributes.
            raw_payload = r["payload"] if ("payload" in r.keys() and r["payload"] is not None) else lookup_payload(r["attributes_id"])

        payload_text = decode_payload(raw_payload)
        lr, lr_type = extract_last_reset(payload_text)
        prefix = (payload_text[:200] + "…") if isinstance(payload_text, str) and len(payload_text) > 200 else payload_text
        print(
            {
                "ts": r["ts"],
                "state": r["state"],
                "attributes_id": r["attributes_id"] if "attributes_id" in r.keys() else None,
                "last_reset": lr,
                "last_reset_type": lr_type,
                "payload_prefix": prefix,
            }
        )

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
