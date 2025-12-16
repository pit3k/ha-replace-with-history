from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class DbSummaryError(RuntimeError):
    pass


@dataclass(frozen=True)
class OccurrenceRun:
    start_ts: str
    end_ts: str
    count: int
    value: str


@dataclass(frozen=True)
class RowSummary:
    row_count: int
    oldest_ts: str | None
    oldest_value: str | None
    oldest_sum: str | None
    latest_ts: str | None
    latest_value: str | None
    latest_sum: str | None
    unavailable_count: int
    oldest_available_ts: str | None
    oldest_available_value: str | None
    oldest_available_sum: str | None
    latest_available_ts: str | None
    latest_available_value: str | None
    latest_available_sum: str | None
    unavailable_occurrences: tuple[OccurrenceRun, ...]


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        """
        SELECT 1
        FROM (
            SELECT name, type FROM sqlite_master
            UNION ALL
            SELECT name, type FROM sqlite_temp_master
        )
        WHERE type IN ('table','view') AND name=?
        LIMIT 1
        """,
        (table,),
    )
    return cur.fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def connect_readonly_sqlite(db_path: Path) -> sqlite3.Connection:
    db_path = db_path.expanduser()
    if not db_path.exists():
        raise DbSummaryError(f"DB file not found: {db_path}")

    # Use read-only mode to avoid locks/corruption.
    uri = db_path.resolve().as_uri() + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise DbSummaryError(f"Failed to open DB: {db_path} ({exc})") from exc

    conn.row_factory = sqlite3.Row
    return conn

def apply_sql_script(db_path: Path, sql_script: str) -> None:
    """Execute a SQL script against the DB (read-write) and commit.

    Intended for Stage 4 application. The script may include its own BEGIN/COMMIT.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        conn.executescript(sql_script)
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        finally:
            raise
    finally:
        conn.close()


def _fmt_ts(value: Any, *, assume_seconds: bool) -> str | None:
    if value is None:
        return None

    if assume_seconds:
        try:
            ts = float(value)
        except (TypeError, ValueError):
            return str(value)
        # Convert to local timezone and drop offset.
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    # If HA stored a datetime string, try parsing and converting to local time.
    if isinstance(value, str):
        text = value.strip()
        try:
            # Support trailing 'Z'
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                # Assume it's already local time.
                return parsed.strftime("%Y-%m-%d %H:%M:%S")
            return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return text

    return str(value)


def _to_epoch_seconds(value: Any, *, assume_seconds: bool) -> float | None:
    if value is None:
        return None

    if assume_seconds:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                # Assume it's already local time.
                parsed = parsed.astimezone()
            return parsed.astimezone(timezone.utc).timestamp()
        except Exception:
            return None

    return None


def _pick_first_present(columns: set[str], candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def summarize_states(conn: sqlite3.Connection, entity_id: str) -> RowSummary:
    """Summarize rows in `states` for an entity.

    Supports both:
    - legacy schema with `states.entity_id`
    - modern schema with `states.metadata_id` joining `states_meta.metadata_id` to `states_meta.entity_id`
    """
    if not _table_exists(conn, "states"):
        return RowSummary(
            row_count=0,
            oldest_ts=None,
            oldest_value=None,
            oldest_sum=None,
            latest_ts=None,
            latest_value=None,
            latest_sum=None,
            unavailable_count=0,
            oldest_available_ts=None,
            oldest_available_value=None,
            oldest_available_sum=None,
            latest_available_ts=None,
            latest_available_value=None,
            latest_available_sum=None,
            unavailable_occurrences=(),
        )

    states_cols = _columns(conn, "states")

    # Prefer numeric timestamps when available
    ts_col = _pick_first_present(states_cols, ["last_updated_ts", "last_changed_ts", "last_updated", "last_changed"])
    if ts_col is None:
        # Fall back to whatever exists
        ts_col = "last_updated_ts" if "last_updated_ts" in states_cols else "last_updated"

    ts_is_seconds = ts_col.endswith("_ts")

    where_sql: str
    params: tuple[Any, ...]
    from_sql: str
    if "metadata_id" in states_cols and _table_exists(conn, "states_meta"):
        from_sql = "states s JOIN states_meta sm ON sm.metadata_id = s.metadata_id"
        where_sql = "sm.entity_id = ?"
        params = (entity_id,)
    elif "entity_id" in states_cols:
        from_sql = "states s"
        where_sql = "s.entity_id = ?"
        params = (entity_id,)
    else:
        return RowSummary(
            row_count=0,
            oldest_ts=None,
            oldest_value=None,
            oldest_sum=None,
            latest_ts=None,
            latest_value=None,
            latest_sum=None,
            unavailable_count=0,
            oldest_available_ts=None,
            oldest_available_value=None,
            oldest_available_sum=None,
            latest_available_ts=None,
            latest_available_value=None,
            latest_available_sum=None,
            unavailable_occurrences=(),
        )

    # Count total
    row_count = int(
        conn.execute(
            f"SELECT COUNT(*) FROM {from_sql} WHERE {where_sql}",
            params,
        ).fetchone()[0]
    )

    # Home Assistant often uses literal strings like 'unavailable'/'unknown' to represent missing state.
    unavailable_where = (
        "(s.state IS NULL OR TRIM(s.state) = '' "
        "OR LOWER(TRIM(s.state)) IN ('unavailable','unknown'))"
    )

    # Count unavailable states
    unavailable_count = int(
        conn.execute(
            f"SELECT COUNT(*) FROM {from_sql} WHERE {where_sql} AND {unavailable_where}",
            params,
        ).fetchone()[0]
    )

    # Oldest
    oldest = conn.execute(
        f"SELECT s.{ts_col} AS ts, s.state AS value FROM {from_sql} WHERE {where_sql} ORDER BY s.{ts_col} ASC LIMIT 1",
        params,
    ).fetchone()

    # Latest
    latest = conn.execute(
        f"SELECT s.{ts_col} AS ts, s.state AS value FROM {from_sql} WHERE {where_sql} ORDER BY s.{ts_col} DESC LIMIT 1",
        params,
    ).fetchone()

    # Oldest/latest available (state defined)
    available_where = f"NOT {unavailable_where}"
    oldest_available = conn.execute(
        f"SELECT s.{ts_col} AS ts, s.state AS value FROM {from_sql} WHERE {where_sql} AND {available_where} ORDER BY s.{ts_col} ASC LIMIT 1",
        params,
    ).fetchone()
    latest_available = conn.execute(
        f"SELECT s.{ts_col} AS ts, s.state AS value FROM {from_sql} WHERE {where_sql} AND {available_where} ORDER BY s.{ts_col} DESC LIMIT 1",
        params,
    ).fetchone()

    unavailable_occurrences: tuple[OccurrenceRun, ...] = ()
    if 0 < unavailable_count <= 200:
        # Build consecutive runs over the *entity-filtered* row stream.
        # Runs split whenever availability flips OR the exact state string changes.
        is_unavail_expr = (
            "CASE WHEN s.state IS NULL OR TRIM(s.state) = '' "
            "OR LOWER(TRIM(s.state)) IN ('unavailable','unknown') THEN 1 ELSE 0 END"
        )
        state_norm = "COALESCE(s.state, '__NULL__')"
        sql = (
            "WITH ordered AS ("
            f"  SELECT s.{ts_col} AS ts, ({is_unavail_expr}) AS is_unavail, {state_norm} AS state_norm, "
            f"         LAG(({is_unavail_expr})) OVER (ORDER BY s.{ts_col}) AS prev_unavail, "
            f"         LAG({state_norm}) OVER (ORDER BY s.{ts_col}) AS prev_state_norm "
            f"  FROM {from_sql} WHERE {where_sql} ORDER BY s.{ts_col}"
            "), marked AS ("
            "  SELECT ts, is_unavail, state_norm, "
            "         CASE WHEN prev_unavail IS NULL THEN 1 "
            "              WHEN prev_unavail != is_unavail THEN 1 "
            "              WHEN prev_state_norm != state_norm THEN 1 "
            "              ELSE 0 END AS new_group "
            "  FROM ordered"
            "), grouped AS ("
            "  SELECT ts, is_unavail, state_norm, "
            "         SUM(new_group) OVER (ORDER BY ts) AS grp "
            "  FROM marked"
            ") "
            "SELECT grp, state_norm, MIN(ts) AS start_ts, MAX(ts) AS end_ts, COUNT(*) AS run_len "
            "FROM grouped WHERE is_unavail = 1 GROUP BY grp, state_norm ORDER BY MIN(ts)"
        )

        runs = conn.execute(sql, params).fetchall()
        formatted: list[OccurrenceRun] = []
        for r in runs:
            start_ts = _fmt_ts(r["start_ts"], assume_seconds=ts_is_seconds)
            end_ts = _fmt_ts(r["end_ts"], assume_seconds=ts_is_seconds)
            run_len = int(r["run_len"])
            if start_ts is None or end_ts is None:
                continue

            value_norm = r["state_norm"]
            value_display = "NULL" if value_norm == "__NULL__" else str(value_norm)
            formatted.append(
                OccurrenceRun(
                    start_ts=start_ts,
                    end_ts=end_ts,
                    count=run_len,
                    value=value_display,
                )
            )

        unavailable_occurrences = tuple(formatted)

    oldest_ts = _fmt_ts(oldest["ts"], assume_seconds=ts_is_seconds) if oldest else None
    oldest_value = None if not oldest else (None if oldest["value"] is None else str(oldest["value"]))

    latest_ts = _fmt_ts(latest["ts"], assume_seconds=ts_is_seconds) if latest else None
    latest_value = None if not latest else (None if latest["value"] is None else str(latest["value"]))

    oldest_available_ts = _fmt_ts(oldest_available["ts"], assume_seconds=ts_is_seconds) if oldest_available else None
    oldest_available_value = (
        None if not oldest_available else (None if oldest_available["value"] is None else str(oldest_available["value"]))
    )

    latest_available_ts = _fmt_ts(latest_available["ts"], assume_seconds=ts_is_seconds) if latest_available else None
    latest_available_value = (
        None if not latest_available else (None if latest_available["value"] is None else str(latest_available["value"]))
    )

    return RowSummary(
        row_count=row_count,
        oldest_ts=oldest_ts,
        oldest_value=oldest_value,
        oldest_sum=None,
        latest_ts=latest_ts,
        latest_value=latest_value,
        latest_sum=None,
        unavailable_count=unavailable_count,
        oldest_available_ts=oldest_available_ts,
        oldest_available_value=oldest_available_value,
        oldest_available_sum=None,
        latest_available_ts=latest_available_ts,
        latest_available_value=latest_available_value,
        latest_available_sum=None,
        unavailable_occurrences=unavailable_occurrences,
    )


def summarize_statistics(conn: sqlite3.Connection, table: str, statistic_id: str) -> RowSummary:
    """Summarize rows in `statistics` / `statistics_short_term` for a statistic_id.

    Supports metadata schema (`statistics_meta` + `metadata_id`) and legacy schema with `statistic_id` directly.

    Value selection (display): prefers `state`, else `mean`, else `min`, else `max`.
    `sum` is tracked separately (and may be shown for total_increasing entities).
    """
    if not _table_exists(conn, table):
        return RowSummary(
            row_count=0,
            oldest_ts=None,
            oldest_value=None,
            oldest_sum=None,
            latest_ts=None,
            latest_value=None,
            latest_sum=None,
            unavailable_count=0,
            oldest_available_ts=None,
            oldest_available_value=None,
            oldest_available_sum=None,
            latest_available_ts=None,
            latest_available_value=None,
            latest_available_sum=None,
            unavailable_occurrences=(),
        )

    cols = _columns(conn, table)
    ts_col = _pick_first_present(cols, ["start_ts", "start"])
    if ts_col is None:
        return RowSummary(
            row_count=0,
            oldest_ts=None,
            oldest_value=None,
            oldest_sum=None,
            latest_ts=None,
            latest_value=None,
            latest_sum=None,
            unavailable_count=0,
            oldest_available_ts=None,
            oldest_available_value=None,
            oldest_available_sum=None,
            latest_available_ts=None,
            latest_available_value=None,
            latest_available_sum=None,
            unavailable_occurrences=(),
        )

    ts_is_seconds = ts_col.endswith("_ts")

    display_cols = [c for c in ("state", "mean", "min", "max") if c in cols]
    candidate_cols = [c for c in ("state", "sum", "mean", "min", "max") if c in cols]
    if not candidate_cols:
        return RowSummary(
            row_count=0,
            oldest_ts=None,
            oldest_value=None,
            oldest_sum=None,
            latest_ts=None,
            latest_value=None,
            latest_sum=None,
            unavailable_count=0,
            oldest_available_ts=None,
            oldest_available_value=None,
            oldest_available_sum=None,
            latest_available_ts=None,
            latest_available_value=None,
            latest_available_sum=None,
            unavailable_occurrences=(),
        )

    def _build_coalesce(cols_: list[str]) -> str:
        if not cols_:
            return "NULL"
        if len(cols_) == 1:
            return f"t.{cols_[0]}"
        return "COALESCE(" + ",".join(f"t.{c}" for c in cols_) + ")"

    value_expr = _build_coalesce(display_cols if display_cols else (["sum"] if "sum" in cols else []))
    candidate_expr = _build_coalesce(candidate_cols)

    from_sql: str
    where_sql: str
    params: tuple[Any, ...]

    if "metadata_id" in cols and _table_exists(conn, "statistics_meta"):
        meta_cols = _columns(conn, "statistics_meta")
        if "statistic_id" in meta_cols and "id" in meta_cols:
            from_sql = f"{table} t JOIN statistics_meta m ON m.id = t.metadata_id"
            where_sql = "m.statistic_id = ?"
            params = (statistic_id,)
        else:
            return RowSummary(
                row_count=0,
                oldest_ts=None,
                oldest_value=None,
                oldest_sum=None,
                latest_ts=None,
                latest_value=None,
                latest_sum=None,
                unavailable_count=0,
                oldest_available_ts=None,
                oldest_available_value=None,
                oldest_available_sum=None,
                latest_available_ts=None,
                latest_available_value=None,
                latest_available_sum=None,
                unavailable_occurrences=(),
            )
    elif "statistic_id" in cols:
        from_sql = f"{table} t"
        where_sql = "t.statistic_id = ?"
        params = (statistic_id,)
    else:
        return RowSummary(
            row_count=0,
            oldest_ts=None,
            oldest_value=None,
            oldest_sum=None,
            latest_ts=None,
            latest_value=None,
            latest_sum=None,
            unavailable_count=0,
            oldest_available_ts=None,
            oldest_available_value=None,
            oldest_available_sum=None,
            latest_available_ts=None,
            latest_available_value=None,
            latest_available_sum=None,
            unavailable_occurrences=(),
        )

    row_count = int(conn.execute(f"SELECT COUNT(*) FROM {from_sql} WHERE {where_sql}", params).fetchone()[0])

    unavailable_where = f"({candidate_expr}) IS NULL"
    unavailable_count = int(
        conn.execute(
            f"SELECT COUNT(*) FROM {from_sql} WHERE {where_sql} AND {unavailable_where}",
            params,
        ).fetchone()[0]
    )

    sum_select = "t.sum" if "sum" in cols else "NULL"

    oldest = conn.execute(
        f"SELECT t.{ts_col} AS ts, {value_expr} AS value, {sum_select} AS sum_value FROM {from_sql} WHERE {where_sql} ORDER BY t.{ts_col} ASC LIMIT 1",
        params,
    ).fetchone()

    latest = conn.execute(
        f"SELECT t.{ts_col} AS ts, {value_expr} AS value, {sum_select} AS sum_value FROM {from_sql} WHERE {where_sql} ORDER BY t.{ts_col} DESC LIMIT 1",
        params,
    ).fetchone()

    available_where = f"NOT {unavailable_where}"
    oldest_available = conn.execute(
        f"SELECT t.{ts_col} AS ts, {value_expr} AS value, {sum_select} AS sum_value FROM {from_sql} WHERE {where_sql} AND {available_where} ORDER BY t.{ts_col} ASC LIMIT 1",
        params,
    ).fetchone()
    latest_available = conn.execute(
        f"SELECT t.{ts_col} AS ts, {value_expr} AS value, {sum_select} AS sum_value FROM {from_sql} WHERE {where_sql} AND {available_where} ORDER BY t.{ts_col} DESC LIMIT 1",
        params,
    ).fetchone()

    unavailable_occurrences: tuple[OccurrenceRun, ...] = ()
    if 0 < unavailable_count <= 200:
        # Build consecutive-run ranges over the statistic_id-filtered row stream.
        is_unavail_expr = f"CASE WHEN ({candidate_expr}) IS NULL THEN 1 ELSE 0 END"
        seg_expr = f"SUM(CASE WHEN ({is_unavail_expr}) = 0 THEN 1 ELSE 0 END) OVER (ORDER BY t.{ts_col})"
        sql = (
            "WITH ordered AS ("
            f"  SELECT t.{ts_col} AS ts, ({is_unavail_expr}) AS is_unavail, {seg_expr} AS seg "
            f"  FROM {from_sql} WHERE {where_sql} ORDER BY t.{ts_col}"
            ") "
            "SELECT seg, MIN(ts) AS start_ts, MAX(ts) AS end_ts, COUNT(*) AS run_len "
            "FROM ordered WHERE is_unavail = 1 GROUP BY seg ORDER BY MIN(ts)"
        )
        runs = conn.execute(sql, params).fetchall()
        formatted: list[OccurrenceRun] = []
        for r in runs:
            start_ts = _fmt_ts(r["start_ts"], assume_seconds=ts_is_seconds)
            end_ts = _fmt_ts(r["end_ts"], assume_seconds=ts_is_seconds)
            run_len = int(r["run_len"])
            if start_ts is None or end_ts is None:
                continue
            formatted.append(
                OccurrenceRun(
                    start_ts=start_ts,
                    end_ts=end_ts,
                    count=run_len,
                    value="NULL",
                )
            )

        unavailable_occurrences = tuple(formatted)

    oldest_ts = _fmt_ts(oldest["ts"], assume_seconds=ts_is_seconds) if oldest else None
    oldest_value = None if not oldest else (None if oldest["value"] is None else str(oldest["value"]))
    oldest_sum = None if not oldest else (None if oldest["sum_value"] is None else str(oldest["sum_value"]))

    latest_ts = _fmt_ts(latest["ts"], assume_seconds=ts_is_seconds) if latest else None
    latest_value = None if not latest else (None if latest["value"] is None else str(latest["value"]))
    latest_sum = None if not latest else (None if latest["sum_value"] is None else str(latest["sum_value"]))

    oldest_available_ts = _fmt_ts(oldest_available["ts"], assume_seconds=ts_is_seconds) if oldest_available else None
    oldest_available_value = (
        None if not oldest_available else (None if oldest_available["value"] is None else str(oldest_available["value"]))
    )
    oldest_available_sum = (
        None if not oldest_available else (None if oldest_available["sum_value"] is None else str(oldest_available["sum_value"]))
    )

    latest_available_ts = _fmt_ts(latest_available["ts"], assume_seconds=ts_is_seconds) if latest_available else None
    latest_available_value = (
        None if not latest_available else (None if latest_available["value"] is None else str(latest_available["value"]))
    )
    latest_available_sum = (
        None if not latest_available else (None if latest_available["sum_value"] is None else str(latest_available["sum_value"]))
    )

    return RowSummary(
        row_count=row_count,
        oldest_ts=oldest_ts,
        oldest_value=oldest_value,
        oldest_sum=oldest_sum,
        latest_ts=latest_ts,
        latest_value=latest_value,
        latest_sum=latest_sum,
        unavailable_count=unavailable_count,
        oldest_available_ts=oldest_available_ts,
        oldest_available_value=oldest_available_value,
        oldest_available_sum=oldest_available_sum,
        latest_available_ts=latest_available_ts,
        latest_available_value=latest_available_value,
        latest_available_sum=latest_available_sum,
        unavailable_occurrences=unavailable_occurrences,
    )


def summarize_all(conn: sqlite3.Connection, entity_id: str, *, total_increasing: bool = False) -> dict[str, Any]:
    """Build a nested dict suitable for report rendering (flattened by report.py)."""
    states = summarize_states(conn, entity_id)
    stats = summarize_statistics(conn, "statistics", entity_id)
    stats_st = summarize_statistics(conn, "statistics_short_term", entity_id)

    def _format_value_with_sum(value: str | None, sum_value: str | None) -> str | None:
        if value is None:
            return None
        if total_increasing and sum_value is not None:
            return f"{value} ({sum_value})"
        return value

    def pack(s: RowSummary, *, include_sum: bool) -> dict[str, Any]:
        out: dict[str, Any] = {
            "row_count": s.row_count,
            "earliest": {
                "ts": s.oldest_ts,
                "value": _format_value_with_sum(s.oldest_value, s.oldest_sum) if include_sum else s.oldest_value,
            },
            "latest": {
                "ts": s.latest_ts,
                "value": _format_value_with_sum(s.latest_value, s.latest_sum) if include_sum else s.latest_value,
            },
            "unavailable_count": s.unavailable_count,
        }

        oldest_av_value = (
            _format_value_with_sum(s.oldest_available_value, s.oldest_available_sum)
            if include_sum
            else s.oldest_available_value
        )
        latest_av_value = (
            _format_value_with_sum(s.latest_available_value, s.latest_available_sum)
            if include_sum
            else s.latest_available_value
        )

        # Omit *_available fields when they are identical to earliest/latest.
        if not (s.oldest_available_ts == s.oldest_ts and oldest_av_value == out["earliest"]["value"]):
            out["earliest_available"] = {"ts": s.oldest_available_ts, "value": oldest_av_value}
        if not (s.latest_available_ts == s.latest_ts and latest_av_value == out["latest"]["value"]):
            out["latest_available"] = {"ts": s.latest_available_ts, "value": latest_av_value}

        return out

    return {
        "states": pack(states, include_sum=False),
        "statistics": pack(stats, include_sum=True),
        "statistics_short_term": pack(stats_st, include_sum=True),
    }


def collect_unavailable_occurrence_rows(conn: sqlite3.Connection, entity_id: str) -> list[dict[str, str]]:
    """Return rows for `render_unavailable_occurrences_report()`.

    Only includes details when unavailable_count is between 1 and 200.
    """
    out: list[dict[str, str]] = []

    for table_name, summary in (
        ("states", summarize_states(conn, entity_id)),
        ("statistics", summarize_statistics(conn, "statistics", entity_id)),
        ("statistics_short_term", summarize_statistics(conn, "statistics_short_term", entity_id)),
    ):
        if not (0 < summary.unavailable_count <= 200):
            continue

        for run in summary.unavailable_occurrences:
            when = run.start_ts if run.count <= 1 else f"{run.start_ts} - {run.end_ts}"
            out.append(
                {
                    "entity": entity_id,
                    "table": table_name,
                    "when": when,
                    "value": run.value,
                    "count": str(run.count),
                }
            )

    return out


def collect_missing_statistics_row_ranges(
    conn: sqlite3.Connection,
    table: str,
    statistic_id: str,
    *,
    interval_seconds: int,
) -> list[dict[str, str]]:
    """Detect missing row ranges in a statistics table.

    Home Assistant statistics tables are expected to have regular rows:
    - statistics: 1 hour
    - statistics_short_term: 5 minutes

    Reports gaps as: `before_ts (state/sum) - after_ts (state/sum) [N rows]`.
    """
    if not _table_exists(conn, table):
        return []

    cols = _columns(conn, table)
    ts_col = _pick_first_present(cols, ["start_ts", "start"])
    if ts_col is None:
        return []
    ts_is_seconds = ts_col.endswith("_ts")

    if "metadata_id" in cols and _table_exists(conn, "statistics_meta"):
        meta_cols = _columns(conn, "statistics_meta")
        if not ("statistic_id" in meta_cols and "id" in meta_cols):
            return []
        from_sql = f"{table} t JOIN statistics_meta m ON m.id = t.metadata_id"
        where_sql = "m.statistic_id = ?"
        params = (statistic_id,)
    elif "statistic_id" in cols:
        from_sql = f"{table} t"
        where_sql = "t.statistic_id = ?"
        params = (statistic_id,)
    else:
        return []

    state_sel = "t.state AS state" if "state" in cols else "NULL AS state"
    sum_sel = "t.sum AS sum" if "sum" in cols else "NULL AS sum"

    rows = conn.execute(
        f"SELECT t.{ts_col} AS ts, {state_sel}, {sum_sel} FROM {from_sql} WHERE {where_sql} ORDER BY t.{ts_col} ASC",
        params,
    ).fetchall()

    def fmt_state_sum(state: Any, sum_value: Any) -> str:
        def fmt(v: Any) -> str:
            return "NULL" if v is None else str(v)

        have_state = "state" in cols
        have_sum = "sum" in cols
        if have_state and have_sum:
            return f"{fmt(state)}/{fmt(sum_value)}"
        if have_state:
            return fmt(state)
        if have_sum:
            return fmt(sum_value)
        return ""

    out: list[dict[str, str]] = []
    prev_epoch: float | None = None
    prev_ts_raw: Any = None
    prev_state: Any = None
    prev_sum: Any = None

    # tolerate small clock/float drift
    tol_seconds = max(2.0, interval_seconds * 0.01)

    for r in rows:
        cur_epoch = _to_epoch_seconds(r["ts"], assume_seconds=ts_is_seconds)
        if cur_epoch is None:
            # Can't compute gaps without a usable timestamp.
            prev_epoch = None
            prev_ts_raw = None
            prev_state = None
            prev_sum = None
            continue

        if prev_epoch is not None:
            delta = cur_epoch - prev_epoch
            if delta > interval_seconds * 1.5:
                n = int(round(delta / interval_seconds))
                missing = n - 1
                if missing <= 0:
                    missing = int(delta // interval_seconds) - 1
                if missing > 0:
                    # If the delta isn't close to a multiple, still report with the computed count.
                    if abs(delta - (missing + 1) * interval_seconds) > tol_seconds:
                        # keep the computed missing; just don't try to “correct” it
                        pass

                    before_ts = _fmt_ts(prev_ts_raw, assume_seconds=ts_is_seconds) or ""
                    after_ts = _fmt_ts(r["ts"], assume_seconds=ts_is_seconds) or ""
                    before_val = fmt_state_sum(prev_state, prev_sum)
                    after_val = fmt_state_sum(r["state"], r["sum"])
                    out.append(
                        {
                            "entity": statistic_id,
                            "table": table,
                            "gap_start_epoch": str(prev_epoch),
                            "gap": f"{before_ts} ({before_val}) - {after_ts} ({after_val}) [{missing} rows]",
                        }
                    )

        prev_epoch = cur_epoch
        prev_ts_raw = r["ts"]
        prev_state = r["state"]
        prev_sum = r["sum"]

    return out


def get_earliest_state_ts_epoch(conn: sqlite3.Connection, entity_id: str) -> float | None:
    """Return earliest state timestamp (epoch seconds) for an entity."""
    if not _table_exists(conn, "states"):
        return None

    cols = _columns(conn, "states")
    ts_col = _pick_first_present(cols, ["last_updated_ts", "last_changed_ts", "last_updated", "last_changed"])
    if ts_col is None:
        return None
    ts_is_seconds = ts_col.endswith("_ts")

    if "metadata_id" in cols and _table_exists(conn, "states_meta"):
        from_sql = "states s JOIN states_meta sm ON sm.metadata_id = s.metadata_id"
        where_sql = "sm.entity_id = ?"
        params = (entity_id,)
    elif "entity_id" in cols:
        from_sql = "states s"
        where_sql = "s.entity_id = ?"
        params = (entity_id,)
    else:
        return None

    row = conn.execute(
        f"SELECT MIN(s.{ts_col}) AS ts FROM {from_sql} WHERE {where_sql}",
        params,
    ).fetchone()
    if not row:
        return None

    return _to_epoch_seconds(row["ts"], assume_seconds=ts_is_seconds)


def get_latest_statistics_ts_epoch(conn: sqlite3.Connection, table: str, statistic_id: str) -> float | None:
    """Return latest statistics timestamp (epoch seconds) for a statistic_id in a table."""
    if not _table_exists(conn, table):
        return None

    cols = _columns(conn, table)
    ts_col = _pick_first_present(cols, ["start_ts", "start"])
    if ts_col is None:
        return None
    ts_is_seconds = ts_col.endswith("_ts")

    if "metadata_id" in cols and _table_exists(conn, "statistics_meta"):
        meta_cols = _columns(conn, "statistics_meta")
        if not ("statistic_id" in meta_cols and "id" in meta_cols):
            return None
        from_sql = f"{table} t JOIN statistics_meta m ON m.id = t.metadata_id"
        where_sql = "m.statistic_id = ?"
        params = (statistic_id,)
    elif "statistic_id" in cols:
        from_sql = f"{table} t"
        where_sql = "t.statistic_id = ?"
        params = (statistic_id,)
    else:
        return None

    row = conn.execute(
        f"SELECT MAX(t.{ts_col}) AS ts FROM {from_sql} WHERE {where_sql}",
        params,
    ).fetchone()
    if not row:
        return None
    return _to_epoch_seconds(row["ts"], assume_seconds=ts_is_seconds)


def create_statistics_generated_view(
    conn: sqlite3.Connection,
    *,
    view_name: str,
    source_table: str,
    old_statistic_id: str,
    new_statistic_id: str,
    interval_seconds: int,
    statistics_kind: str,
    new_entity_started_from_0: bool,
) -> None:
    """Create a TEMP VIEW for generated statistics (Stage 4)."""
    sql = build_statistics_generated_view_sql(
        conn,
        view_name=view_name,
        source_table=source_table,
        old_statistic_id=old_statistic_id,
        new_statistic_id=new_statistic_id,
        interval_seconds=interval_seconds,
        statistics_kind=statistics_kind,
        new_entity_started_from_0=new_entity_started_from_0,
    )

    conn.execute(f"DROP VIEW IF EXISTS {view_name}")
    conn.execute(sql)


def build_statistics_generated_view_sql(
    conn: sqlite3.Connection,
    *,
    view_name: str,
    source_table: str,
    old_statistic_id: str,
    new_statistic_id: str,
    interval_seconds: int,
    statistics_kind: str,
    new_entity_started_from_0: bool,
) -> str:
    """Return the CREATE TEMP VIEW SQL for generated statistics (Stage 4)."""
    if not _table_exists(conn, source_table):
        raise DbSummaryError(f"Missing statistics table: {source_table}")
    if not _table_exists(conn, "states"):
        raise DbSummaryError("Missing states table")

    def sql_quote(text: str) -> str:
        return "'" + text.replace("'", "''") + "'"

    old_id_q = sql_quote(old_statistic_id)
    new_id_q = sql_quote(new_statistic_id)

    # Stats table schema
    s_cols = _columns(conn, source_table)
    s_ts_col = _pick_first_present(s_cols, ["start_ts", "start"])
    if s_ts_col is None:
        raise DbSummaryError(f"No start_ts/start column in {source_table}")
    stats_ts_is_seconds = s_ts_col.endswith("_ts")

    if "metadata_id" in s_cols and _table_exists(conn, "statistics_meta"):
        meta_cols = _columns(conn, "statistics_meta")
        if not ("statistic_id" in meta_cols and "id" in meta_cols):
            raise DbSummaryError("statistics_meta missing required columns")
        stats_from_sql = f"{source_table} t JOIN statistics_meta m ON m.id = t.metadata_id"
        stats_where_sql = f"m.statistic_id = {old_id_q}"
    elif "statistic_id" in s_cols:
        stats_from_sql = f"{source_table} t"
        stats_where_sql = f"t.statistic_id = {old_id_q}"
    else:
        raise DbSummaryError(f"Unsupported schema for {source_table}")

    # States schema
    st_cols = _columns(conn, "states")
    st_ts_col = _pick_first_present(st_cols, ["last_updated_ts", "last_changed_ts", "last_updated", "last_changed"])
    if st_ts_col is None:
        raise DbSummaryError("No timestamp column in states")
    states_ts_is_seconds = st_ts_col.endswith("_ts")

    if "metadata_id" in st_cols and _table_exists(conn, "states_meta"):
        states_from_sql = "states s JOIN states_meta sm ON sm.metadata_id = s.metadata_id"
        states_where_sql = f"sm.entity_id = {new_id_q}"
    elif "entity_id" in st_cols:
        states_from_sql = "states s"
        states_where_sql = f"s.entity_id = {new_id_q}"
    else:
        raise DbSummaryError("Unsupported schema for states")

    def epoch_expr(col_ref: str, *, assume_seconds: bool) -> str:
        if assume_seconds:
            return f"CAST({col_ref} AS REAL)"
        return f"CAST(strftime('%s', {col_ref}) AS REAL)"

    stats_epoch = epoch_expr(f"t.{s_ts_col}", assume_seconds=stats_ts_is_seconds)
    states_epoch = epoch_expr(f"s.{st_ts_col}", assume_seconds=states_ts_is_seconds)

    stats_state_sel = "t.state" if "state" in s_cols else "NULL"
    stats_sum_sel = "t.sum" if "sum" in s_cols else "NULL"
    stats_mean_sel = "t.mean" if "mean" in s_cols else "NULL"
    stats_min_sel = "t.min" if "min" in s_cols else "NULL"
    stats_max_sel = "t.max" if "max" in s_cols else "NULL"
    stats_created_sel = "t.created_ts" if "created_ts" in s_cols else "NULL"

    if statistics_kind not in {"total_increasing", "measurement"}:
        raise DbSummaryError(f"Unsupported statistics_kind: {statistics_kind}")

    if statistics_kind == "total_increasing":
        if "sum" in s_cols:
            base_sum_expr = "COALESCE((SELECT os.sum FROM old_stats os WHERE os.sum IS NOT NULL ORDER BY os.start_ts DESC LIMIT 1), 0.0)"
        else:
            base_sum_expr = "0.0"
    else:
        base_sum_expr = "0.0"

    first_dv_expr = "v" if (statistics_kind == "total_increasing" and new_entity_started_from_0) else "0.0"

    if statistics_kind == "total_increasing":
        new_stats_cte = f"""
new_stats AS (
    SELECT
        {new_id_q} AS statistic_id,
        CAST(strftime('%s','now') AS REAL) AS created_ts,
        start_ts,
        state,
        sum,
        NULL AS mean,
        NULL AS min,
        NULL AS max
    FROM bucket_last
    WHERE rn = 1
)
""".strip()
    else:
        new_stats_cte = f"""
bucket_aggs AS (
    SELECT
        CAST(ts / {interval_seconds} AS INT) * {interval_seconds} AS start_ts,
        MIN(v) AS min,
        AVG(v) AS mean,
        MAX(v) AS max
    FROM num_states
    GROUP BY CAST(ts / {interval_seconds} AS INT) * {interval_seconds}
),
bucket_last2 AS (
    SELECT
        CAST(ts / {interval_seconds} AS INT) * {interval_seconds} AS start_ts,
        v AS state,
        ROW_NUMBER() OVER (PARTITION BY CAST(ts / {interval_seconds} AS INT) * {interval_seconds} ORDER BY ts DESC) AS rn
    FROM num_states
),
new_stats AS (
    SELECT
        {new_id_q} AS statistic_id,
        CAST(strftime('%s','now') AS REAL) AS created_ts,
        a.start_ts,
        l.state AS state,
        NULL AS sum,
        a.mean AS mean,
        a.min AS min,
        a.max AS max
    FROM bucket_aggs a
    JOIN bucket_last2 l ON l.start_ts = a.start_ts
    WHERE l.rn = 1
)
""".strip()

    return f"""
CREATE TEMP VIEW {view_name} AS
WITH
old_stats AS (
    SELECT
        {new_id_q} AS statistic_id,
        {stats_created_sel} AS created_ts,
        {stats_epoch} AS start_ts,
        {stats_state_sel} AS state,
        {stats_sum_sel} AS sum
        ,{stats_mean_sel} AS mean
        ,{stats_min_sel} AS min
        ,{stats_max_sel} AS max
    FROM {stats_from_sql}
    WHERE {stats_where_sql}
),
base AS (
    SELECT {base_sum_expr} AS base_sum
),
raw_states AS (
    SELECT
        {states_epoch} AS ts,
        s.state AS raw_state
    FROM {states_from_sql}
    WHERE {states_where_sql}
),
num_states AS (
    SELECT
        ts,
        CAST(raw_state AS REAL) AS v
    FROM raw_states
    WHERE raw_state IS NOT NULL
        AND TRIM(CAST(raw_state AS TEXT)) <> ''
        AND LOWER(TRIM(CAST(raw_state AS TEXT))) NOT IN ('unavailable','unknown')
),
deltas AS (
    SELECT
        ts,
        v,
        CASE
            WHEN LAG(v) OVER (ORDER BY ts) IS NULL THEN {first_dv_expr}
            WHEN v - LAG(v) OVER (ORDER BY ts) < 0 THEN 0.0
            ELSE v - LAG(v) OVER (ORDER BY ts)
        END AS dv
    FROM num_states
),
cumulative AS (
    SELECT
        ts,
        v,
        (SELECT base_sum FROM base) + SUM(dv) OVER (ORDER BY ts ROWS UNBOUNDED PRECEDING) AS sum_v,
        CAST(ts / {interval_seconds} AS INT) * {interval_seconds} AS bucket
    FROM deltas
),
bucket_last AS (
    SELECT
        bucket AS start_ts,
        v AS state,
        sum_v AS sum,
        ROW_NUMBER() OVER (PARTITION BY bucket ORDER BY ts DESC) AS rn
    FROM cumulative
),
{new_stats_cte}
SELECT statistic_id, created_ts, start_ts, state, sum, mean, min, max FROM old_stats
UNION ALL
SELECT statistic_id, created_ts, start_ts, state, sum, mean, min, max FROM new_stats;
""".strip()


def build_statistics_update_sql_script(
    conn: sqlite3.Connection,
    *,
    old_entity_id: str,
    new_entity_id: str,
    stats_view: str,
    stats_st_view: str,
    statistics_kind: str,
    new_entity_started_from_0: bool,
) -> str:
    """Build a SQL script that deletes+inserts statistics rows for new_entity_id.

    Stage 4: the script recreates the TEMP VIEWs (single-SELECT generation) and then:
    - deletes existing rows for the new entity
    - inserts the generated rows
    """

    def sql_quote(text: str) -> str:
        return "'" + text.replace("'", "''") + "'"

    new_id_q = sql_quote(new_entity_id)

    def table_dml(table: str, view: str) -> str:
        cols = _columns(conn, table)
        ts_col = "start_ts" if "start_ts" in cols else ("start" if "start" in cols else None)
        if ts_col is None:
            raise DbSummaryError(f"No start_ts/start column in {table}")

        # Destination columns to populate
        dest_cols: list[str] = []
        select_cols: list[str] = []

        if "metadata_id" in cols:
            if not _table_exists(conn, "statistics_meta"):
                raise DbSummaryError("statistics_meta missing (required for metadata schema)")
            meta_cols = _columns(conn, "statistics_meta")
            if not ("id" in meta_cols and "statistic_id" in meta_cols):
                raise DbSummaryError("statistics_meta missing required columns")
            meta_id_expr = f"(SELECT id FROM statistics_meta WHERE statistic_id = {new_id_q})"
            delete_sql = f"DELETE FROM {table} WHERE metadata_id = {meta_id_expr};"
            dest_cols.append("metadata_id")
            select_cols.append(f"{meta_id_expr} AS metadata_id")
        elif "statistic_id" in cols:
            delete_sql = f"DELETE FROM {table} WHERE statistic_id = {new_id_q};"
            dest_cols.append("statistic_id")
            select_cols.append("statistic_id")
        else:
            raise DbSummaryError(f"Unsupported schema for {table}")

        if ts_col == "start_ts":
            dest_cols.append("start_ts")
            select_cols.append("start_ts")
        else:
            dest_cols.append("start")
            select_cols.append("datetime(start_ts, 'unixepoch') AS start")

        if "created_ts" in cols:
            dest_cols.append("created_ts")
            select_cols.append("COALESCE(created_ts, CAST(strftime('%s','now') AS REAL)) AS created_ts")

        # Measurement aggregates
        for c in ("min", "mean", "max"):
            if c in cols:
                dest_cols.append(c)
                select_cols.append(c)

        if "state" in cols:
            dest_cols.append("state")
            select_cols.append("state")
        if "sum" in cols:
            dest_cols.append("sum")
            select_cols.append("sum")

        insert_sql = (
            f"INSERT INTO {table}({', '.join(dest_cols)})\n"
            f"SELECT {', '.join(select_cols)} FROM {view} WHERE statistic_id = {new_id_q};"
        )

        return f"-- {table}\n{delete_sql}\n{insert_sql}"

    sql_stats = build_statistics_generated_view_sql(
        conn,
        view_name=stats_view,
        source_table="statistics",
        old_statistic_id=old_entity_id,
        new_statistic_id=new_entity_id,
        interval_seconds=3600,
        statistics_kind=statistics_kind,
        new_entity_started_from_0=new_entity_started_from_0,
    )
    sql_st = build_statistics_generated_view_sql(
        conn,
        view_name=stats_st_view,
        source_table="statistics_short_term",
        old_statistic_id=old_entity_id,
        new_statistic_id=new_entity_id,
        interval_seconds=300,
        statistics_kind=statistics_kind,
        new_entity_started_from_0=new_entity_started_from_0,
    )

    parts = [
        "-- Stage 4 statistics generation script",
        f"-- old_entity_id: {old_entity_id}",
        f"-- new_entity_id: {new_entity_id}",
        "-- WARNING: Review before executing. This script deletes existing statistics rows for new_entity_id.",
        "BEGIN;",
        f"DROP VIEW IF EXISTS {stats_view};",
        sql_stats,
        f"DROP VIEW IF EXISTS {stats_st_view};",
        sql_st,
        table_dml("statistics", stats_view),
        table_dml("statistics_short_term", stats_st_view),
        "COMMIT;",
        "-- End",
    ]

    return "\n\n".join(parts) + "\n"


def collect_generated_statistics_preview_rows(
    conn: sqlite3.Connection,
    *,
    source_table: str,
    old_entity_id: str,
    new_entity_id: str,
    generated_view: str,
    first_generated: int = 3,
    last_generated: int = 3,
) -> list[dict[str, str]]:
    """Preview of generated rows: 1 copied row + first/last generated rows.

    - "Copied" row is the last row at or before the old entity's latest timestamp in
      the corresponding source table.
    - "Generated" rows are those with start_ts strictly after that timestamp.
    - If there are more generated rows than shown, inserts an ellipsis row.
    """

    if not _table_exists(conn, generated_view):
        return []

    old_latest = get_latest_statistics_ts_epoch(conn, source_table, old_entity_id)
    if old_latest is None:
        return []

    def fmt_row(r: sqlite3.Row) -> dict[str, str]:
        return {
            "ts": _fmt_ts(r["start_ts"], assume_seconds=True) or "",
            "state": "" if r["state"] is None else str(r["state"]),
            "sum": "" if r["sum"] is None else str(r["sum"]),
            "mean": "" if r["mean"] is None else str(r["mean"]),
            "min": "" if r["min"] is None else str(r["min"]),
            "max": "" if r["max"] is None else str(r["max"]),
        }

    copied = conn.execute(
        f"""
                SELECT start_ts, state, sum, mean, min, max
                FROM {generated_view}
        WHERE statistic_id = ?
          AND start_ts <= ?
        ORDER BY start_ts DESC
        LIMIT 1
        """.strip(),
        (new_entity_id, float(old_latest)),
    ).fetchall()

    gen_first = conn.execute(
        f"""
                SELECT start_ts, state, sum, mean, min, max
                FROM {generated_view}
        WHERE statistic_id = ?
          AND start_ts > ?
        ORDER BY start_ts ASC
        LIMIT ?
        """.strip(),
        (new_entity_id, float(old_latest), int(first_generated)),
    ).fetchall()

    gen_last_desc = conn.execute(
        f"""
                SELECT start_ts, state, sum, mean, min, max
                FROM {generated_view}
        WHERE statistic_id = ?
          AND start_ts > ?
        ORDER BY start_ts DESC
        LIMIT ?
        """.strip(),
        (new_entity_id, float(old_latest), int(last_generated)),
    ).fetchall()
    gen_last = list(reversed(gen_last_desc))

    gen_count = conn.execute(
        f"""
        SELECT COUNT(*) AS c
                FROM {generated_view}
        WHERE statistic_id = ?
          AND start_ts > ?
        """.strip(),
        (new_entity_id, float(old_latest)),
    ).fetchone()["c"]

    out: list[dict[str, str]] = []
    if copied:
        out.append(fmt_row(copied[0]))

    # If total generated rows fit, show all of them.
    if gen_count <= (first_generated + last_generated):
        all_gen = conn.execute(
            f"""
                        SELECT start_ts, state, sum, mean, min, max
                        FROM {generated_view}
            WHERE statistic_id = ?
              AND start_ts > ?
            ORDER BY start_ts ASC
            """.strip(),
            (new_entity_id, float(old_latest)),
        ).fetchall()
        out.extend([fmt_row(r) for r in all_gen])
        return out

    out.extend([fmt_row(r) for r in gen_first])
    out.append({"ts": "...", "state": "...", "sum": "...", "mean": "...", "min": "...", "max": "..."})
    out.extend([fmt_row(r) for r in gen_last])
    return out


def collect_reset_events_states(conn: sqlite3.Connection, entity_id: str) -> list[dict[str, str]]:
    """Reset events where numeric `state` decreases compared to the previous row."""
    if not _table_exists(conn, "states"):
        return []

    states_cols = _columns(conn, "states")
    ts_col = _pick_first_present(states_cols, ["last_updated_ts", "last_changed_ts", "last_updated", "last_changed"])
    if ts_col is None:
        return []
    ts_is_seconds = ts_col.endswith("_ts")

    if "metadata_id" in states_cols and _table_exists(conn, "states_meta"):
        from_sql = "states s JOIN states_meta sm ON sm.metadata_id = s.metadata_id"
        where_sql = "sm.entity_id = ?"
        params = (entity_id,)
    elif "entity_id" in states_cols:
        from_sql = "states s"
        where_sql = "s.entity_id = ?"
        params = (entity_id,)
    else:
        return []

    rows = conn.execute(
        f"SELECT s.{ts_col} AS ts, s.state AS state FROM {from_sql} WHERE {where_sql} ORDER BY s.{ts_col} ASC",
        params,
    ).fetchall()

    def is_unavailable_text(val: Any) -> bool:
        if val is None:
            return True
        if not isinstance(val, str):
            return False
        s = val.strip()
        if s == "":
            return True
        return s.lower() in ("unavailable", "unknown")

    out: list[dict[str, str]] = []
    prev_num: float | None = None
    prev_raw: str | None = None
    prev_ts: str | None = None
    for r in rows:
        raw = r["state"]
        if is_unavailable_text(raw):
            continue

        try:
            cur_num = float(str(raw))
        except Exception:
            continue

        if prev_num is not None and cur_num < prev_num:
            cur_ts = _fmt_ts(r["ts"], assume_seconds=ts_is_seconds) or ""
            cur_epoch = _to_epoch_seconds(r["ts"], assume_seconds=ts_is_seconds)
            before_ts = prev_ts or ""
            before_val = prev_raw if prev_raw is not None else ""
            out.append(
                {
                    "entity": entity_id,
                    "table": "states",
                    "event_epoch": "" if cur_epoch is None else str(cur_epoch),
                    "before": f"{before_ts} ({before_val})",
                    "after": f"{cur_ts} ({str(raw)})",
                }
            )

        prev_num = cur_num
        prev_raw = str(raw)
        prev_ts = _fmt_ts(r["ts"], assume_seconds=ts_is_seconds) or ""

    return out


def collect_reset_events_statistics(conn: sqlite3.Connection, table: str, statistic_id: str) -> list[dict[str, str]]:
    """Reset events where numeric value decreases compared to the previous row.

    Prefers `state` when present (it can show resets even when `sum` is reset-adjusted),
    else falls back to `sum`, then other numeric columns.
    """
    if not _table_exists(conn, table):
        return []

    cols = _columns(conn, table)
    ts_col = _pick_first_present(cols, ["start_ts", "start"])
    if ts_col is None:
        return []
    ts_is_seconds = ts_col.endswith("_ts")

    value_cols = [c for c in ("state", "sum", "mean", "min", "max") if c in cols]
    if not value_cols:
        return []

    if "metadata_id" in cols and _table_exists(conn, "statistics_meta"):
        meta_cols = _columns(conn, "statistics_meta")
        if not ("statistic_id" in meta_cols and "id" in meta_cols):
            return []
        from_sql = f"{table} t JOIN statistics_meta m ON m.id = t.metadata_id"
        where_sql = "m.statistic_id = ?"
        params = (statistic_id,)
    elif "statistic_id" in cols:
        from_sql = f"{table} t"
        where_sql = "t.statistic_id = ?"
        params = (statistic_id,)
    else:
        return []

    # Prefer state over sum: HA may keep `sum` monotonic even when the underlying
    # `state` resets (which is what we want to detect here).
    prefer = [c for c in ("state", "sum", "mean", "min", "max") if c in cols]
    if len(prefer) == 1:
        select_expr = f"t.{prefer[0]}"
    else:
        select_expr = "COALESCE(" + ",".join(f"t.{c}" for c in prefer) + ")"

    last_reset_col = _pick_first_present(cols, ["last_reset_ts", "last_reset"])
    last_reset_sel = f"t.{last_reset_col}" if last_reset_col is not None else "NULL"

    rows = conn.execute(
        f"SELECT t.{ts_col} AS ts, {select_expr} AS v, {last_reset_sel} AS lr FROM {from_sql} WHERE {where_sql} ORDER BY t.{ts_col} ASC",
        params,
    ).fetchall()

    out: list[dict[str, str]] = []
    prev_num: float | None = None
    prev_raw: str | None = None
    prev_ts: str | None = None
    for r in rows:
        v = r["v"]
        if v is None:
            continue
        try:
            cur_num = float(v)
        except Exception:
            continue

        if prev_num is not None and cur_num < prev_num:
            cur_ts = _fmt_ts(r["ts"], assume_seconds=ts_is_seconds) or ""
            cur_epoch = _to_epoch_seconds(r["ts"], assume_seconds=ts_is_seconds)
            before_ts = prev_ts or ""
            before_val = prev_raw if prev_raw is not None else ""

            lr_val = r["lr"]
            if lr_val is None or last_reset_col is None:
                last_reset = ""
            else:
                last_reset = _fmt_ts(lr_val, assume_seconds=last_reset_col.endswith("_ts")) or str(lr_val)

            out.append(
                {
                    "entity": statistic_id,
                    "table": table,
                    "event_epoch": "" if cur_epoch is None else str(cur_epoch),
                    "last_reset": last_reset,
                    "before": f"{before_ts} ({before_val})",
                    "after": f"{cur_ts} ({str(v)})",
                }
            )

        prev_num = cur_num
        prev_raw = str(v)
        prev_ts = _fmt_ts(r["ts"], assume_seconds=ts_is_seconds) or ""

    return out


def collect_last_reset_rows(conn: sqlite3.Connection, statistic_id: str) -> list[dict[str, str]]:
    """List statistics rows where last_reset is non-null (if column exists)."""
    out: list[dict[str, str]] = []
    for table in ("statistics", "statistics_short_term"):
        if not _table_exists(conn, table):
            continue

        cols = _columns(conn, table)
        ts_col = _pick_first_present(cols, ["start_ts", "start"])
        if ts_col is None:
            continue
        ts_is_seconds = ts_col.endswith("_ts")

        last_reset_col = _pick_first_present(cols, ["last_reset_ts", "last_reset"])
        if last_reset_col is None:
            continue

        if "metadata_id" in cols and _table_exists(conn, "statistics_meta"):
            meta_cols = _columns(conn, "statistics_meta")
            if not ("statistic_id" in meta_cols and "id" in meta_cols):
                continue
            from_sql = f"{table} t JOIN statistics_meta m ON m.id = t.metadata_id"
            where_sql = f"m.statistic_id = ? AND t.{last_reset_col} IS NOT NULL"
            params = (statistic_id,)
        elif "statistic_id" in cols:
            from_sql = f"{table} t"
            where_sql = f"t.statistic_id = ? AND t.{last_reset_col} IS NOT NULL"
            params = (statistic_id,)
        else:
            continue

        rows = conn.execute(
            f"SELECT t.{ts_col} AS ts, t.{last_reset_col} AS lr FROM {from_sql} WHERE {where_sql} ORDER BY t.{ts_col} ASC",
            params,
        ).fetchall()

        for r in rows:
            out.append(
                {
                    "entity": statistic_id,
                    "table": table,
                    "start": _fmt_ts(r["ts"], assume_seconds=ts_is_seconds) or "",
                    "last_reset": _fmt_ts(r["lr"], assume_seconds=last_reset_col.endswith("_ts")) or str(r["lr"]),
                }
            )

    return out
