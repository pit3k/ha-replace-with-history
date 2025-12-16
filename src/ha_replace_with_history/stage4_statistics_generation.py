from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .db_summary import (
    DbSummaryError,
    apply_sql_script,
    build_statistics_update_sql_script,
    collect_generated_statistics_preview_rows,
    collect_missing_statistics_row_ranges,
    collect_reset_events_statistics,
    create_statistics_generated_view,
    get_earliest_state_ts_epoch,
    get_latest_statistics_ts_epoch,
)
from .report import render_simple_table


@dataclass(frozen=True)
class Stage4Result:
    ran: bool
    sql_path: Path | None


def run_statistics_generation(
    conn: sqlite3.Connection,
    *,
    db_path: Path,
    old_entity_id: str,
    new_entity_id: str,
    old_state_class: str | None,
    new_state_class: str | None,
    new_entity_started_from_0: bool,
    apply: bool,
    color: bool,
) -> Stage4Result:
    print("*** Stage 4: Statistics generation")

    generated_stats_view = "statistics_generated"
    generated_st_view = "statistics_short_term_generated"

    old_total_increasing = old_state_class == "total_increasing"
    new_total_increasing = new_state_class == "total_increasing"
    old_total = old_state_class == "total"
    new_total = new_state_class == "total"
    old_measurement = old_state_class == "measurement"
    new_measurement = new_state_class == "measurement"

    if old_total_increasing and new_total_increasing:
        statistics_kind = "total_increasing"
    elif old_total and new_total:
        statistics_kind = "total"
    elif old_measurement and new_measurement:
        statistics_kind = "measurement"
    else:
        print("Stage 4 skipped: both entities must be total_increasing, total, or measurement (and match).")
        return Stage4Result(ran=False, sql_path=None)

    # Preconditions:
    # - new states earliest > old stats latest
    new_states_earliest = get_earliest_state_ts_epoch(conn, new_entity_id)
    old_stats_latest = get_latest_statistics_ts_epoch(conn, "statistics", old_entity_id)
    old_stats_st_latest = get_latest_statistics_ts_epoch(conn, "statistics_short_term", old_entity_id)
    old_stats_latest_any = max([v for v in (old_stats_latest, old_stats_st_latest) if v is not None], default=None)

    if new_states_earliest is None or old_stats_latest_any is None:
        print("Stage 4 skipped: missing states/statistics timestamps for precondition checks.")
        return Stage4Result(ran=False, sql_path=None)

    if not (new_states_earliest > old_stats_latest_any):
        print("Stage 4 skipped: precondition failed (new states earliest must be after old statistics latest).")
        return Stage4Result(ran=False, sql_path=None)

    try:
        create_statistics_generated_view(
            conn,
            view_name=generated_stats_view,
            source_table="statistics",
            old_statistic_id=old_entity_id,
            new_statistic_id=new_entity_id,
            interval_seconds=3600,
            statistics_kind=statistics_kind,
            new_entity_started_from_0=new_entity_started_from_0,
        )
        create_statistics_generated_view(
            conn,
            view_name=generated_st_view,
            source_table="statistics_short_term",
            old_statistic_id=old_entity_id,
            new_statistic_id=new_entity_id,
            interval_seconds=300,
            statistics_kind=statistics_kind,
            new_entity_started_from_0=new_entity_started_from_0,
        )
    except DbSummaryError as exc:
        print(f"Stage 4 skipped: {exc}")
        return Stage4Result(ran=False, sql_path=None)

    sql_path = Path.cwd() / "update.sql"
    sql_script = build_statistics_update_sql_script(
        conn,
        old_entity_id=old_entity_id,
        new_entity_id=new_entity_id,
        stats_view=generated_stats_view,
        stats_st_view=generated_st_view,
        statistics_kind=statistics_kind,
        new_entity_started_from_0=new_entity_started_from_0,
    )
    sql_path.write_text(sql_script, encoding="utf-8")
    print(f"Stage 4 update SQL written: {sql_path}")

    if apply:
        try:
            apply_sql_script(db_path, sql_script)
        except Exception as exc:
            print(f"Stage 4 apply failed: {exc}")
        else:
            print(f"Stage 4 apply succeeded: updated {db_path}")

    # Generated statistics analysis
    reset_rows: list[dict[str, str]] = []
    if statistics_kind in {"total_increasing", "total"}:
        reset_rows.extend(
            collect_reset_events_statistics(conn, generated_stats_view, new_entity_id, state_class=statistics_kind)
        )
        reset_rows.extend(
            collect_reset_events_statistics(conn, generated_st_view, new_entity_id, state_class=statistics_kind)
        )

    if reset_rows:
        print("Generated statistics reset events report:")
        reset_rows.sort(key=lambda r: float(r.get("event_epoch", "inf") or "inf"))

        def split_ts_val(cell: str) -> tuple[str, str]:
            if not cell:
                return "", ""
            if cell.endswith(")") and " (" in cell:
                ts, val = cell.rsplit(" (", 1)
                return ts, val[:-1]
            return cell, ""

        headers = ["entity", "table", "range"]
        rows: list[list[str]] = []
        for r in reset_rows:
            before_ts, before_val = split_ts_val(r.get("before", ""))
            after_ts, after_val = split_ts_val(r.get("after", ""))
            ts_line = f"{before_ts} - {after_ts}".strip()
            val_line = "" if (before_val == "" and after_val == "") else f"{before_val} - {after_val}"
            cell = ts_line if not val_line else f"{ts_line}\n{val_line}"
            rows.append([r.get("entity", ""), r.get("table", ""), cell])
        print(render_simple_table(headers=headers, rows=rows, color=color, color_code="35"), end="")

    gap_rows: list[dict[str, str]] = []
    gap_rows.extend(
        collect_missing_statistics_row_ranges(conn, generated_stats_view, new_entity_id, interval_seconds=3600)
    )
    gap_rows.extend(
        collect_missing_statistics_row_ranges(conn, generated_st_view, new_entity_id, interval_seconds=300)
    )
    if gap_rows:
        print("Generated statistics missing-rows report:")
        gap_rows.sort(key=lambda r: float(r.get("gap_start_epoch", "inf")))
        headers = ["entity", "table", "gap"]
        rows = [[r["entity"], r["table"], r["gap"]] for r in gap_rows]
        print(render_simple_table(headers=headers, rows=rows, color=color, color_code="36"), end="")

    print("Generated statistics row counts report:")
    try:
        n_long = conn.execute(f"SELECT COUNT(*) AS c FROM {generated_stats_view}").fetchone()["c"]
        n_short = conn.execute(f"SELECT COUNT(*) AS c FROM {generated_st_view}").fetchone()["c"]
    except Exception:
        n_long = None
        n_short = None

    rows = [
        ["statistics", "" if n_long is None else str(n_long)],
        ["statistics_short_term", "" if n_short is None else str(n_short)],
    ]
    print(render_simple_table(headers=["table", "row_count"], rows=rows, color=color, color_code="32"), end="")

    stats_preview = collect_generated_statistics_preview_rows(
        conn,
        source_table="statistics",
        old_entity_id=old_entity_id,
        new_entity_id=new_entity_id,
        generated_view=generated_stats_view,
        first_generated=3,
        last_generated=3,
    )
    st_preview = collect_generated_statistics_preview_rows(
        conn,
        source_table="statistics_short_term",
        old_entity_id=old_entity_id,
        new_entity_id=new_entity_id,
        generated_view=generated_st_view,
        first_generated=3,
        last_generated=3,
    )

    if stats_preview or st_preview:
        print("Generated statistics rows report (sample):")

    def _print_preview(title: str, preview: list[dict[str, str]]) -> None:
        if not preview:
            return
        print(title)
        if statistics_kind in {"total_increasing", "total"}:
            headers = ["ts", "state", "sum"]
            table_rows = [[r["ts"], r["state"], r["sum"]] for r in preview]
        else:
            headers = ["ts", "min", "mean", "max"]
            table_rows = [[r["ts"], r.get("min", ""), r.get("mean", ""), r.get("max", "")] for r in preview]
        print(render_simple_table(headers=headers, rows=table_rows, color=color, color_code="32"), end="")

    _print_preview("=== GENERATED PREVIEW: statistics ===", stats_preview)
    _print_preview("=== GENERATED PREVIEW: statistics_short_term ===", st_preview)

    return Stage4Result(ran=True, sql_path=sql_path)
