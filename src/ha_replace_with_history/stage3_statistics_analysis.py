from __future__ import annotations

import sqlite3

from .db_summary import (
    collect_missing_statistics_row_ranges,
    collect_reset_events_statistics,
)
from .report import render_entity_registry_report, render_simple_table


def run_statistics_analysis(
    conn: sqlite3.Connection,
    *,
    old_entity_id: str,
    new_entity_id: str,
    old_total_increasing: bool,
    new_total_increasing: bool,
    old_summary: dict[str, object],
    new_summary: dict[str, object],
    tick: str,
    color: bool,
) -> None:
    print("*** Stage 3: Statistics analysis")

    print("Statistics analysis report:")
    stats_report = render_entity_registry_report(
        old_entity_id=old_entity_id,
        new_entity_id=new_entity_id,
        old={
            "statistics": old_summary.get("statistics"),
            "statistics_short_term": old_summary.get("statistics_short_term"),
        },
        new={
            "statistics": new_summary.get("statistics"),
            "statistics_short_term": new_summary.get("statistics_short_term"),
        },
        tick=tick,
        color=color,
    )
    print(stats_report, end="")

    # Reset events only apply to total_increasing sensors.
    reset_rows: list[dict[str, str]] = []
    if old_total_increasing:
        reset_rows.extend(collect_reset_events_statistics(conn, "statistics", old_entity_id))
        reset_rows.extend(collect_reset_events_statistics(conn, "statistics_short_term", old_entity_id))
    if new_total_increasing:
        reset_rows.extend(collect_reset_events_statistics(conn, "statistics", new_entity_id))
        reset_rows.extend(collect_reset_events_statistics(conn, "statistics_short_term", new_entity_id))

    if reset_rows:
        print("Statistics reset events report:")
        reset_rows.sort(key=lambda r: float(r.get("event_epoch", "inf") or "inf"))
        headers = ["entity", "table", "before", "after"]
        rows = [[r["entity"], r["table"], r["before"], r["after"]] for r in reset_rows]
        print(render_simple_table(headers=headers, rows=rows, color=color, color_code="35"), end="")

    gap_rows: list[dict[str, str]] = []
    for entity_id in (old_entity_id, new_entity_id):
        gap_rows.extend(collect_missing_statistics_row_ranges(conn, "statistics", entity_id, interval_seconds=3600))
        gap_rows.extend(
            collect_missing_statistics_row_ranges(conn, "statistics_short_term", entity_id, interval_seconds=300)
        )

    if gap_rows:
        print("Missing statistics rows report:")
        gap_rows.sort(key=lambda r: float(r.get("gap_start_epoch", "inf")))
        headers = ["entity", "table", "gap"]
        rows = [[r["entity"], r["table"], r["gap"]] for r in gap_rows]
        print(render_simple_table(headers=headers, rows=rows, color=color, color_code="36"), end="")
