from __future__ import annotations

import sqlite3

from .db_summary import (
    collect_missing_statistics_row_ranges,
    collect_reset_events_statistics,
    collect_reset_events_states,
)
from .report import render_entity_registry_report, render_simple_table


def run_statistics_analysis(
    conn: sqlite3.Connection,
    *,
    old_entity_id: str,
    new_entity_id: str,
    old_total_like: bool,
    new_total_like: bool,
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

    # Reset events apply to total-like sensors (total_increasing and total).
    state_reset_rows: list[dict[str, str]] = []
    if old_total_like:
        state_reset_rows.extend(collect_reset_events_states(conn, old_entity_id))
    if new_total_like:
        state_reset_rows.extend(collect_reset_events_states(conn, new_entity_id))

    stats_reset_rows: list[dict[str, str]] = []
    if old_total_like:
        stats_reset_rows.extend(collect_reset_events_statistics(conn, "statistics", old_entity_id))
        stats_reset_rows.extend(collect_reset_events_statistics(conn, "statistics_short_term", old_entity_id))
    if new_total_like:
        stats_reset_rows.extend(collect_reset_events_statistics(conn, "statistics", new_entity_id))
        stats_reset_rows.extend(collect_reset_events_statistics(conn, "statistics_short_term", new_entity_id))

    # Print two combined reset tables:
    # 1) states + statistics
    # 2) states + statistics_short_term
    if state_reset_rows or stats_reset_rows:
        stats_rows = [r for r in stats_reset_rows if r.get("table") == "statistics"]
        st_rows = [r for r in stats_reset_rows if r.get("table") == "statistics_short_term"]

        def print_combined(title: str, combined: list[dict[str, str]]) -> None:
            if not combined:
                return
            print(title)
            combined.sort(key=lambda r: float(r.get("event_epoch", "inf") or "inf"))
            headers = ["entity", "table", "before", "after", "last_reset"]
            rows = [
                [r["entity"], r["table"], r["before"], r["after"], r.get("last_reset", "")]
                for r in combined
            ]
            print(render_simple_table(headers=headers, rows=rows, color=color, color_code="35"), end="")

        print_combined(
            "Reset events report (states + statistics):",
            [*state_reset_rows, *stats_rows],
        )
        print_combined(
            "Reset events report (states + statistics_short_term):",
            [*state_reset_rows, *st_rows],
        )

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
