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
    old_state_class: str | None,
    new_state_class: str | None,
    old_summary: dict[str, object],
    new_summary: dict[str, object],
    report_stats: bool,
    report_stats_short_term: bool,
    tick: str,
    color: bool,
) -> None:
    print("*** Stage 3: Statistics analysis")

    if not report_stats and not report_stats_short_term:
        print("Stage 3 skipped: statistics reporting disabled.")
        return

    print("Statistics analysis report:")
    old_payload: dict[str, object] = {}
    new_payload: dict[str, object] = {}
    if report_stats:
        old_payload["statistics"] = old_summary.get("statistics")
        new_payload["statistics"] = new_summary.get("statistics")
    if report_stats_short_term:
        old_payload["statistics_short_term"] = old_summary.get("statistics_short_term")
        new_payload["statistics_short_term"] = new_summary.get("statistics_short_term")

    stats_report = render_entity_registry_report(
        old_entity_id=old_entity_id,
        new_entity_id=new_entity_id,
        old=old_payload,
        new=new_payload,
        tick=tick,
        color=color,
    )
    print(stats_report, end="")

    # Reset events apply to total-like sensors (total_increasing and total).
    old_total_like = old_state_class in {"total_increasing", "total"}
    new_total_like = new_state_class in {"total_increasing", "total"}

    state_reset_rows: list[dict[str, str]] = []
    if old_total_like:
        state_reset_rows.extend(collect_reset_events_states(conn, old_entity_id, state_class=old_state_class))
    if new_total_like:
        state_reset_rows.extend(collect_reset_events_states(conn, new_entity_id, state_class=new_state_class))

    stats_reset_rows: list[dict[str, str]] = []
    if report_stats:
        if old_total_like:
            stats_reset_rows.extend(
                collect_reset_events_statistics(conn, "statistics", old_entity_id, state_class=old_state_class)
            )
        if new_total_like:
            stats_reset_rows.extend(
                collect_reset_events_statistics(conn, "statistics", new_entity_id, state_class=new_state_class)
            )
    if report_stats_short_term:
        if old_total_like:
            stats_reset_rows.extend(
                collect_reset_events_statistics(
                    conn, "statistics_short_term", old_entity_id, state_class=old_state_class
                )
            )
        if new_total_like:
            stats_reset_rows.extend(
                collect_reset_events_statistics(
                    conn, "statistics_short_term", new_entity_id, state_class=new_state_class
                )
            )

    # Print two combined reset tables:
    # 1) states + statistics
    # 2) states + statistics_short_term
    if state_reset_rows or stats_reset_rows:
        stats_rows = [r for r in stats_reset_rows if r.get("table") == "statistics"]
        st_rows = [r for r in stats_reset_rows if r.get("table") == "statistics_short_term"]

        def split_ts_val(cell: str) -> tuple[str, str]:
            if not cell:
                return "", ""
            if cell.endswith(")") and " (" in cell:
                ts, val = cell.rsplit(" (", 1)
                return ts, val[:-1]
            return cell, ""

        def range_cell(before: str, after: str) -> str:
            before_ts, before_val = split_ts_val(before)
            after_ts, after_val = split_ts_val(after)
            ts_line = f"{before_ts} - {after_ts}".strip()
            val_line = "" if (before_val == "" and after_val == "") else f"{before_val} - {after_val}"
            return ts_line if not val_line else f"{ts_line}\n{val_line}"

        def print_combined(title: str, combined: list[dict[str, str]]) -> None:
            if not combined:
                return
            print(title)
            combined.sort(key=lambda r: float(r.get("event_epoch", "inf") or "inf"))
            headers = ["entity", "table", "range", "last_reset"]
            rows: list[list[str]] = []
            for r in combined:
                rows.append(
                    [
                        r.get("entity", ""),
                        r.get("table", ""),
                        range_cell(r.get("before", ""), r.get("after", "")),
                        r.get("last_reset", ""),
                    ]
                )
            print(render_simple_table(headers=headers, rows=rows, color=color, color_code="35"), end="")

        if report_stats:
            print_combined(
                "Reset events report (states + statistics):",
                [*state_reset_rows, *stats_rows],
            )
        if report_stats_short_term:
            print_combined(
                "Reset events report (states + statistics_short_term):",
                [*state_reset_rows, *st_rows],
            )

    gap_rows: list[dict[str, str]] = []
    for entity_id in (old_entity_id, new_entity_id):
        if report_stats:
            gap_rows.extend(
                collect_missing_statistics_row_ranges(conn, "statistics", entity_id, interval_seconds=3600)
            )
        if report_stats_short_term:
            gap_rows.extend(
                collect_missing_statistics_row_ranges(conn, "statistics_short_term", entity_id, interval_seconds=300)
            )

    if gap_rows:
        print("Missing statistics rows report:")
        gap_rows.sort(key=lambda r: float(r.get("gap_start_epoch", "inf")))
        headers = ["entity", "table", "gap"]
        rows = [[r["entity"], r["table"], r["gap"]] for r in gap_rows]
        print(render_simple_table(headers=headers, rows=rows, color=color, color_code="36"), end="")
