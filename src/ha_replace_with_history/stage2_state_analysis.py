from __future__ import annotations

import sqlite3

from .db_summary import (
    collect_reset_events_states,
    collect_unavailable_occurrence_rows,
    summarize_all,
)
from .report import render_entity_registry_report, render_simple_table, render_unavailable_occurrences_report


def run_state_analysis(
    conn: sqlite3.Connection,
    *,
    old_entity_id: str,
    new_entity_id: str,
    old_total_increasing: bool,
    new_total_increasing: bool,
    tick: str,
    color: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    print("*** Stage 2: State analysis")

    with conn:
        old_summary = summarize_all(conn, old_entity_id, total_increasing=old_total_increasing)
        new_summary = summarize_all(conn, new_entity_id, total_increasing=new_total_increasing)

    print("State analysis report:")
    state_report = render_entity_registry_report(
        old_entity_id=old_entity_id,
        new_entity_id=new_entity_id,
        old={"states": old_summary.get("states")},
        new={"states": new_summary.get("states")},
        tick=tick,
        color=color,
    )
    print(state_report, end="")

    # Unavailable occurrences report (states only)
    occ_rows: list[dict[str, str]] = []
    occ_rows.extend(
        [r for r in collect_unavailable_occurrence_rows(conn, old_entity_id) if r["table"] == "states"]
    )
    occ_rows.extend(
        [r for r in collect_unavailable_occurrence_rows(conn, new_entity_id) if r["table"] == "states"]
    )
    if occ_rows:
        print("Unavailable occurrences report:")
        print(render_unavailable_occurrences_report(rows=occ_rows, color=color), end="")

    # Reset events only apply to total_increasing sensors.
    reset_rows: list[dict[str, str]] = []
    if old_total_increasing:
        reset_rows.extend(collect_reset_events_states(conn, old_entity_id))
    if new_total_increasing:
        reset_rows.extend(collect_reset_events_states(conn, new_entity_id))

    if reset_rows:
        print("State reset events report:")
        reset_rows.sort(key=lambda r: float(r.get("event_epoch", "inf") or "inf"))
        headers = ["entity", "table", "before", "after"]
        rows = [[r["entity"], r["table"], r["before"], r["after"]] for r in reset_rows]
        print(render_simple_table(headers=headers, rows=rows, color=color, color_code="35"), end="")

    return old_summary, new_summary
