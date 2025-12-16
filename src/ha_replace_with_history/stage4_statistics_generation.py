from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .db_summary import (
    DbSummaryError,
    apply_sql_script,
    build_statistics_change_report,
    build_statistics_change_report_with_epochs,
    build_statistics_update_sql_script,
    collect_generated_statistics_preview_rows,
    collect_missing_statistics_row_ranges,
    collect_reset_events_statistics,
    connect_readonly_sqlite,
    create_statistics_generated_view,
    get_earliest_state_ts_epoch,
    get_latest_statistics_ts_epoch,
    snapshot_statistics_rows,
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

    # Stage 5 diffs should only consider newly generated rows (based on new entity states),
    # excluding the portion copied from the old entity statistics.
    cutoff_long = old_stats_latest
    cutoff_short = old_stats_st_latest

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

    def _condense_epoch_rows(
        headers: list[str],
        epoch_rows: list[tuple[float, list[str]]],
        *,
        interval_seconds: int,
    ) -> list[list[str]]:
        if not epoch_rows:
            return []

        def is_contiguous(prev_epoch: float, next_epoch: float) -> bool:
            return abs((next_epoch - prev_epoch) - float(interval_seconds)) < 1e-6

        blocks: list[list[tuple[float, list[str]]]] = []
        cur: list[tuple[float, list[str]]] = [epoch_rows[0]]
        for e, r in epoch_rows[1:]:
            prev_e = cur[-1][0]
            if is_contiguous(prev_e, e):
                cur.append((e, r))
            else:
                blocks.append(cur)
                cur = [(e, r)]
        blocks.append(cur)

        out: list[list[str]] = []
        dots_row = ["..."] + ["..."] * (len(headers) - 1)

        for b in blocks:
            if len(b) <= 6:
                out.extend([r for _, r in b])
                continue
            out.extend([r for _, r in b[:3]])
            out.append(dots_row)
            out.extend([r for _, r in b[-3:]])
        return out

    def format_diff_condensed(
        title: str,
        before_snap,
        after_snap,
        *,
        interval_seconds: int,
        code: str,
    ) -> str:
        b_ts_col, b_ts_is_sec, b_rows = before_snap
        a_ts_col, a_ts_is_sec, a_rows = after_snap
        if not b_rows and not a_rows:
            return ""
        if b_ts_col is None or a_ts_col is None:
            return ""

        headers, epoch_rows = build_statistics_change_report_with_epochs(
            before=b_rows,
            after=a_rows,
            before_ts_col=b_ts_col,
            before_ts_is_seconds=b_ts_is_sec,
            after_ts_col=a_ts_col,
            after_ts_is_seconds=a_ts_is_sec,
        )
        if len(headers) <= 1 or not epoch_rows:
            return ""
        condensed_rows = _condense_epoch_rows(headers, epoch_rows, interval_seconds=interval_seconds)
        if not condensed_rows:
            return ""
        return title + "\n" + render_simple_table(headers=headers, rows=condensed_rows, color=color, color_code=code)

    # Stage 5 (dry-run): compare current rows (would be deleted) vs generated rows (would be inserted).
    # This runs even without --apply.
    before_long_dry = snapshot_statistics_rows(conn, "statistics", new_entity_id, start_epoch_gt=cutoff_long)
    after_long_dry = snapshot_statistics_rows(conn, generated_stats_view, new_entity_id, start_epoch_gt=cutoff_long)
    before_short_dry = snapshot_statistics_rows(
        conn, "statistics_short_term", new_entity_id, start_epoch_gt=cutoff_short
    )
    after_short_dry = snapshot_statistics_rows(conn, generated_st_view, new_entity_id, start_epoch_gt=cutoff_short)

    planned_stage5_long = format_diff_condensed(
        "Statistics changes (long-term):",
        before_long_dry,
        after_long_dry,
        interval_seconds=3600,
        code="33",
    )
    planned_stage5_short = format_diff_condensed(
        "Statistics changes (short-term):",
        before_short_dry,
        after_short_dry,
        interval_seconds=300,
        code="33",
    )

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

    applied_ok = False
    before_long: tuple[str | None, bool, dict[float, dict[str, object]]] | None = None
    before_short: tuple[str | None, bool, dict[float, dict[str, object]]] | None = None
    applied_stage5 = ""
    if apply:
        # Snapshot pre-apply rows for Stage 5.
        before_long = snapshot_statistics_rows(conn, "statistics", new_entity_id, start_epoch_gt=cutoff_long)
        before_short = snapshot_statistics_rows(
            conn, "statistics_short_term", new_entity_id, start_epoch_gt=cutoff_short
        )

        try:
            apply_sql_script(db_path, sql_script)
        except Exception as exc:
            print(f"Stage 4 apply failed: {exc}")
        else:
            applied_ok = True
            print(f"Stage 4 apply succeeded: updated {db_path}")

    if applied_ok and before_long is not None:
        after_conn = connect_readonly_sqlite(db_path)
        try:
            after_long = snapshot_statistics_rows(after_conn, "statistics", new_entity_id, start_epoch_gt=cutoff_long)
            after_short = snapshot_statistics_rows(
                after_conn, "statistics_short_term", new_entity_id, start_epoch_gt=cutoff_short
            )
        finally:
            after_conn.close()

        applied_stage5_long = format_diff_condensed(
            "Statistics changes (long-term):",
            before_long,
            after_long,
            interval_seconds=3600,
            code="33",
        )
        applied_stage5_short = ""
        if before_short is not None:
            applied_stage5_short = format_diff_condensed(
                "Statistics changes (short-term):",
                before_short,
                after_short,
                interval_seconds=300,
                code="33",
            )

        applied_stage5 = "".join([applied_stage5_long, applied_stage5_short])

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

    # Print Stage 5 reports last.
    if planned_stage5_long or planned_stage5_short:
        print("*** Stage 5: Planned statistics diff")
        if planned_stage5_long:
            print(planned_stage5_long, end="")
        if planned_stage5_short:
            print(planned_stage5_short, end="")
    if applied_stage5:
        print("*** Stage 5: Applied statistics diff")
        print(applied_stage5, end="")

    return Stage4Result(ran=True, sql_path=sql_path)
