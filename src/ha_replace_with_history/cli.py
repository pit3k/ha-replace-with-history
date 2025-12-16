from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .db_summary import (
    DbSummaryError,
    apply_sql_script,
    build_stage3_update_sql_script,
    collect_stage3_statistics_preview_rows,
    create_stage3_statistics_view,
    collect_missing_statistics_row_ranges,
    collect_reset_events_statistics,
    collect_reset_events_states,
    collect_unavailable_occurrence_rows,
    connect_readonly_sqlite,
    get_earliest_state_ts_epoch,
    get_latest_statistics_ts_epoch,
    summarize_all,
)
from .entity_registry import EntityRegistryError, get_entity, load_entity_registry
from .report import render_entity_registry_report, render_simple_table, render_unavailable_occurrences_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ha-replace-with-history",
        description=(
            "Check/fix Home Assistant entity states/statistics "
            "(stage 1: entity registry report; stage 2: DB summaries)."
        ),
    )

    p.add_argument(
        "--db",
        default="./home-assistant_v2.db",
        help="Path to Home Assistant SQLite DB (default: ./home-assistant_v2.db).",
    )
    p.add_argument(
        "--storage",
        default="./.storage/",
        help="Path to Home Assistant storage dir (default: ./.storage/).",
    )
    p.add_argument(
        "--entity-registry-file",
        default=None,
        help=(
            "Override path to entity registry file. If relative, it's resolved relative to --storage. "
            "Default: <storage>/core.entity_registry"
        ),
    )

    def parse_bool(value: str) -> bool:
        v = value.strip().lower()
        if v in {"true", "t", "1", "yes", "y", "on"}:
            return True
        if v in {"false", "f", "0", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError("Expected true/false")

    p.add_argument(
        "--new-entity-started-from-0",
        type=parse_bool,
        default=True,
        metavar="{true|false}",
        help=(
            "Stage 3 only: if true (default), treat the new entity as having restarted from 0 at first sample, "
            "so the first generated sum increases by the full first reading; if false, keep current behavior."
        ),
    )

    p.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Stage 3 only: execute the generated Stage 3 update SQL against --db and commit the changes. "
            "(The SQL file is still written for review.)"
        ),
    )

    p.add_argument("old_entity_id")
    p.add_argument("new_entity_id")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    storage_dir = Path(args.storage)
    entity_registry_file = Path(args.entity_registry_file) if args.entity_registry_file else None

    print("*** Looking up entity registry")
    try:
        registry = load_entity_registry(storage_dir=storage_dir, entity_registry_file=entity_registry_file)
    except EntityRegistryError as exc:
        raise SystemExit(str(exc))

    old = get_entity(registry, args.old_entity_id)
    new = get_entity(registry, args.new_entity_id)

    def get_state_class(entity: dict[str, object] | None) -> str | None:
        if not entity:
            return None
        caps = entity.get("capabilities")
        if isinstance(caps, dict):
            sc = caps.get("state_class")
            if isinstance(sc, str) and sc:
                return sc
        for key in ("state_class", "original_state_class"):
            v = entity.get(key)
            if isinstance(v, str) and v:
                return v
        return None

    old_state_class = get_state_class(old)
    new_state_class = get_state_class(new)
    old_total_increasing = old_state_class == "total_increasing"
    new_total_increasing = new_state_class == "total_increasing"
    old_measurement = old_state_class == "measurement"
    new_measurement = new_state_class == "measurement"

    tick = "✓"
    encoding = sys.stdout.encoding or "utf-8"
    for candidate in ("✓", "√", "v"):
        try:
            candidate.encode(encoding)
            tick = candidate
            break
        except Exception:
            continue

    use_color = bool(sys.stdout.isatty()) and ("NO_COLOR" not in os.environ)

    print("Entity details report:")
    report = render_entity_registry_report(
        old_entity_id=args.old_entity_id,
        new_entity_id=args.new_entity_id,
        old=old,
        new=new,
        tick=tick,
        color=use_color,
    )
    print(report, end="")

    db_exit: int = 0
    try:
        conn = connect_readonly_sqlite(Path(args.db))
    except DbSummaryError as exc:
        print(f"DB summary skipped: {exc}", file=sys.stderr)
        db_exit = 3
        conn = None

    if conn is not None:
        print("*** Analyzing states")
        with conn:
            old_summary = summarize_all(conn, args.old_entity_id, total_increasing=old_total_increasing)
            new_summary = summarize_all(conn, args.new_entity_id, total_increasing=new_total_increasing)

        # State analysis report (states section only)
        print("State analysis report:")
        state_report = render_entity_registry_report(
            old_entity_id=args.old_entity_id,
            new_entity_id=args.new_entity_id,
            old={"states": old_summary.get("states")},
            new={"states": new_summary.get("states")},
            tick=tick,
            color=use_color,
        )
        print(state_report, end="")

        # Statistics analysis report (statistics sections only)
        print("Statistics analysis report:")
        stats_report = render_entity_registry_report(
            old_entity_id=args.old_entity_id,
            new_entity_id=args.new_entity_id,
            old={
                "statistics": old_summary.get("statistics"),
                "statistics_short_term": old_summary.get("statistics_short_term"),
            },
            new={
                "statistics": new_summary.get("statistics"),
                "statistics_short_term": new_summary.get("statistics_short_term"),
            },
            tick=tick,
            color=use_color,
        )
        print(stats_report, end="")

        # Unavailable occurrences report (states only)
        occ_rows: list[dict[str, str]] = []
        occ_rows.extend([r for r in collect_unavailable_occurrence_rows(conn, args.old_entity_id) if r["table"] == "states"])
        occ_rows.extend([r for r in collect_unavailable_occurrence_rows(conn, args.new_entity_id) if r["table"] == "states"])
        if occ_rows:
            print("Unavailable occurrences report:")
            print(render_unavailable_occurrences_report(rows=occ_rows, color=use_color), end="")

        print("*** Rebuilding and analyzing statistics")

        stage3_enabled = False
        stage3_stats_table = "stage3_statistics"
        stage3_st_table = "stage3_statistics_short_term"
        stage3_sql_path: Path | None = None
        stage3_sql_script: str | None = None

        # Preconditions for stage3:
        # 1) new states earliest > old stats latest
        # 2) both entities share a supported state_class (total_increasing or measurement)
        if old_total_increasing and new_total_increasing:
            stage3_kind = "total_increasing"
        elif old_measurement and new_measurement:
            stage3_kind = "measurement"
        else:
            stage3_kind = None

        if stage3_kind is None:
            print("Stage 3 skipped: both entities must be total_increasing or measurement (and match).")
        else:
            new_states_earliest = get_earliest_state_ts_epoch(conn, args.new_entity_id)
            old_stats_latest = get_latest_statistics_ts_epoch(conn, "statistics", args.old_entity_id)
            old_stats_st_latest = get_latest_statistics_ts_epoch(conn, "statistics_short_term", args.old_entity_id)
            old_stats_latest_any = max(
                [v for v in (old_stats_latest, old_stats_st_latest) if v is not None],
                default=None,
            )

            if new_states_earliest is None or old_stats_latest_any is None:
                print("Stage 3 skipped: missing states/statistics timestamps for precondition checks.")
            elif not (new_states_earliest > old_stats_latest_any):
                print("Stage 3 skipped: precondition failed (new states earliest must be after old statistics latest).")
            else:
                try:
                    create_stage3_statistics_view(
                        conn,
                        view_name=stage3_stats_table,
                        source_table="statistics",
                        old_statistic_id=args.old_entity_id,
                        new_statistic_id=args.new_entity_id,
                        interval_seconds=3600,
                        statistics_kind=stage3_kind,
                        new_entity_started_from_0=args.new_entity_started_from_0,
                    )
                    create_stage3_statistics_view(
                        conn,
                        view_name=stage3_st_table,
                        source_table="statistics_short_term",
                        old_statistic_id=args.old_entity_id,
                        new_statistic_id=args.new_entity_id,
                        interval_seconds=300,
                        statistics_kind=stage3_kind,
                        new_entity_started_from_0=args.new_entity_started_from_0,
                    )
                except DbSummaryError as exc:
                    print(f"Stage 3 skipped: {exc}")
                else:
                    stage3_enabled = True

                    # Write an update script for applying the rebuilt stats to the DB.
                    stage3_sql_path = Path.cwd() / "update.sql"
                    stage3_sql_script = build_stage3_update_sql_script(
                        conn,
                        old_entity_id=args.old_entity_id,
                        new_entity_id=args.new_entity_id,
                        stats_view=stage3_stats_table,
                        stats_st_view=stage3_st_table,
                        statistics_kind=stage3_kind,
                        new_entity_started_from_0=args.new_entity_started_from_0,
                    )
                    stage3_sql_path.write_text(stage3_sql_script, encoding="utf-8")

        # Reset events + last_reset reports only apply to total_increasing sensors.
        reset_rows: list[dict[str, str]] = []
        if old_total_increasing:
            reset_rows.extend(collect_reset_events_states(conn, args.old_entity_id))
            reset_rows.extend(collect_reset_events_statistics(conn, "statistics", args.old_entity_id))
            reset_rows.extend(collect_reset_events_statistics(conn, "statistics_short_term", args.old_entity_id))
        if new_total_increasing:
            reset_rows.extend(collect_reset_events_states(conn, args.new_entity_id))
            reset_rows.extend(collect_reset_events_statistics(conn, "statistics", args.new_entity_id))
            reset_rows.extend(collect_reset_events_statistics(conn, "statistics_short_term", args.new_entity_id))
        if stage3_enabled and stage3_kind == "total_increasing":
            reset_rows.extend(collect_reset_events_statistics(conn, stage3_stats_table, args.new_entity_id))
            reset_rows.extend(collect_reset_events_statistics(conn, stage3_st_table, args.new_entity_id))

        if reset_rows:
            print("Reset events report:")
            reset_rows.sort(key=lambda r: float(r.get("event_epoch", "inf") or "inf"))
            headers = ["entity", "table", "before", "after"]
            rows = [[r["entity"], r["table"], r["before"], r["after"]] for r in reset_rows]
            print(render_simple_table(headers=headers, rows=rows, color=use_color, color_code="35"), end="")

        gap_rows: list[dict[str, str]] = []
        for entity_id in (args.old_entity_id, args.new_entity_id):
            gap_rows.extend(
                collect_missing_statistics_row_ranges(conn, "statistics", entity_id, interval_seconds=3600)
            )
            gap_rows.extend(
                collect_missing_statistics_row_ranges(conn, "statistics_short_term", entity_id, interval_seconds=300)
            )
        if stage3_enabled:
            gap_rows.extend(
                collect_missing_statistics_row_ranges(conn, stage3_stats_table, args.new_entity_id, interval_seconds=3600)
            )
            gap_rows.extend(
                collect_missing_statistics_row_ranges(conn, stage3_st_table, args.new_entity_id, interval_seconds=300)
            )

        if gap_rows:
            print("Missing stats report:")
            gap_rows.sort(key=lambda r: float(r.get("gap_start_epoch", "inf")))
            headers = ["entity", "table", "gap"]
            rows = [[r["entity"], r["table"], r["gap"]] for r in gap_rows]
            print(render_simple_table(headers=headers, rows=rows, color=use_color, color_code="36"), end="")

        if stage3_enabled:
            gen_rows: list[list[str]] = []
            try:
                n_long = conn.execute(
                    f"SELECT COUNT(*) AS c FROM {stage3_stats_table}",
                ).fetchone()["c"]
                n_short = conn.execute(
                    f"SELECT COUNT(*) AS c FROM {stage3_st_table}",
                ).fetchone()["c"]
            except Exception:
                n_long = None
                n_short = None

            gen_rows.append(["statistics", "" if n_long is None else str(n_long)])
            gen_rows.append(["statistics_short_term", "" if n_short is None else str(n_short)])

            print("Generated stats report:")
            print(
                render_simple_table(
                    headers=["table", "row_count"],
                    rows=gen_rows,
                    color=use_color,
                    color_code="32",
                ),
                end="",
            )

        if stage3_sql_path is not None:
            print(f"Stage 3 update SQL written: {stage3_sql_path}")

        if stage3_sql_path is not None and stage3_sql_script is not None and args.apply:
            try:
                apply_sql_script(Path(args.db), stage3_sql_script)
            except Exception as exc:
                print(f"Stage 3 apply failed: {exc}")
            else:
                print(f"Stage 3 apply succeeded: updated {args.db}")

        if stage3_enabled:
            stats_preview = collect_stage3_statistics_preview_rows(
                conn,
                source_table="statistics",
                old_entity_id=args.old_entity_id,
                new_entity_id=args.new_entity_id,
                stage3_view=stage3_stats_table,
                first_generated=3,
                last_generated=3,
            )
            st_preview = collect_stage3_statistics_preview_rows(
                conn,
                source_table="statistics_short_term",
                old_entity_id=args.old_entity_id,
                new_entity_id=args.new_entity_id,
                stage3_view=stage3_st_table,
                first_generated=3,
                last_generated=3,
            )

            if stats_preview or st_preview:
                print("Generated statistics rows report (sample):")
            if stats_preview:
                print("=== STAGE 3 PREVIEW: statistics ===")
                print(
                    render_simple_table(
                        headers=(
                            ["ts", "state", "sum"]
                            if stage3_kind == "total_increasing"
                            else ["ts", "min", "mean", "max"]
                        ),
                        rows=(
                            [[r["ts"], r["state"], r["sum"]] for r in stats_preview]
                            if stage3_kind == "total_increasing"
                            else [[r["ts"], r.get("min", ""), r.get("mean", ""), r.get("max", "")] for r in stats_preview]
                        ),
                        color=use_color,
                        color_code="32",
                    ),
                    end="",
                )
            if st_preview:
                print("=== STAGE 3 PREVIEW: statistics_short_term ===")
                print(
                    render_simple_table(
                        headers=(
                            ["ts", "state", "sum"]
                            if stage3_kind == "total_increasing"
                            else ["ts", "min", "mean", "max"]
                        ),
                        rows=(
                            [[r["ts"], r["state"], r["sum"]] for r in st_preview]
                            if stage3_kind == "total_increasing"
                            else [[r["ts"], r.get("min", ""), r.get("mean", ""), r.get("max", "")] for r in st_preview]
                        ),
                        color=use_color,
                        color_code="32",
                    ),
                    end="",
                )

    # Non-zero exit if either entity is missing (still prints the report).
    if old is None or new is None:
        return 2
    return db_exit
