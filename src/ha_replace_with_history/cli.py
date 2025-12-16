from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .db_summary import DbSummaryError, connect_readonly_sqlite
from .stage1_entity_analysis import run_entity_analysis
from .stage2_state_analysis import run_state_analysis
from .stage3_statistics_analysis import run_statistics_analysis
from .stage4_statistics_generation import run_statistics_generation


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ha-replace-with-history",
        description=(
            "Analyze and rebuild Home Assistant entity history. "
            "Stages: 1) entity analysis 2) state analysis 3) statistics analysis 4) statistics generation."
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
            "If true (default), treat the first reading as a restart from 0; if false, don't add it to the first sum."
        ),
    )

    p.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Execute the generated update SQL against --db and commit changes (still writes update.sql)."
        ),
    )

    p.add_argument("old_entity_id")
    p.add_argument("new_entity_id")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    storage_dir = Path(args.storage)

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

    stage1 = run_entity_analysis(
        storage_dir=storage_dir,
        old_entity_id=args.old_entity_id,
        new_entity_id=args.new_entity_id,
        tick=tick,
        color=use_color,
    )

    old_total_increasing = stage1.old_state_class == "total_increasing"
    new_total_increasing = stage1.new_state_class == "total_increasing"
    old_total_like = stage1.old_state_class in {"total_increasing", "total"}
    new_total_like = stage1.new_state_class in {"total_increasing", "total"}

    db_exit: int = 0
    try:
        conn = connect_readonly_sqlite(Path(args.db))
    except DbSummaryError as exc:
        print(f"DB analysis skipped: {exc}", file=sys.stderr)
        db_exit = 3
        conn = None

    if conn is not None:
        old_summary, new_summary = run_state_analysis(
            conn,
            old_entity_id=args.old_entity_id,
            new_entity_id=args.new_entity_id,
            old_total_increasing=old_total_increasing,
            new_total_increasing=new_total_increasing,
            tick=tick,
            color=use_color,
        )

        run_statistics_analysis(
            conn,
            old_entity_id=args.old_entity_id,
            new_entity_id=args.new_entity_id,
            old_total_like=old_total_like,
            new_total_like=new_total_like,
            old_summary=old_summary,
            new_summary=new_summary,
            tick=tick,
            color=use_color,
        )

        run_statistics_generation(
            conn,
            db_path=Path(args.db),
            old_entity_id=args.old_entity_id,
            new_entity_id=args.new_entity_id,
            old_state_class=stage1.old_state_class,
            new_state_class=stage1.new_state_class,
            new_entity_started_from_0=args.new_entity_started_from_0,
            apply=args.apply,
            color=use_color,
        )

    # Non-zero exit if either entity is missing (still prints the report).
    if stage1.old_entity is None or stage1.new_entity is None:
        return 2
    return db_exit
