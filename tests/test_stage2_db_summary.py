import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from ha_replace_with_history.db_summary import (
    build_statistics_change_report,
    build_statistics_change_report_with_epochs,
    collect_last_reset_rows,
    collect_missing_statistics_row_ranges,
    collect_reset_events_statistics,
    collect_reset_events_states,
    condense_statistics_change_report_rows,
    connect_readonly_sqlite,
    snapshot_statistics_rows,
    summarize_all,
    summarize_statistics,
    summarize_states,
)


class TestStage2DbSummary(unittest.TestCase):
    def _fmt_local(self, ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

    def _make_db(self) -> Path:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        tmp.close()
        return Path(tmp.name)

    def test_states_legacy_schema(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "CREATE TABLE states (entity_id TEXT, state TEXT, last_updated_ts REAL)"
            )
            conn.executemany(
                "INSERT INTO states(entity_id, state, last_updated_ts) VALUES(?,?,?)",
                [
                    ("sensor.old", "1", 100.0),
                    ("sensor.old", "", 200.0),
                    ("sensor.old", None, 300.0),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            summary = summarize_states(ro, "sensor.old")
        finally:
            ro.close()

        self.assertEqual(summary.row_count, 3)
        self.assertEqual(summary.unavailable_count, 2)
        self.assertEqual(summary.oldest_value, "1")
        self.assertEqual(summary.latest_value, None)
        self.assertEqual(summary.oldest_available_value, "1")
        self.assertEqual(summary.latest_available_value, "1")
        self.assertIsNotNone(summary.oldest_ts)
        self.assertIsNotNone(summary.latest_ts)
        # Runs are split by exact value (no <mixed>): empty-string and NULL are separate runs.
        self.assertEqual(len(summary.unavailable_occurrences), 2)

        run1 = summary.unavailable_occurrences[0]
        self.assertEqual(run1.start_ts, self._fmt_local(200.0))
        self.assertEqual(run1.end_ts, self._fmt_local(200.0))
        self.assertEqual(run1.count, 1)
        self.assertEqual(run1.value, "")

        run2 = summary.unavailable_occurrences[1]
        self.assertEqual(run2.start_ts, self._fmt_local(300.0))
        self.assertEqual(run2.end_ts, self._fmt_local(300.0))
        self.assertEqual(run2.count, 1)
        self.assertEqual(run2.value, "NULL")

    def test_states_latest_unavailable_has_different_latest_available(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "CREATE TABLE states (entity_id TEXT, state TEXT, last_updated_ts REAL)"
            )
            conn.executemany(
                "INSERT INTO states(entity_id, state, last_updated_ts) VALUES(?,?,?)",
                [
                    ("sensor.x", "1", 100.0),
                    ("sensor.x", "unavailable", 200.0),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            states = summarize_states(ro, "sensor.x")
            all_summary = summarize_all(ro, "sensor.x")
        finally:
            ro.close()

        self.assertEqual(states.latest_value, "unavailable")
        self.assertEqual(states.latest_available_value, "1")
        self.assertNotEqual(states.latest_available_ts, states.latest_ts)
        self.assertEqual(states.unavailable_count, 1)
        self.assertEqual(len(states.unavailable_occurrences), 1)
        run = states.unavailable_occurrences[0]
        self.assertEqual(run.start_ts, self._fmt_local(200.0))
        self.assertEqual(run.end_ts, self._fmt_local(200.0))
        self.assertEqual(run.count, 1)
        self.assertEqual(run.value, "unavailable")

        # summarize_all no longer embeds occurrences (printed as separate report)
        self.assertNotIn("unavailable_occurrences", all_summary["states"])

    def test_states_metadata_schema(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "CREATE TABLE states_meta (metadata_id INTEGER PRIMARY KEY, entity_id TEXT)"
            )
            conn.execute(
                "CREATE TABLE states (metadata_id INTEGER, state TEXT, last_updated_ts REAL)"
            )
            conn.execute("INSERT INTO states_meta(metadata_id, entity_id) VALUES(1, ?)", ("sensor.new",))
            conn.executemany(
                "INSERT INTO states(metadata_id, state, last_updated_ts) VALUES(?,?,?)",
                [
                    (1, "a", 10.0),
                    (1, "b", 20.0),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            summary = summarize_states(ro, "sensor.new")
        finally:
            ro.close()

        self.assertEqual(summary.row_count, 2)
        self.assertEqual(summary.unavailable_count, 0)
        self.assertEqual(summary.oldest_value, "a")
        self.assertEqual(summary.latest_value, "b")
        self.assertEqual(summary.oldest_available_value, "a")
        self.assertEqual(summary.latest_available_value, "b")
        self.assertEqual(len(summary.unavailable_occurrences), 0)

    def test_statistics_metadata_schema(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "CREATE TABLE statistics_meta (id INTEGER PRIMARY KEY, statistic_id TEXT)"
            )
            conn.execute(
                "CREATE TABLE statistics (metadata_id INTEGER, start_ts REAL, mean REAL, sum REAL, last_reset_ts REAL)"
            )
            conn.execute(
                "CREATE TABLE statistics_short_term (metadata_id INTEGER, start_ts REAL, mean REAL)"
            )
            conn.execute("INSERT INTO statistics_meta(id, statistic_id) VALUES(7, ?)", ("sensor.stat",))
            conn.executemany(
                "INSERT INTO statistics(metadata_id, start_ts, mean, sum, last_reset_ts) VALUES(?,?,?,?,?)",
                [
                    (7, 1000.0, 1.5, 10.0, None),
                    (7, 2000.0, 2.5, 20.0, 1500.0),
                ],
            )
            conn.executemany(
                "INSERT INTO statistics_short_term(metadata_id, start_ts, mean) VALUES(?,?,?)",
                [
                    (7, 3000.0, 3.5),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            summary = summarize_statistics(ro, "statistics", "sensor.stat")
            st_summary = summarize_statistics(ro, "statistics_short_term", "sensor.stat")
            all_summary = summarize_all(ro, "sensor.stat")
        finally:
            ro.close()

        self.assertEqual(summary.row_count, 2)
        self.assertEqual(summary.unavailable_count, 0)
        self.assertEqual(summary.oldest_value, "1.5")
        self.assertEqual(summary.latest_value, "2.5")
        self.assertEqual(summary.oldest_available_value, "1.5")
        self.assertEqual(summary.latest_available_value, "2.5")
        self.assertEqual(summary.oldest_sum, "10.0")
        self.assertEqual(summary.latest_sum, "20.0")
        self.assertEqual(st_summary.row_count, 1)
        self.assertEqual(st_summary.latest_value, "3.5")

        self.assertIn("states", all_summary)
        self.assertIn("statistics", all_summary)
        self.assertIn("statistics_short_term", all_summary)

        # total_increasing formatting includes sum in parentheses for statistics
        ro = connect_readonly_sqlite(db_path)
        try:
            ti = summarize_all(ro, "sensor.stat", total_increasing=True)
        finally:
            ro.close()
        self.assertEqual(ti["statistics"]["earliest"]["value"], "1.5 (10.0)")
        self.assertEqual(ti["statistics"]["latest"]["value"], "2.5 (20.0)")

        # last_reset rows are collected
        ro = connect_readonly_sqlite(db_path)
        try:
            lr = collect_last_reset_rows(ro, "sensor.stat")
        finally:
            ro.close()
        self.assertEqual(len(lr), 1)
        self.assertEqual(lr[0]["table"], "statistics")

    def test_reset_events_states_and_statistics(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE states (entity_id TEXT, state TEXT, last_updated_ts REAL)")
            conn.executemany(
                "INSERT INTO states(entity_id, state, last_updated_ts) VALUES(?,?,?)",
                [
                    ("sensor.r", "5", 100.0),
                    ("sensor.r", "6", 200.0),
                    ("sensor.r", "1", 300.0),
                ],
            )

            conn.execute("CREATE TABLE statistics_meta (id INTEGER PRIMARY KEY, statistic_id TEXT)")
            conn.execute("CREATE TABLE statistics (metadata_id INTEGER, start_ts REAL, sum REAL)")
            conn.execute("INSERT INTO statistics_meta(id, statistic_id) VALUES(1, ?)", ("sensor.r",))
            conn.executemany(
                "INSERT INTO statistics(metadata_id, start_ts, sum) VALUES(?,?,?)",
                [
                    (1, 10.0, 10.0),
                    (1, 20.0, 11.0),
                    (1, 30.0, 2.0),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            s_resets = collect_reset_events_states(ro, "sensor.r", state_class="total_increasing")
            st_resets = collect_reset_events_statistics(ro, "statistics", "sensor.r", state_class="total_increasing")
        finally:
            ro.close()

        self.assertEqual(len(s_resets), 1)
        self.assertEqual(s_resets[0]["before"], f"{self._fmt_local(200.0)} (6)")
        self.assertEqual(s_resets[0]["after"], f"{self._fmt_local(300.0)} (1)")

        self.assertEqual(len(st_resets), 1)
        self.assertEqual(st_resets[0]["before"], f"{self._fmt_local(20.0)} (11.0)")
        self.assertEqual(st_resets[0]["after"], f"{self._fmt_local(30.0)} (2.0)")

    def test_reset_events_statistics_includes_last_reset_ts(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE statistics_meta (id INTEGER PRIMARY KEY, statistic_id TEXT)")
            conn.execute(
                "CREATE TABLE statistics (metadata_id INTEGER, start_ts REAL, sum REAL, last_reset_ts REAL)"
            )
            conn.execute("INSERT INTO statistics_meta(id, statistic_id) VALUES(1, ?)", ("sensor.r",))
            conn.executemany(
                "INSERT INTO statistics(metadata_id, start_ts, sum, last_reset_ts) VALUES(?,?,?,?)",
                [
                    (1, 10.0, 10.0, None),
                    (1, 20.0, 11.0, None),
                    (1, 30.0, 2.0, 25.0),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            resets = collect_reset_events_statistics(ro, "statistics", "sensor.r", state_class="total_increasing")
        finally:
            ro.close()

        self.assertEqual(len(resets), 1)
        self.assertEqual(resets[0]["before"], f"{self._fmt_local(20.0)} (11.0)")
        self.assertEqual(resets[0]["after"], f"{self._fmt_local(30.0)} (2.0)")
        self.assertEqual(resets[0]["last_reset"], f"{self._fmt_local(25.0)}")

    def test_reset_events_statistics_prefers_state_over_sum(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE statistics_meta (id INTEGER PRIMARY KEY, statistic_id TEXT)")
            # Both columns exist; sum stays monotonic but state resets.
            conn.execute("CREATE TABLE statistics (metadata_id INTEGER, start_ts REAL, state REAL, sum REAL)")
            conn.execute("INSERT INTO statistics_meta(id, statistic_id) VALUES(1, ?)", ("sensor.r",))
            conn.executemany(
                "INSERT INTO statistics(metadata_id, start_ts, state, sum) VALUES(?,?,?,?)",
                [
                    (1, 10.0, 5.0, 100.0),
                    (1, 20.0, 6.0, 110.0),
                    (1, 30.0, 1.0, 120.0),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            resets = collect_reset_events_statistics(ro, "statistics", "sensor.r", state_class="total_increasing")
        finally:
            ro.close()

        self.assertEqual(len(resets), 1)
        self.assertEqual(resets[0]["before"], f"{self._fmt_local(20.0)} (6.0)")
        self.assertEqual(resets[0]["after"], f"{self._fmt_local(30.0)} (1.0)")

    def test_missing_statistics_row_ranges(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE statistics_meta (id INTEGER PRIMARY KEY, statistic_id TEXT)")
            conn.execute("CREATE TABLE statistics (metadata_id INTEGER, start_ts REAL, state REAL, sum REAL)")
            conn.execute("CREATE TABLE statistics_short_term (metadata_id INTEGER, start_ts REAL, state REAL, sum REAL)")
            conn.execute("INSERT INTO statistics_meta(id, statistic_id) VALUES(1, ?)", ("sensor.gap",))

            # statistics (1h interval): gap between 2000 and 9200 => missing 1 row (5600)
            conn.executemany(
                "INSERT INTO statistics(metadata_id, start_ts, state, sum) VALUES(?,?,?,?)",
                [
                    (1, 2000.0, 1.0, 10.0),
                    (1, 5600.0, 2.0, 20.0),
                    (1, 9200.0, 3.0, 30.0),
                ],
            )

            # statistics_short_term (5m interval): gap between 1000 and 1600 => missing 1 row (1300)
            conn.executemany(
                "INSERT INTO statistics_short_term(metadata_id, start_ts, state, sum) VALUES(?,?,?,?)",
                [
                    (1, 1000.0, 5.0, 50.0),
                    (1, 1300.0, 6.0, 60.0),
                    (1, 1600.0, 7.0, 70.0),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            gaps_stats = collect_missing_statistics_row_ranges(ro, "statistics", "sensor.gap", interval_seconds=3600)
            gaps_st = collect_missing_statistics_row_ranges(
                ro, "statistics_short_term", "sensor.gap", interval_seconds=300
            )
        finally:
            ro.close()

        # No gaps: we inserted all expected rows above.
        self.assertEqual(gaps_stats, [])
        self.assertEqual(gaps_st, [])

        # Now create a gap by removing the middle rows.
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("DELETE FROM statistics WHERE start_ts = ?", (5600.0,))
            conn.execute("DELETE FROM statistics_short_term WHERE start_ts = ?", (1300.0,))
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            gaps_stats = collect_missing_statistics_row_ranges(ro, "statistics", "sensor.gap", interval_seconds=3600)
            gaps_st = collect_missing_statistics_row_ranges(
                ro, "statistics_short_term", "sensor.gap", interval_seconds=300
            )
        finally:
            ro.close()

        self.assertEqual(len(gaps_stats), 1)
        self.assertEqual(gaps_stats[0]["entity"], "sensor.gap")
        self.assertEqual(gaps_stats[0]["table"], "statistics")
        self.assertEqual(
            gaps_stats[0]["gap"],
            f"{self._fmt_local(2000.0)} - {self._fmt_local(9200.0)} [1 rows]\n1.0/10.0 - 3.0/30.0",
        )

        self.assertEqual(len(gaps_st), 1)
        self.assertEqual(gaps_st[0]["entity"], "sensor.gap")
        self.assertEqual(gaps_st[0]["table"], "statistics_short_term")
        self.assertEqual(
            gaps_st[0]["gap"],
            f"{self._fmt_local(1000.0)} - {self._fmt_local(1600.0)} [1 rows]\n5.0/50.0 - 7.0/70.0",
        )

    def test_total_increasing_small_dip_is_not_reset(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE states (entity_id TEXT, state TEXT, last_updated_ts REAL)")
            conn.executemany(
                "INSERT INTO states(entity_id, state, last_updated_ts) VALUES(?,?,?)",
                [
                    ("sensor.dip", "10", 100.0),
                    ("sensor.dip", "9.5", 200.0),  # 5% dip => warn in HA, not a reset
                    ("sensor.dip", "9.6", 300.0),
                ],
            )

            conn.execute("CREATE TABLE statistics (statistic_id TEXT, start_ts REAL, state REAL, sum REAL)")
            conn.executemany(
                "INSERT INTO statistics(statistic_id, start_ts, state, sum) VALUES(?,?,?,?)",
                [
                    ("sensor.dip", 10.0, 10.0, 10.0),
                    ("sensor.dip", 20.0, 9.5, 10.0),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            s_resets = collect_reset_events_states(ro, "sensor.dip", state_class="total_increasing")
            st_resets = collect_reset_events_statistics(ro, "statistics", "sensor.dip", state_class="total_increasing")
        finally:
            ro.close()

        self.assertEqual(s_resets, [])
        self.assertEqual(st_resets, [])

    def test_total_resets_use_last_reset_change_in_states(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "CREATE TABLE state_attributes (attributes_id INTEGER PRIMARY KEY, shared_attrs TEXT)"
            )
            conn.execute(
                "CREATE TABLE states (entity_id TEXT, state TEXT, attributes_id INTEGER, last_updated_ts REAL)"
            )

            conn.executemany(
                "INSERT INTO state_attributes(attributes_id, shared_attrs) VALUES(?,?)",
                [
                    (1, '{"last_reset": "2025-01-01T00:00:00+00:00"}'),
                    (2, '{"last_reset": "2025-02-01T00:00:00+00:00"}'),
                    (3, '{"last_reset": "bogus"}'),
                ],
            )
            conn.executemany(
                "INSERT INTO states(entity_id, state, attributes_id, last_updated_ts) VALUES(?,?,?,?)",
                [
                    (
                        "sensor.total",
                        "100",
                        1,
                        100.0,
                    ),
                    (
                        "sensor.total",
                        "110",
                        1,
                        200.0,
                    ),
                    (
                        "sensor.total",
                        "5",
                        2,
                        300.0,
                    ),
                    (
                        "sensor.total",
                        "6",
                        3,
                        400.0,
                    ),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            resets = collect_reset_events_states(ro, "sensor.total", state_class="total")
        finally:
            ro.close()

        self.assertEqual(len(resets), 1)
        self.assertEqual(resets[0]["before"], f"{self._fmt_local(200.0)} (110)")
        self.assertEqual(resets[0]["after"], f"{self._fmt_local(300.0)} (5)")
        self.assertIn("2025", resets[0].get("last_reset", ""))

    def test_stage5_statistics_change_report(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "CREATE TABLE statistics (id INTEGER PRIMARY KEY, statistic_id TEXT, start_ts REAL, state REAL, sum REAL, created_ts REAL)"
            )

            # Before snapshot:
            # - 0 exists and will change state
            # - 3600 exists and will be deleted
            conn.executemany(
                "INSERT INTO statistics(statistic_id, start_ts, state, sum, created_ts) VALUES(?,?,?,?,?)",
                [
                    ("sensor.new", 0.0, 1.0, 10.0, 1.0),
                    ("sensor.new", 3600.0, 2.0, 20.0, 2.0),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            before = snapshot_statistics_rows(ro, "statistics", "sensor.new")
        finally:
            ro.close()

        # Mutate DB to represent "after apply".
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("DELETE FROM statistics WHERE statistic_id = ?", ("sensor.new",))
            conn.executemany(
                "INSERT INTO statistics(statistic_id, start_ts, state, sum, created_ts) VALUES(?,?,?,?,?)",
                [
                    ("sensor.new", 0.0, 1.5, 10.0, 3.0),  # state + created_ts changed
                    ("sensor.new", 7200.0, 3.0, 30.0, 4.0),  # new row
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            after = snapshot_statistics_rows(ro, "statistics", "sensor.new")
        finally:
            ro.close()

        b_ts_col, b_ts_sec, b_rows = before
        a_ts_col, a_ts_sec, a_rows = after
        self.assertIsNotNone(b_ts_col)
        self.assertIsNotNone(a_ts_col)
        b_ts_col_s = cast(str, b_ts_col)
        a_ts_col_s = cast(str, a_ts_col)
        self.assertEqual(b_ts_col_s, "start_ts")
        self.assertEqual(a_ts_col_s, "start_ts")

        headers, rows = build_statistics_change_report(
            before=b_rows,
            after=a_rows,
            before_ts_col=b_ts_col_s,
            before_ts_is_seconds=b_ts_sec,
            after_ts_col=a_ts_col_s,
            after_ts_is_seconds=a_ts_sec,
        )

        # Expect columns for state + created_ts + sum (sum differs due to inserted/deleted rows).
        self.assertEqual(headers[0], "start_dt")
        self.assertIn("state", headers)
        self.assertIn("created_ts", headers)
        self.assertIn("sum", headers)

        # Rows should include timestamps for: 0, 3600 (deleted), 7200 (inserted)
        self.assertEqual(len(rows), 3)
        by_dt = {r[0]: r for r in rows}
        r0 = by_dt[self._fmt_local(0.0)]
        r3600 = by_dt[self._fmt_local(3600.0)]
        r7200 = by_dt[self._fmt_local(7200.0)]

        state_idx = headers.index("state")
        created_idx = headers.index("created_ts")
        sum_idx = headers.index("sum")

        self.assertEqual(r0[state_idx], "1.0 -> 1.5 (+0.5)")
        self.assertTrue(r0[created_idx].startswith(self._fmt_local(1.0)))
        self.assertTrue(r0[created_idx].endswith(self._fmt_local(3.0)))
        self.assertEqual(r0[sum_idx], "")

        self.assertEqual(r3600[state_idx], "2.0 -> (none)")
        self.assertEqual(r3600[sum_idx], "20.0 -> (none)")
        self.assertEqual(r7200[state_idx], "(none) -> 3.0")
        self.assertEqual(r7200[sum_idx], "(none) -> 30.0")

    def test_stage5_condense_keeps_nontrivial_rows(self) -> None:
        # Build a long contiguous range where all rows differ in sum/created_ts,
        # but one middle row also differs in state. Condensing must keep that row.
        before: dict[float, dict[str, object]] = {}
        after: dict[float, dict[str, object]] = {}
        for i in range(10):
            ts = float(i * 3600)
            before[ts] = {"start_ts": ts, "state": 1.0, "sum": 10.0 + i, "created_ts": 1.0}
            after_state = 1.0
            if i == 5:
                after_state = 2.0
            after[ts] = {"start_ts": ts, "state": after_state, "sum": 100.0 + i, "created_ts": 2.0}

        headers, epoch_rows = build_statistics_change_report_with_epochs(
            before=before,
            after=after,
            before_ts_col="start_ts",
            before_ts_is_seconds=True,
            after_ts_col="start_ts",
            after_ts_is_seconds=True,
        )
        condensed = condense_statistics_change_report_rows(
            headers,
            epoch_rows,
            interval_seconds=3600,
            trivial_columns=("sum", "created_ts"),
        )

        # Expect an ellipsis (we condensed), and the middle non-trivial row preserved.
        self.assertTrue(any(r and r[0] == "..." for r in condensed))
        middle_dt = self._fmt_local(5 * 3600.0)
        self.assertTrue(any(r and r[0] == middle_dt for r in condensed))

    def test_total_resets_use_last_reset_change_in_statistics(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "CREATE TABLE statistics (statistic_id TEXT, start_ts REAL, state REAL, sum REAL, last_reset_ts REAL)"
            )
            conn.executemany(
                "INSERT INTO statistics(statistic_id, start_ts, state, sum, last_reset_ts) VALUES(?,?,?,?,?)",
                [
                    ("sensor.total", 10.0, 100.0, 100.0, 1000.0),
                    ("sensor.total", 20.0, 110.0, 110.0, 1000.0),
                    ("sensor.total", 30.0, 5.0, 115.0, 2000.0),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            resets = collect_reset_events_statistics(ro, "statistics", "sensor.total", state_class="total")
        finally:
            ro.close()

        self.assertEqual(len(resets), 1)
        self.assertEqual(resets[0]["before"], f"{self._fmt_local(20.0)} (110.0)")
        self.assertEqual(resets[0]["after"], f"{self._fmt_local(30.0)} (5.0)")
        self.assertEqual(resets[0]["last_reset"], f"{self._fmt_local(2000.0)}")
