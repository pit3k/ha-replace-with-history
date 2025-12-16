import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from ha_replace_with_history.db_summary import (
    apply_sql_script,
    build_statistics_update_sql_script,
    collect_missing_statistics_row_ranges,
    collect_reset_events_statistics,
    collect_generated_statistics_preview_rows,
    connect_readonly_sqlite,
    create_statistics_generated_view,
    get_earliest_state_ts_epoch,
    get_latest_statistics_ts_epoch,
)


class TestStage3StatisticsSimulation(unittest.TestCase):
    def _fmt_local(self, ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

    def _make_db(self) -> Path:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        tmp.close()
        return Path(tmp.name)

    def test_stage3_views_and_gap_detection(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE states (entity_id TEXT, state TEXT, last_updated_ts REAL)")
            conn.execute(
                "CREATE TABLE statistics (statistic_id TEXT, start_ts REAL, state REAL, sum REAL)"
            )
            conn.execute(
                "CREATE TABLE statistics_short_term (statistic_id TEXT, start_ts REAL, state REAL, sum REAL)"
            )

            # Old stats (aligned to 1h) for old entity.
            conn.executemany(
                "INSERT INTO statistics(statistic_id, start_ts, state, sum) VALUES(?,?,?,?)",
                [
                    ("sensor.old", 0.0, 10.0, 100.0),
                    ("sensor.old", 3600.0, 20.0, 110.0),
                ],
            )

            # New entity states start later, so precondition can pass.
            # Also include an internal 1h bucket (14400) with no states; HA keeps generating
            # stats with unchanged values from the prior bucket, so no gap should be reported
            # inside the generated portion.
            conn.executemany(
                "INSERT INTO states(entity_id, state, last_updated_ts) VALUES(?,?,?)",
                [
                    ("sensor.new", "30", 11000.0),
                    ("sensor.new", "31", 11100.0),
                    # next complete bucket with a state
                    ("sensor.new", "40", 18100.0),
                    # state in the cutoff (in-progress) bucket so 18000 is included
                    ("sensor.new", "41", 22000.0),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            new_earliest = get_earliest_state_ts_epoch(ro, "sensor.new")
            old_latest = get_latest_statistics_ts_epoch(ro, "statistics", "sensor.old")
            self.assertIsNotNone(new_earliest)
            self.assertIsNotNone(old_latest)
            self.assertGreater(new_earliest, old_latest)

            create_statistics_generated_view(
                ro,
                view_name="statistics_generated",
                source_table="statistics",
                old_statistic_id="sensor.old",
                new_statistic_id="sensor.new",
                interval_seconds=3600,
                statistics_kind="total_increasing",
                new_entity_started_from_0=False,
            )

            # Simulated resets should be none for this data.
            resets = collect_reset_events_statistics(
                ro,
                "statistics_generated",
                "sensor.new",
                state_class="total_increasing",
            )
            self.assertEqual(resets, [])

            # Only the pre-first-state gap should be detected (missing one 7200 row).
            # The internal 14400 bucket is forward-filled and should not create a gap.
            gaps = collect_missing_statistics_row_ranges(
                ro, "statistics_generated", "sensor.new", interval_seconds=3600
            )
            self.assertEqual(len(gaps), 1)
            self.assertEqual(gaps[0]["entity"], "sensor.new")
            self.assertEqual(gaps[0]["table"], "statistics_generated")
            self.assertEqual(
                gaps[0]["gap"],
                f"{self._fmt_local(3600.0)} - {self._fmt_local(10800.0)} [1 rows]\n20.0/110.0 - 31.0/111.0",
            )
        finally:
            ro.close()

    def test_stage3_update_sql_script_legacy_schema(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE states (entity_id TEXT, state TEXT, last_updated_ts REAL)")
            conn.execute(
                "CREATE TABLE statistics (statistic_id TEXT, start_ts REAL, state REAL, sum REAL)"
            )
            conn.execute(
                "CREATE TABLE statistics_short_term (statistic_id TEXT, start_ts REAL, state REAL, sum REAL)"
            )

            # Old stats, plus some existing new stats (to be deleted by the script)
            conn.executemany(
                "INSERT INTO statistics(statistic_id, start_ts, state, sum) VALUES(?,?,?,?)",
                [
                    ("sensor.old", 0.0, 10.0, 100.0),
                    ("sensor.old", 3600.0, 20.0, 110.0),
                    ("sensor.new", 0.0, 1.0, 1.0),
                ],
            )
            conn.executemany(
                "INSERT INTO states(entity_id, state, last_updated_ts) VALUES(?,?,?)",
                [
                    ("sensor.new", "30", 11000.0),
                    ("sensor.new", "31", 11100.0),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            script = build_statistics_update_sql_script(
                ro,
                old_entity_id="sensor.old",
                new_entity_id="sensor.new",
                stats_view="statistics_generated",
                stats_st_view="statistics_short_term_generated",
                statistics_kind="total_increasing",
                new_entity_started_from_0=False,
            )

            self.assertIn("BEGIN;", script)
            self.assertIn("COMMIT;", script)
            self.assertIn("DELETE FROM statistics WHERE statistic_id = 'sensor.new';", script)
            self.assertIn("DELETE FROM statistics_short_term WHERE statistic_id = 'sensor.new';", script)
            self.assertIn("INSERT INTO statistics(", script)
            self.assertIn("INSERT INTO statistics_short_term(", script)
            # Ensure we don't emit accidental double-semicolons.
            self.assertNotIn(";;", script)
        finally:
            ro.close()

    def test_stage4_total_generated_view_and_sql_script(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "CREATE TABLE states (entity_id TEXT, state TEXT, last_updated_ts REAL, attributes TEXT)"
            )
            conn.execute(
                "CREATE TABLE statistics (statistic_id TEXT, start_ts REAL, state REAL, sum REAL, last_reset TEXT, last_reset_ts REAL)"
            )
            conn.execute(
                "CREATE TABLE statistics_short_term (statistic_id TEXT, start_ts REAL, state REAL, sum REAL, last_reset TEXT, last_reset_ts REAL)"
            )

            # Old entity has one historical statistics row.
            conn.executemany(
                "INSERT INTO statistics(statistic_id, start_ts, state, sum, last_reset, last_reset_ts) VALUES(?,?,?,?,?,?)",
                [
                    (
                        "sensor.old",
                        0.0,
                        10.0,
                        100.0,
                        "2025-01-01T00:00:00+00:00",
                        1735689600.0,
                    )
                ],
            )

            # New entity states start after old stats; last_reset changes should be reflected.
            conn.executemany(
                "INSERT INTO states(entity_id, state, last_updated_ts, attributes) VALUES(?,?,?,?)",
                [
                    (
                        "sensor.new",
                        "1",
                        4000.0,
                        '{"last_reset":"2025-01-02T00:00:00+00:00"}',
                    ),
                    (
                        "sensor.new",
                        "2",
                        4100.0,
                        '{"last_reset":"2025-01-02T00:00:00+00:00"}',
                    ),
                    (
                        "sensor.new",
                        "0.5",
                        8000.0,
                        '{"last_reset":"2025-01-03T00:00:00+00:00"}',
                    ),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            create_statistics_generated_view(
                ro,
                view_name="statistics_generated",
                source_table="statistics",
                old_statistic_id="sensor.old",
                new_statistic_id="sensor.new",
                interval_seconds=3600,
                statistics_kind="total",
                new_entity_started_from_0=True,
            )

            # Legacy last_reset should be NULL when last_reset_ts exists.
            row = ro.execute(
                "SELECT last_reset FROM statistics_generated WHERE statistic_id = ? AND start_ts > 0 ORDER BY start_ts ASC LIMIT 1",
                ("sensor.new",),
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertIsNone(row[0])

            # last_reset_ts should be populated.
            row2 = ro.execute(
                "SELECT last_reset_ts FROM statistics_generated WHERE statistic_id = ? AND start_ts > 0 ORDER BY start_ts ASC LIMIT 1",
                ("sensor.new",),
            ).fetchone()
            self.assertIsNotNone(row2)
            self.assertIsNotNone(row2[0])

            script = build_statistics_update_sql_script(
                ro,
                old_entity_id="sensor.old",
                new_entity_id="sensor.new",
                stats_view="statistics_generated",
                stats_st_view="statistics_short_term_generated",
                statistics_kind="total",
                new_entity_started_from_0=True,
            )
            # Ensure we populate last_reset columns when present.
            self.assertIn("last_reset", script)
            self.assertIn("last_reset_ts", script)
        finally:
            ro.close()

    def test_stage3_update_sql_script_metadata_schema(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE states_meta (metadata_id INTEGER PRIMARY KEY, entity_id TEXT)")
            conn.execute(
                "CREATE TABLE states (metadata_id INTEGER, state TEXT, last_updated_ts REAL)"
            )
            conn.execute("CREATE TABLE statistics_meta (id INTEGER PRIMARY KEY, statistic_id TEXT)")
            conn.execute(
                "CREATE TABLE statistics (metadata_id INTEGER, start_ts REAL, state REAL, sum REAL)"
            )
            conn.execute(
                "CREATE TABLE statistics_short_term (metadata_id INTEGER, start_ts REAL, state REAL, sum REAL)"
            )

            conn.execute("INSERT INTO statistics_meta(id, statistic_id) VALUES(1, 'sensor.old')")
            conn.execute("INSERT INTO statistics_meta(id, statistic_id) VALUES(2, 'sensor.new')")

            conn.execute("INSERT INTO states_meta(metadata_id, entity_id) VALUES(10, 'sensor.new')")

            # Old stats stored via metadata_id=1
            conn.executemany(
                "INSERT INTO statistics(metadata_id, start_ts, state, sum) VALUES(?,?,?,?)",
                [
                    (1, 0.0, 10.0, 100.0),
                    (1, 3600.0, 20.0, 110.0),
                ],
            )

            # New entity states via states_meta join
            conn.executemany(
                "INSERT INTO states(metadata_id, state, last_updated_ts) VALUES(?,?,?)",
                [
                    (10, "30", 11000.0),
                    (10, "31", 11100.0),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            script = build_statistics_update_sql_script(
                ro,
                old_entity_id="sensor.old",
                new_entity_id="sensor.new",
                stats_view="statistics_generated",
                stats_st_view="statistics_short_term_generated",
                statistics_kind="total_increasing",
                new_entity_started_from_0=False,
            )

            # metadata schema uses metadata_id deletes/inserts via statistics_meta lookup
            self.assertIn(
                "DELETE FROM statistics WHERE metadata_id = (SELECT id FROM statistics_meta WHERE statistic_id = 'sensor.new');",
                script,
            )
            self.assertIn(
                "INSERT INTO statistics(metadata_id, start_ts, state, sum)",
                script,
            )
            self.assertIn(
                "(SELECT id FROM statistics_meta WHERE statistic_id = 'sensor.new') AS metadata_id",
                script,
            )
            self.assertNotIn(";;", script)
        finally:
            ro.close()

    def test_stage3_tail_rows_omit_earlier_copied(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE states (entity_id TEXT, state TEXT, last_updated_ts REAL)")
            conn.execute(
                "CREATE TABLE statistics (statistic_id TEXT, start_ts REAL, state REAL, sum REAL)"
            )

            # Old stats include earlier rows that should be omitted from the tail report.
            conn.executemany(
                "INSERT INTO statistics(statistic_id, start_ts, state, sum) VALUES(?,?,?,?)",
                [
                    ("sensor.old", 0.0, 10.0, 100.0),
                    ("sensor.old", 3600.0, 20.0, 110.0),
                ],
            )

            # New entity states start later. To consider the 10800 bucket complete,
            # include at least one state in the following bucket (14400), which will
            # itself be excluded as the last/in-progress bucket.
            conn.executemany(
                "INSERT INTO states(entity_id, state, last_updated_ts) VALUES(?,?,?)",
                [
                    ("sensor.new", "30", 11000.0),
                    ("sensor.new", "31", 11100.0),
                    ("sensor.new", "32", 14500.0),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            create_statistics_generated_view(
                ro,
                view_name="statistics_generated",
                source_table="statistics",
                old_statistic_id="sensor.old",
                new_statistic_id="sensor.new",
                interval_seconds=3600,
                statistics_kind="total_increasing",
                new_entity_started_from_0=False,
            )

            preview = collect_generated_statistics_preview_rows(
                ro,
                source_table="statistics",
                old_entity_id="sensor.old",
                new_entity_id="sensor.new",
                generated_view="statistics_generated",
                first_generated=3,
                last_generated=3,
            )

            # Expect to start at the last copied old row (3600) and include the generated row (10800).
            self.assertEqual([r["ts"] for r in preview], [self._fmt_local(3600.0), self._fmt_local(10800.0)])
        finally:
            ro.close()

    def test_stage3_preview_rows_includes_ellipsis(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE states (entity_id TEXT, state TEXT, last_updated_ts REAL)")
            conn.execute(
                "CREATE TABLE statistics (statistic_id TEXT, start_ts REAL, state REAL, sum REAL)"
            )

            # Old stats last row at 3600.
            conn.executemany(
                "INSERT INTO statistics(statistic_id, start_ts, state, sum) VALUES(?,?,?,?)",
                [
                    ("sensor.old", 0.0, 10.0, 100.0),
                    ("sensor.old", 3600.0, 20.0, 110.0),
                ],
            )

            # Create 8 buckets worth of states so that after excluding the last
            # in-progress bucket we still have 7 generated buckets: 10800 .. 32400.
            states = []
            v = 30
            for t in (11000.0, 14600.0, 18200.0, 21800.0, 25400.0, 29000.0, 32600.0, 36200.0):
                states.append(("sensor.new", str(v), t))
                v += 1
            conn.executemany(
                "INSERT INTO states(entity_id, state, last_updated_ts) VALUES(?,?,?)",
                states,
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            create_statistics_generated_view(
                ro,
                view_name="statistics_generated",
                source_table="statistics",
                old_statistic_id="sensor.old",
                new_statistic_id="sensor.new",
                interval_seconds=3600,
                statistics_kind="total_increasing",
                new_entity_started_from_0=False,
            )

            preview = collect_generated_statistics_preview_rows(
                ro,
                source_table="statistics",
                old_entity_id="sensor.old",
                new_entity_id="sensor.new",
                generated_view="statistics_generated",
                first_generated=3,
                last_generated=3,
            )

            # copied row + 3 first + ellipsis + 3 last = 8 rows
            self.assertEqual(len(preview), 8)
            self.assertEqual(preview[0]["ts"], self._fmt_local(3600.0))
            self.assertEqual(
                [r["ts"] for r in preview[1:4]],
                [self._fmt_local(10800.0), self._fmt_local(14400.0), self._fmt_local(18000.0)],
            )
            self.assertEqual(preview[4]["ts"], "...")
            self.assertEqual(
                [r["ts"] for r in preview[5:]],
                [self._fmt_local(25200.0), self._fmt_local(28800.0), self._fmt_local(32400.0)],
            )
        finally:
            ro.close()

    def test_stage3_new_entity_started_from_0_changes_first_sum(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE states (entity_id TEXT, state TEXT, last_updated_ts REAL)")
            conn.execute(
                "CREATE TABLE statistics (statistic_id TEXT, start_ts REAL, state REAL, sum REAL)"
            )

            # Old last sum is 110 at 3600.
            conn.executemany(
                "INSERT INTO statistics(statistic_id, start_ts, state, sum) VALUES(?,?,?,?)",
                [
                    ("sensor.old", 0.0, 10.0, 100.0),
                    ("sensor.old", 3600.0, 20.0, 110.0),
                ],
            )

            # New entity first reading is 30 in bucket 10800.
            conn.executemany(
                "INSERT INTO states(entity_id, state, last_updated_ts) VALUES(?,?,?)",
                [
                    ("sensor.new", "30", 11000.0),
                    ("sensor.new", "31", 14600.0),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            # started_from_0 = false => first generated sum stays at base_sum (110.0)
            create_statistics_generated_view(
                ro,
                view_name="statistics_generated_false",
                source_table="statistics",
                old_statistic_id="sensor.old",
                new_statistic_id="sensor.new",
                interval_seconds=3600,
                statistics_kind="total_increasing",
                new_entity_started_from_0=False,
            )
            sum_false = ro.execute(
                "SELECT sum FROM statistics_generated_false WHERE statistic_id = ? AND start_ts = ?",
                ("sensor.new", 10800.0),
            ).fetchone()["sum"]
            self.assertEqual(sum_false, 110.0)

            # started_from_0 = true => first generated sum includes full first reading (110 + 30)
            create_statistics_generated_view(
                ro,
                view_name="statistics_generated_true",
                source_table="statistics",
                old_statistic_id="sensor.old",
                new_statistic_id="sensor.new",
                interval_seconds=3600,
                statistics_kind="total_increasing",
                new_entity_started_from_0=True,
            )
            sum_true = ro.execute(
                "SELECT sum FROM statistics_generated_true WHERE statistic_id = ? AND start_ts = ?",
                ("sensor.new", 10800.0),
            ).fetchone()["sum"]
            self.assertEqual(sum_true, 140.0)
        finally:
            ro.close()

    def test_stage3_populates_created_ts_when_required(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE states (entity_id TEXT, state TEXT, last_updated_ts REAL)")
            # Destination tables require created_ts.
            conn.execute(
                "CREATE TABLE statistics (statistic_id TEXT, created_ts REAL, start_ts REAL, state REAL, sum REAL)"
            )
            conn.execute(
                "CREATE TABLE statistics_short_term (statistic_id TEXT, created_ts REAL, start_ts REAL, state REAL, sum REAL)"
            )

            # Old stats include created_ts, which should be preserved for copied rows in the view.
            conn.executemany(
                "INSERT INTO statistics(statistic_id, created_ts, start_ts, state, sum) VALUES(?,?,?,?,?)",
                [
                    ("sensor.old", 123.0, 0.0, 10.0, 100.0),
                    ("sensor.old", 124.0, 3600.0, 20.0, 110.0),
                ],
            )
            conn.executemany(
                "INSERT INTO statistics_short_term(statistic_id, created_ts, start_ts, state, sum) VALUES(?,?,?,?,?)",
                [
                    ("sensor.old", 200.0, 0.0, 10.0, 100.0),
                ],
            )

            conn.executemany(
                "INSERT INTO states(entity_id, state, last_updated_ts) VALUES(?,?,?)",
                [
                    ("sensor.new", "30", 11000.0),
                    ("sensor.new", "31", 14600.0),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            create_statistics_generated_view(
                ro,
                view_name="statistics_generated",
                source_table="statistics",
                old_statistic_id="sensor.old",
                new_statistic_id="sensor.new",
                interval_seconds=3600,
                statistics_kind="total_increasing",
                new_entity_started_from_0=False,
            )

            copied_created = ro.execute(
                "SELECT created_ts FROM statistics_generated WHERE statistic_id = ? AND start_ts = ?",
                ("sensor.new", 0.0),
            ).fetchone()["created_ts"]
            self.assertEqual(copied_created, 123.0)

            script = build_statistics_update_sql_script(
                ro,
                old_entity_id="sensor.old",
                new_entity_id="sensor.new",
                stats_view="statistics_generated",
                stats_st_view="statistics_short_term_generated",
                statistics_kind="total_increasing",
                new_entity_started_from_0=False,
            )

            self.assertIn("created_ts", script)
        finally:
            ro.close()

    def test_stage3_measurement_generates_min_mean_max(self) -> None:
        db_path = self._make_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE states (entity_id TEXT, state TEXT, last_updated_ts REAL)")
            conn.execute(
                "CREATE TABLE statistics (statistic_id TEXT, start_ts REAL, state REAL, min REAL, mean REAL, max REAL)"
            )

            # Old stats copied as-is.
            conn.executemany(
                "INSERT INTO statistics(statistic_id, start_ts, state, min, mean, max) VALUES(?,?,?,?,?,?)",
                [
                    ("sensor.old", 0.0, 1.0, 1.0, 1.0, 1.0),
                    ("sensor.old", 3600.0, 2.0, 2.0, 2.0, 2.0),
                ],
            )

            # New entity values within bucket 10800: 1, 3, 5 => min=1, mean=3, max=5; last state=5.
            # Add one state in the next bucket (14400) to mark 10800 as complete.
            conn.executemany(
                "INSERT INTO states(entity_id, state, last_updated_ts) VALUES(?,?,?)",
                [
                    ("sensor.new", "1", 11000.0),
                    ("sensor.new", "3", 11100.0),
                    ("sensor.new", "5", 11200.0),
                    ("sensor.new", "6", 14500.0),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            create_statistics_generated_view(
                ro,
                view_name="statistics_generated",
                source_table="statistics",
                old_statistic_id="sensor.old",
                new_statistic_id="sensor.new",
                interval_seconds=3600,
                statistics_kind="measurement",
                new_entity_started_from_0=False,
            )

            row = ro.execute(
                "SELECT start_ts, state, min, mean, max FROM statistics_generated WHERE statistic_id = ? AND start_ts = ?",
                ("sensor.new", 10800.0),
            ).fetchone()
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row["state"], 5.0)
            self.assertEqual(row["min"], 1.0)
            self.assertEqual(row["mean"], 3.0)
            self.assertEqual(row["max"], 5.0)
        finally:
            ro.close()

    def test_apply_sql_script_executes_stage3_update(self) -> None:
        # Build a file-backed DB so apply_sql_script can reopen it RW.
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        tmp.close()
        db_path = Path(tmp.name)

        conn = sqlite3.connect(str(db_path))
        try:
            # Legacy schema for simplicity.
            conn.execute("CREATE TABLE states (entity_id TEXT, state TEXT, last_updated_ts REAL)")
            conn.execute(
                "CREATE TABLE statistics (statistic_id TEXT, start_ts REAL, state REAL, sum REAL)"
            )
            conn.execute(
                "CREATE TABLE statistics_short_term (statistic_id TEXT, start_ts REAL, state REAL, sum REAL)"
            )

            # Old statistics.
            conn.executemany(
                "INSERT INTO statistics(statistic_id, start_ts, state, sum) VALUES(?,?,?,?)",
                [
                    ("sensor.old", 0.0, 10.0, 100.0),
                    ("sensor.old", 3600.0, 20.0, 110.0),
                ],
            )

            # Existing new stats that should get deleted.
            conn.executemany(
                "INSERT INTO statistics(statistic_id, start_ts, state, sum) VALUES(?,?,?,?)",
                [
                    ("sensor.new", 0.0, 999.0, 999.0),
                ],
            )

            # New states to generate at bucket 10800.
            conn.executemany(
                "INSERT INTO states(entity_id, state, last_updated_ts) VALUES(?,?,?)",
                [
                    ("sensor.new", "30", 11000.0),
                    ("sensor.new", "31", 14600.0),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        ro = connect_readonly_sqlite(db_path)
        try:
            script = build_statistics_update_sql_script(
                ro,
                old_entity_id="sensor.old",
                new_entity_id="sensor.new",
                stats_view="statistics_generated",
                stats_st_view="statistics_short_term_generated",
                statistics_kind="total_increasing",
                new_entity_started_from_0=False,
            )
        finally:
            ro.close()

        apply_sql_script(db_path, script)

        # Verify rows were replaced.
        chk = sqlite3.connect(str(db_path))
        chk.row_factory = sqlite3.Row
        try:
            rows = chk.execute(
                "SELECT start_ts, state, sum FROM statistics WHERE statistic_id = ? ORDER BY start_ts",
                ("sensor.new",),
            ).fetchall()
            # Last in-progress bucket is excluded, so only the complete generated bucket remains.
            self.assertEqual([r["start_ts"] for r in rows], [0.0, 3600.0, 10800.0])
            # Copied row at 0.0 should match old's 0.0 state/sum, not the old pre-existing new row.
            self.assertEqual((rows[0]["state"], rows[0]["sum"]), (10.0, 100.0))
        finally:
            chk.close()
