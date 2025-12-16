import json
import tempfile
import unittest
from pathlib import Path

from ha_replace_with_history.entity_registry import EntityRegistryError, get_entity, load_entity_registry
from ha_replace_with_history.report import render_entity_registry_report


class TestStage1EntityRegistryReport(unittest.TestCase):
    def test_load_and_lookup_entity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = Path(td)
            (storage / "core.entity_registry").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "key": "core.entity_registry",
                        "data": {
                            "entities": [
                                {
                                    "entity_id": "sensor.old",
                                    "unique_id": "u1",
                                    "platform": "test",
                                    "has_entity_name": True,
                                    "aliases": ["not scalar"],
                                },
                                {
                                    "entity_id": "sensor.new",
                                    "unique_id": "u2",
                                    "platform": "test",
                                    "has_entity_name": True,
                                },
                            ]
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            registry = load_entity_registry(storage_dir=storage)
            self.assertEqual(registry.path, storage / "core.entity_registry")

            old = get_entity(registry, "sensor.old")
            new = get_entity(registry, "sensor.new")
            self.assertIsNotNone(old)
            self.assertIsNotNone(new)
            assert old is not None
            assert new is not None
            self.assertEqual(old["unique_id"], "u1")
            self.assertEqual(new["unique_id"], "u2")

    def test_missing_registry_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            storage = Path(td)
            with self.assertRaises(EntityRegistryError):
                load_entity_registry(storage_dir=storage)

    def test_report_includes_lists_and_flattens_dicts(self) -> None:
        old = {
            "entity_id": "sensor.old",
            "unique_id": "u1",
            "has_entity_name": True,
            "area_id": None,
            "aliases": ["a"],
            "options": {"a": 1, "b": 2},
        }
        new = {
            "entity_id": "sensor.new",
            "unique_id": "u1",
            "has_entity_name": True,
            "area_id": None,
            "aliases": ["b"],
            "options": {"a": 1, "b": 3},
        }

        out = render_entity_registry_report(
            old_entity_id="sensor.old",
            new_entity_id="sensor.new",
            old=old,
            new=new,
        )

        self.assertIn("unique_id", out)
        self.assertIn("✓", out)  # unique_id matches
        self.assertIn("entity_id", out)
        self.assertIn("X", out)  # entity_id differs

        # Both-null scalar fields should be skipped
        self.assertNotIn("area_id", out)

        # List field is included (as a leaf row)
        self.assertIn("aliases", out)

        # Dict field is exploded into path rows
        self.assertIn("options/a", out)
        self.assertIn("options/b", out)

    def test_report_missing_attribute_marks_left_right(self) -> None:
        old = {"entity_id": "sensor.old", "unique_id": "u1"}
        new = {"entity_id": "sensor.new"}

        out = render_entity_registry_report(
            old_entity_id="sensor.old",
            new_entity_id="sensor.new",
            old=old,
            new=new,
        )

        # unique_id exists only on old => <<
        self.assertIn("unique_id", out)
        self.assertIn("<<", out)


if __name__ == "__main__":
    unittest.main()
