from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable


_MISSING = object()


@dataclass(frozen=True)
class Cell:
    present: bool
    value: Any


def _cell(entity: dict[str, Any] | None, key: str) -> Cell:
    if entity is None:
        return Cell(present=False, value=_MISSING)
    if key not in entity:
        return Cell(present=False, value=_MISSING)
    return Cell(present=True, value=entity.get(key))


def _flatten_value(value: Any, *, prefix: str) -> dict[str, Cell]:
    """Flatten nested dicts into path keys like `a/b`.

    - Dicts are exploded into sub-keys.
    - Empty dicts are kept as a leaf value to keep the attribute visible.
    - Lists are kept as leaf values (JSON-stringified for display).
    - Scalars (including None) are leaf values.
    """
    if isinstance(value, dict):
        if not value:
            return {prefix: Cell(present=True, value=value)}

        out: dict[str, Cell] = {}
        for k, v in value.items():
            key = str(k)
            child_prefix = f"{prefix}/{key}" if prefix else key
            out.update(_flatten_value(v, prefix=child_prefix))
        return out

    if isinstance(value, list):
        return {prefix: Cell(present=True, value=value)}

    return {prefix: Cell(present=True, value=value)}


def flatten_entity(entity: dict[str, Any] | None) -> dict[str, Cell]:
    if entity is None:
        return {}

    out: dict[str, Cell] = {}
    for k, v in entity.items():
        out.update(_flatten_value(v, prefix=str(k)))
    return out


def _stringify(key: str, value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value

    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except TypeError:
            return str(value)

    return str(value)


def diff_symbol(old: Cell, new: Cell, *, tick: str = "✓") -> str:
    """Return one of: tick (same), X (different), << (only old), >> (only new)."""
    if old.present and new.present:
        return tick if old.value == new.value else "X"
    if old.present and not new.present:
        return "<<"
    if new.present and not old.present:
        return ">>"
    return ""


def iter_comparable_keys(old_entity: dict[str, Any] | None, new_entity: dict[str, Any] | None) -> list[str]:
    # Helpful stable ordering: put common identifiers first if present.
    preferred = [
        "entity_id",
        "id",
        "unique_id",
        "platform",
        "config_entry_id",
        "config_subentry_id",
        "device_id",
        "area_id",
        "name",
        "original_name",
        "icon",
        "original_icon",
        "disabled_by",
        "hidden_by",
        "entity_category",
        "has_entity_name",
        "device_class",
        "original_device_class",
        "unit_of_measurement",
        "supported_features",
        "translation_key",
        "previous_unique_id",
        "created_at",
        "modified_at",
    ]

    old_flat = flatten_entity(old_entity)
    new_flat = flatten_entity(new_entity)
    keys = list(set(old_flat.keys()) | set(new_flat.keys()))

    preferred_index = {name: idx for idx, name in enumerate(preferred)}

    def sort_key(path: str) -> tuple[int, str]:
        top = path.split("/", 1)[0]
        return (preferred_index.get(top, 10_000), path)

    keys.sort(key=sort_key)
    return keys


def render_entity_registry_report(
    *,
    old_entity_id: str,
    new_entity_id: str,
    old: dict[str, Any] | None,
    new: dict[str, Any] | None,
    tick: str = "✓",
    color: bool = False,
) -> str:
    headers = ["attribute", old_entity_id, new_entity_id, "diff"]

    keys = iter_comparable_keys(old, new)

    old_flat = flatten_entity(old)
    new_flat = flatten_entity(new)

    rows: list[list[str]] = []
    for key in keys:
        old_cell = old_flat.get(key, Cell(present=False, value=_MISSING))
        new_cell = new_flat.get(key, Cell(present=False, value=_MISSING))

        # Skip rows where both sides explicitly have a null value and are equal.
        # Home Assistant's entity registry stores many keys with null values; this reduces noise.
        if old_cell.present and new_cell.present and old_cell.value is None and new_cell.value is None:
            continue

        old_text = "" if not old_cell.present else _stringify(key, old_cell.value)
        new_text = "" if not new_cell.present else _stringify(key, new_cell.value)
        rows.append([key, old_text, new_text, diff_symbol(old_cell, new_cell, tick=tick)])

    def _split_cell(text: str) -> list[str]:
        # Keep empty as single-line.
        parts = text.split("\n")
        return parts if parts else [""]

    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], max((len(p) for p in _split_cell(c)), default=0))

    def fmt_row(cols: Iterable[str]) -> str:
        cols_l = list(cols)
        return " | ".join(cols_l[i].ljust(widths[i]) for i in range(4)).rstrip()

    def fmt_row_multiline(cols: list[str]) -> list[str]:
        col_lines = [_split_cell(c) for c in cols]
        height = max(len(x) for x in col_lines)
        out: list[str] = []
        for line_idx in range(height):
            physical = [
                (col_lines[col_idx][line_idx] if line_idx < len(col_lines[col_idx]) else "")
                for col_idx in range(4)
            ]
            out.append(fmt_row(physical))
        return out

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    for r in rows:
        physical_lines = fmt_row_multiline(r)
        if color:
            if r[3] == "X":
                physical_lines = [f"\x1b[31m{ln}\x1b[0m" for ln in physical_lines]
            elif r[3] == tick:
                physical_lines = [f"\x1b[32m{ln}\x1b[0m" for ln in physical_lines]
        lines.extend(physical_lines)
    return "\n".join(lines) + "\n"


def render_unavailable_occurrences_report(
    *,
    rows: list[dict[str, str]],
    color: bool = False,
) -> str:
    """Render a simple table for unavailable occurrence runs.

    Expected keys per row: entity, table, when, value, count
    """
    headers = ["entity", "table", "when", "value", "count"]

    table_rows: list[list[str]] = []
    for r in rows:
        table_rows.append(
            [
                r.get("entity", ""),
                r.get("table", ""),
                r.get("when", ""),
                r.get("value", ""),
                r.get("count", ""),
            ]
        )

    widths = [len(h) for h in headers]
    for r in table_rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def fmt_row(cols: list[str]) -> str:
        return " | ".join(cols[i].ljust(widths[i]) for i in range(len(headers))).rstrip()

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    for r in table_rows:
        line = fmt_row(r)
        if color:
            line = f"\x1b[31m{line}\x1b[0m"
        lines.append(line)
    return "\n".join(lines) + "\n"


def render_simple_table(
    *,
    headers: list[str],
    rows: list[list[str]],
    color: bool = False,
    color_code: str = "31",
) -> str:
    def _split_cell(text: str) -> list[str]:
        parts = text.split("\n")
        return parts if parts else [""]

    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], max((len(p) for p in _split_cell(c)), default=0))

    def fmt_row(cols: list[str]) -> str:
        return " | ".join(cols[i].ljust(widths[i]) for i in range(len(headers))).rstrip()

    def fmt_row_multiline(cols: list[str]) -> list[str]:
        col_lines = [_split_cell(c) for c in cols]
        height = max(len(x) for x in col_lines)
        out: list[str] = []
        for line_idx in range(height):
            physical = [
                (col_lines[col_idx][line_idx] if line_idx < len(col_lines[col_idx]) else "")
                for col_idx in range(len(headers))
            ]
            out.append(fmt_row(physical))
        return out

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    for r in rows:
        physical_lines = fmt_row_multiline(r)
        if color:
            physical_lines = [f"\x1b[{color_code}m{ln}\x1b[0m" for ln in physical_lines]
        lines.extend(physical_lines)
    return "\n".join(lines) + "\n"
