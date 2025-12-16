from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EntityRegistry:
    path: Path
    entities_by_entity_id: dict[str, dict[str, Any]]


class EntityRegistryError(RuntimeError):
    pass


def load_entity_registry(*, storage_dir: Path, entity_registry_file: Path | None = None) -> EntityRegistry:
    """Load Home Assistant entity registry from `.storage/core.entity_registry` (or override).

    `entity_registry_file` may be absolute or relative; if relative, it is resolved relative to `storage_dir`.
    """
    storage_dir = storage_dir.expanduser()

    if entity_registry_file is None:
        registry_path = storage_dir / "core.entity_registry"
    else:
        registry_path = entity_registry_file.expanduser()
        if not registry_path.is_absolute():
            registry_path = storage_dir / registry_path

    try:
        raw = registry_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise EntityRegistryError(f"Entity registry file not found: {registry_path}") from exc
    except OSError as exc:
        raise EntityRegistryError(f"Failed to read entity registry file: {registry_path} ({exc})") from exc

    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EntityRegistryError(f"Invalid JSON in entity registry file: {registry_path} ({exc})") from exc

    if not isinstance(doc, dict):
        raise EntityRegistryError(f"Unexpected JSON root in entity registry file: {registry_path} (expected object)")

    data = doc.get("data")
    if not isinstance(data, dict):
        raise EntityRegistryError(
            f"Unexpected entity registry shape: {registry_path} (expected object at key 'data')"
        )

    entities = data.get("entities")
    if not isinstance(entities, list):
        raise EntityRegistryError(
            f"Unexpected entity registry shape: {registry_path} (expected list at key 'data.entities')"
        )

    entities_by_entity_id: dict[str, dict[str, Any]] = {}
    for entry in entities:
        if not isinstance(entry, dict):
            continue
        entity_id = entry.get("entity_id")
        if isinstance(entity_id, str) and entity_id:
            entities_by_entity_id[entity_id] = entry

    return EntityRegistry(path=registry_path, entities_by_entity_id=entities_by_entity_id)


def get_entity(registry: EntityRegistry, entity_id: str) -> dict[str, Any] | None:
    return registry.entities_by_entity_id.get(entity_id)
