from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .entity_registry import EntityRegistry, EntityRegistryError, get_entity, load_entity_registry
from .report import render_entity_registry_report


@dataclass(frozen=True)
class Stage1Result:
    old_entity: dict[str, object] | None
    new_entity: dict[str, object] | None
    old_state_class: str | None
    new_state_class: str | None


def _get_state_class(entity: dict[str, object] | None) -> str | None:
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


def run_entity_analysis(
    *,
    storage_dir: Path,
    entity_registry_file: Path | None,
    old_entity_id: str,
    new_entity_id: str,
    tick: str,
    color: bool,
) -> Stage1Result:
    print("*** Stage 1: Entity analysis")

    try:
        registry: EntityRegistry = load_entity_registry(
            storage_dir=storage_dir,
            entity_registry_file=entity_registry_file,
        )
    except EntityRegistryError as exc:
        raise SystemExit(str(exc))

    old = get_entity(registry, old_entity_id)
    new = get_entity(registry, new_entity_id)

    print("Entity analysis report:")
    report = render_entity_registry_report(
        old_entity_id=old_entity_id,
        new_entity_id=new_entity_id,
        old=old,
        new=new,
        tick=tick,
        color=color,
    )
    print(report, end="")

    return Stage1Result(
        old_entity=old,
        new_entity=new,
        old_state_class=_get_state_class(old),
        new_state_class=_get_state_class(new),
    )
