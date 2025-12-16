# ha-replace-with-history

Compares two Home Assistant `entity_id`s in:
- Stage 1: entity analysis from the entity registry (`.storage/core.entity_registry`)
- Stage 2: state analysis from the SQLite recorder DB (`home-assistant_v2.db`)
- Stage 3: statistics analysis from the SQLite recorder DB (`home-assistant_v2.db`)
- Stage 4: statistics generation (writes `update.sql`, optionally applies it)

## Usage

Run via module:

```powershell
python -m ha_replace_with_history --storage .\.storage sensor.sun_next_dawn sensor.sun_next_dusk
```

Include DB summary (default DB path is `./home-assistant_v2.db`):

```powershell
python -m ha_replace_with_history --db .\home-assistant_v2.db --storage .\.storage sensor.sun_next_dawn sensor.sun_next_dusk
```

Install editable + run console script:

```powershell
pip install -e .
ha-replace-with-history --storage .\.storage sensor.sun_next_dawn sensor.sun_next_dusk
```

Exit codes:
- `0`: both entities found
- `2`: one or both entities missing (report still printed)

Columns:

Nested dict fields are flattened into path rows like `options/a`. Lists are included as JSON.

## DB Summary

The second table summarizes DB rows for each entity in:
- `states`
- `statistics`
- `statistics_short_term`

Each section includes:
- `row_count`
- `earliest/ts` + `earliest/value`
- `latest/ts` + `latest/value`
- `unavailable_count` (for `states`: `state IS NULL` or empty string; for statistics: all candidate value columns are null)
- `earliest_available/ts` + `earliest_available/value` (ignores unavailable rows)
- `latest_available/ts` + `latest_available/value` (ignores unavailable rows)

If `earliest_available` equals `earliest` (same row/value), it is omitted (same for `latest_available`).
If `unavailable_count` is between 1 and 200, the report also prints `unavailable_occurrences`.
- Timestamps are shown in your local timezone (no offset).
- Consecutive unavailable rows are compressed into ranges: `from - to`.

Unavailable occurrences are printed as a separate report at the end; each row includes the timestamp (or range), the unavailable state value (exact DB string; only `NULL` is printed as `NULL`), and the number of rows in the run.

For `total_increasing` entities, statistics summary values include the `sum` column in parentheses.

For `total_increasing` entities, the tool also prints:
- Reset events: where the numeric value decreases vs the previous row (for `states`, and for statistics tables preferring `state` when present).

The tool also prints:
- Missing statistics table rows: gaps in `statistics` (expected 1h) and `statistics_short_term` (expected 5m), reported as `before_ts (state/sum) - after_ts (state/sum) [N rows]`.

Unavailable occurrences are printed last.

Note: for statistics, the tool currently queries by `statistic_id == entity_id`.

## Stage 3: Statistics rebuild (total_increasing only)

## Stage 4: Statistics generation

When both entities share a supported `state_class` (`total_increasing` or `measurement`) and the precondition passes (`new` states start after `old` statistics end), the tool generates replacement statistics rows for `new_entity_id`.

Stage 4 writes `update.sql` in the current working directory. The script recreates TEMP VIEWs named:
- `statistics_generated`
- `statistics_short_term_generated`

Then it deletes+inserts rows in `statistics` and `statistics_short_term` for `new_entity_id`.

Stage 4 flag:
- `--new-entity-started-from-0={true|false}` (default: `true`): for `total_increasing`, controls whether the first generated `sum` includes the first reading from the new entity (treating it as a restart from 0).

## Getting the DB from HAOS (LAN)

SQLite is a local file; you generally don't "connect" to it remotely. The safe workflow is:
- Copy `home-assistant_v2.db` off the HAOS box to your machine, then run the tool against the local copy.
- Avoid opening the live DB over a network share; copy it first.

Practical options:
- Home Assistant UI: create a full backup and download it, then extract `home-assistant_v2.db` from the backup.
- Samba share add-on: copy `/config/home-assistant_v2.db` to your PC.
- Terminal & SSH add-on: copy via `scp`/SFTP. Optionally create a consistent snapshot first:

```sh
sqlite3 /config/home-assistant_v2.db ".backup /config/home-assistant_v2.db.backup"
```

Then copy `/config/home-assistant_v2.db.backup`.
