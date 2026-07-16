# Sync Tank Archive Notes

Two archives were created before restructuring this work around the cloned GitHub repository.

Local tank-node work tar archive:

```text
/home/one/Projects/archives/sync-tank-local-20260716-130407.tar.gz
```

Fresh GitHub baseline archive:

```text
/home/one/Projects/archives/sync-tank-github-baseline-20260716-130407.tar.gz
```

The cloned baseline repository remains at:

```text
/home/one/Projects/sync-tank-repo
```

The previous local tank-node working tree remains intact at:

```text
/home/one/Projects/sync-tank
```

The new canonical repo structure places the current tank-node application under:

```text
tank
```

The original GitHub repo's legacy script directories were moved into the repository's archive folder:

```text
archive/one
archive/two
archive/three
archive/zero
```

Runtime data was intentionally not copied into the repo:

- `test_uploads/`
- `logs/`
- `*.log`
- `config/ingest_state.json`
- `config/cameras.json`
- `config/tank_layout.json`
- Python caches
- virtual environments
