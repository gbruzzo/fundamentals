# `tests/extras/` - agent guide

Smoke tests for extras topic orchestrators.

## Rules

1. Keep tests subprocess-based so `argparse`, import paths, `--save`, and raw
   data export all run as a user would invoke them.
2. Add each new extras topic slug to `TOPICS` in `test_smoke.py`.
3. Require fresh NPZ+JSON sidecars under `output/data/extras/<topic>/`.
4. Use `MPLBACKEND=Agg`, `PYTHONWARNINGS=error`, and a `src/` `PYTHONPATH`
   entry for every subprocess.

## Running

```bash
pytest tests/extras -v
```
