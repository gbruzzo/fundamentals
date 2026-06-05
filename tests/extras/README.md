# `tests/extras/` - smoke tests for extras orchestrators

Subprocess smoke tests that run every `extras/<topic>/visualize_<topic>.py`
script with `--save` under `MPLBACKEND=Agg`.

| File | Coverage |
|---|---|
| [`test_smoke.py`](test_smoke.py) | Registry-driven smoke coverage for every slug returned by `active_inference.extra_topics.extra_topic_slugs()`. |

## Running

```bash
pytest tests/extras -v
```

## What's Checked

For each extras topic:

1. The script imports `active_inference` through the same subprocess path a
   user would use.
2. The script accepts `--save`.
3. The script exits with code 0.
4. A fresh NPZ+JSON sidecar pair is written under
   `output/data/extras/<topic>/`.

Rendered PNG content is checked by
`scripts/validate_rendered_figures.py --root output/figures`.
