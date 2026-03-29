# Contributing to revid

Thanks for your interest in contributing!

## Getting Started

1. Fork the repo and clone it locally
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

3. Make sure FFmpeg is installed and in your PATH

## Development

### Running tests

```bash
pytest tests/ -v
```

### Linting

```bash
ruff check revid/ tests/
ruff format revid/ tests/
```

### Adding a new FFmpeg filter

1. Add the method to `revid/video.py`
2. Add a test in `tests/test_video.py`
3. Update the README feature table

### Adding a new AI engine

1. Create or update a file in `revid/engines/`
2. Use the `@register("type", "engine_name")` decorator
3. Follow the handler pattern: `(step: dict, input_dir: str, output_dir: str) -> None`
4. Update `revid/engines/registry.py` to import your module
5. Add a test in `tests/test_engines.py`

### Adding a new preset

1. Add the preset function to `revid/presets.py`
2. Register it in the `PRESETS` dict
3. Add a test in `tests/test_presets.py`
4. Update the README presets table

## Pull Requests

- Keep PRs focused on a single change
- Add tests for new features
- Make sure `ruff check` and `pytest` pass
- Update the README if you add user-facing features

## License

By contributing, you agree that your contributions will be licensed under the BSD 3-Clause License.
