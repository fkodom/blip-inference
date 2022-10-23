# BLIP

## Install

```bash
pip install "BLIP @ git+ssh://git@github.com/fkodom/BLIP.git"

# Install all dev dependencies (tests etc.)
pip install "BLIP[all] @ git+ssh://git@github.com/fkodom/BLIP.git"

# Setup pre-commit hooks
pre-commit install
```


## Test

Tests run automatically through GitHub Actions.
* Fast tests run on each push.
* Slow tests (decorated with `@pytest.mark.slow`) run on each PR.

You can also run tests manually with `pytest`:
```bash
pytest BLIP

# For all tests, including slow ones:
pytest --slow BLIP
```


## Release

[Optional] Requires either PyPI or Docker GHA workflows to be enabled.

Just tag a new release in this repo, and GHA will automatically publish Python wheels and/or Docker images.
