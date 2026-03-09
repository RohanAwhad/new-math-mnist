# Devlogs

## 2026-03-09 - Package API for git installs

- Added `new_math_ops` import namespace for package consumers.
- Exposed stable API for prompts, parser, dataset loading, evaluation helpers, and client.
- Updated packaging config in `pyproject.toml` for setuptools build + module/package inclusion.
- Added `tests/test_package_api.py` to lock import-level behavior.
- Documented git dependency installation in `README.md`.
