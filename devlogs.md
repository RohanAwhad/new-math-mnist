# Devlogs

## 2026-03-09 - Package API for git installs

- Added `new_math_ops` import namespace for package consumers.
- Exposed stable API for prompts, parser, dataset loading, evaluation helpers, and client.
- Updated packaging config in `pyproject.toml` for setuptools build + module/package inclusion.
- Added `tests/test_package_api.py` to lock import-level behavior.
- Documented git dependency installation in `README.md`.

## 2026-03-09 - Package-first implementation + shim modules

- Moved implementations for `contracts`, `prompts`, `llm_client`, `evaluate`, and `generate_dataset` into `new_math_ops/`.
- Converted top-level modules into compatibility shims that re-export from `new_math_ops`.
- Added `new_math_ops/generate_dataset.py` so dataset generation logic is package-owned.
- Updated package internals to use relative imports instead of top-level module imports.
