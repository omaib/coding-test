# Refactor Task: Introduce `BaseCNN` in `cnn.py`

## Background

The current `cnn.py` module defines multiple CNN variants that share similar initialization logic, layers, and utility methods. This duplication makes it harder to maintain or extend the module consistently.

## Objective

Create a reusable `BaseCNN` class that captures the shared behaviour across the existing CNN models and update every class in `cnn.py` to inherit from this new base while preserving their public APIs.

## Key Requirements

- Highlight reusability by centralising shared CNN functionality in `BaseCNN` and removing duplicated code from subclasses.
- Ensure each existing CNN class subclasses `BaseCNN` and only overrides or extends what is unique to that architecture.
- Keep backwards compatibility: constructor signatures, expected inputs/outputs, and module-level exports must remain unchanged for downstream users.
- Add inline documentation or docstrings clarifying the responsibilities of `BaseCNN` and any overridden methods.
- Update or add unit tests to cover the refactored structure, demonstrating that existing behaviours remain intact.

## Suggested Steps

1. Audit the current classes in `cnn.py` to map shared logic versus model-specific differences.
2. Sketch the `BaseCNN` interface (initialiser arguments, required abstract methods).
3. Introduce `BaseCNN`, migrate shared code into it, and adapt the concrete subclasses.
4. Run the project’s test suite (or targeted CNN tests) to confirm the refactor has not changed behaviour.
   - Execute `pre-commit run --all-files` to enforce formatting and linting.
   - Execute `pytest` (or the targeted CNN suite) to verify runtime correctness.

## Environment Setup

### Create a Python Environment and Install Required Packages

Choose one of the following approaches before running checks:

- **Virtualenv**
  1. `python3.12 -m venv .venv`
  2. `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)
  3. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` (adjust URL for CUDA if needed)
  4. `pip install pre-commit pytest`

- **Conda**
  1. `conda create -n omaib python=3.12`
  2. `conda activate omaib`
  3. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` (adjust for CUDA if needed)
  4. `pip install pre-commit pytest`

- **uv**
  1. `uv venv`
  2. `source .venv/bin/activate` (uv reuses `.venv`)
  3. `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
  4. `uv pip install pre-commit pytest`

### Run `pre-commit` and `pytest`

Under the root of the repository:

1. `pre-commit install` (if not already done)
2. `pre-commit run --all-files`
3. `pytest`

## Acceptance Criteria

- All CNN classes in `cnn.py` derive from `BaseCNN`, with shared code consolidated in the base class.
- Repository quality checks succeed: `pre-commit run --all-files` and `pytest`.
- Documentation (code comments or module docstring) clearly explains the new hierarchy.
