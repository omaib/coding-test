# Refactor Task: Introduce `BaseCNN` in `cnn.py`

## Background

The current `cnn.py` defines multiple CNN models that share similar initialisation logic, layers, and utility methods.
This duplication makes the codebase harder to maintain and extend consistently.

## Objective

Refactor `cnn.py` by introducing a reusable `BaseCNN` class that captures shared functionality across CNN variants.
Each CNN model should inherit from this base class while preserving its current behaviour and public API.
This task evaluates your ability to design for **reusability**, **efficiency** and **clarity**.

## Key Requirements

- **Reusability**: Move the shared CNN logic into a new `BaseCNN` and remove duplicate code from subclasses.
- **Inheritance**: Each CNN model should inherit from `BaseCNN`, overriding only model-specific parts.
- **Compatibility**: Existing APIs, inputs/outputs, and model behaviour must remain unchanged.
- **Documentation**: Add clear docstrings for `BaseCNN` and all models.
- **Testing**: Ensure all tests pass or update `test_cnn.py` to confirm identical behaviour.

## Suggested Steps

1. Review the CNN classes in `cnn.py` and identify shared logic.
2. Define and implement the `BaseCNN` class.
3. Refactor existing CNN models.
4. Update and run tests to confirm correctness.

## Environment Setup

1. **Create a Python environment** (version 3.10-3.12) using any tool you prefer, such as:
   - Conda: `conda create -n omaib python=3.12 && conda activate omaib`
   - venv: `python3.12 -m venv .venv && source .venv/bin/activate`

2. **Install required packages:**
   - PyTorch and Torchvision (adjust the `index-url` parameter if you wish to use a GPU build):
      ```bash
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      ```
   - Pre-commit and pytest:
      ```bash
      pip install pre-commit pytest
      ```
## Validate code quality and tests
From the root of the repository:
1. Execute `pre-commit install` to set up pre-commit (only required once).
2. Execute `pre-commit run --all-files` to check code style and formatting.
3. Execute `pytest` to verify functional correctness.
