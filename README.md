# Refactor Task: Introduce `BaseCNN` in `cnn.py`

## Background

The current `cnn.py` defines multiple CNN models that share similar initialisation logic, layers, and utility methods.
This duplication makes the codebase harder to maintain and extend consistently.

## Objective

Refactor `cnn.py` by introducing a reusable `BaseCNN` class that captures shared functionality across CNN variants.
Each CNN model should inherit from this base while preserving its current behaviour and public API.
This task evaluates your ability to design for **reusability** and **clarity**.

## Key Requirements

- **Reusability**: Move the shared CNN logic into a new `BaseCNN`. Remove duplicate code from subclasses.
- **Inheritance**: Each CNN model should inherit from `BaseCNN`, overriding only model-specific parts.
- **Compatibility**: Existing APIs, inputs/outputs, and model behaviour must remain unchanged.
- **Documentation**: Add short docstrings for `BaseCNN` and any overridden models.
- **Testing**: Ensure all tests pass or update `test_cnn.py` to confirm identical behaviour.

## Suggested Steps

1. Review CNN classes in `cnn.py` and identify shared logic.
2. Define the `BaseCNN` class.
3. Implement `BaseCNN` and refactor existing CNNs to inherit from it.
4. Run tests to confirm correctness.

## Environment Setup

1. **Create a Python 3.12 environment** using any tool you prefer, such as:
   - Conda: `conda create -n omaib python=3.12 && conda activate omaib`
   - venv: `python3.12 -m venv .venv && source .venv/bin/activate`

2. **Install required packages:**
   - PyTorch and Torchvision: Adjust `index-url` parameter for CUDA if needed
      ```bash
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      ```
   - Pre-commit and PyTest:
      ```bash
      pip install pre-commit pytest
      ```
## Run `pre-commit` and `pytest`

Under the root of the repository:

1. Execute `pre-commit install` to initialise pre-commit (if not already done)
2. Execute `pre-commit run --all-files` to test formatting
3. Execute `pytest` to verify functional correctness

## Submission

Before submitting your solution:

1. Clean your code, by removing unused files, debug prints, and temporary artefacts.
2. Ensure all tests and lint checks pass.
3. Compress your project folder into a single `.zip` file.
4. Upload the archive via the provided Google Form [link].
