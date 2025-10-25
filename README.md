# Refactor Task: Introduce `BaseCNN` in `cnn.py` and Refactor Existing CNN Classes

## Objective

The current `cnn.py` implements multiple CNN classes that share similar initialisation logic, layers, and utility methods. Your task is to introduce a reusable `BaseCNN` class that captures shared functionality across the CNN variants, and refactor the remaining classes to enhance code maintainability and reduce redundancy.

Each existing CNN class should inherit from BaseCNN while preserving its current behaviour and public API.

This task assesses your ability to design for **reusability**, **efficiency** and **clarity**, as well as apply good software engineering practices: using pre-commit hooks and writing effective tests.

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

## Validate Code Quality and Tests

From the root of the repository, run the following commands in your terminal:

1. Install pre-commit hooks (only required once):

   ```bash
   pre-commit install
   ```

2. Run pre-commit checks for code style and formatting on all files:

   ```bash
   pre-commit run --all-files
   ```

3. Run tests cases to verify functionality:

   ```bash
   pytest
   ```
