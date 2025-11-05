# mlFrmScratch

Minimal, readable implementations of machine learning algorithms written from first principles. Focus is on clarity, education, and small, testable modules suitable for learning and experimentation.

## Features
- Simple, well-documented implementations (linear regression, logistic regression, neural networks, gradient descent, etc.)
- Small, dependency-light codebase
- Example datasets and notebooks for hands-on learning
- Unit tests and reproducible examples

## Quickstart

1. Clone the repo
```bash
git clone https://github.com/sntoshsah/mlfromscratch.git
cd mlFrmScratch
```

2. Create a virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Run an example notebook or script
```bash
python code/lin_reg.py
# or open notebooks in Jupyter:
jupyter notebook
```

## Project layout
- README.md - project overview
- src/ - implementations of algorithms
- examples/ - runnable scripts demonstrating workflows
- notebooks/ - interactive exploration and demos
- tests/ - unit tests
- data/ - small example datasets (CSV)

## Usage examples
Train a simple model:
```bash
python examples/train_linear_regression.py --data data/simple.csv --epochs 100
```

Evaluate:
```bash
python examples/evaluate.py --model models/linreg.pkl --test data/test.csv
```

## Contributing
- Keep changes small and focused
- Add tests for new functionality
- Follow the code style in the repo
- Open an issue or a pull request with a clear description of changes

## Testing
Run tests with:
```bash
pytest -q
```

## License
This project is licensed under the MIT License. See LICENSE file for details.

For questions or issues, open an issue on the repository.