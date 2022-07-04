# this target runs checks on all files
quality:
	isort . -c
	flake8
	mypy
	pydocstyle
	black --check .

# this target runs checks on all files and potentially modifies some of them
style:
	isort .
	black .

# Run tests for the library
test:
	coverage run -m pytest tests/

# Check that docs can build
docs:
	sphinx-build docs/source docs/_build -a

docker:
	docker build . -t pyrovision:python3.8.1-slim
