# this target runs checks on all files
quality:
	isort . -c
	flake8 ./
	mypy
	pydocstyle pyrovision/
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
	cd docs && bash build.sh

# Run the Gradio demo
run-demo:
	python demo/app.py --port 8080
