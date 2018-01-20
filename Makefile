help:
	@echo "Usage:"
	@echo "    make help        show this message"
	@echo "    make init        create virtual environment and install dependencies"
	@echo "    make test        run the test suite"

init:
	pip install pipenv
	pipenv install --dev

test:
	pipenv run pytest

.PHONY: help init activate test
