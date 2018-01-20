help:
	@echo "Usage:"
	@echo "    make help        show this message"
	@echo "    make init        create virtual environment and install dependencies"
	@echo "    make test        run the test suite"

init:
	pip install pipenv
	pipenv install --dev

test:
	pipenv run py.test -n auto

.PHONY: help init test
