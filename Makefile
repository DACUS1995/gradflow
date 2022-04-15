.PHONY: init test

init:
	pip -m install -r requirements.txt

format:
	autoflake --remove-all-unused-imports -i **/*.py
	isort **/*.py
	black **/*.py

lint: ## [Local development] Run mypy, pylint and black
	python -m mypy gradflow
	python -m pylint gradflow

typecheck:
	python -m mypy gradflow tests

test:
	python -m unittest

clean:
	rm -rfv **/__pycache__ && echo
	rm -rfv **/.ipynb_checkpoints && echo
	rm -rfv **/.mypy_cache && echo