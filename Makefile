.PHONY: init test

init:
	pip install -r requirements.txt

format:
	autoflake --remove-all-unused-imports -i **/*.py
	isort **/*.py
	black **/*.py

typecheck:
	mypy src/ pytorch-lightning/ tests/

test:
	python -m tests.grad_engine_test

clean:
	rm -rfv **/__pycache__ && echo
	rm -rfv **/.ipynb_checkpoints && echo
	rm -rfv **/.mypy_cache && echo