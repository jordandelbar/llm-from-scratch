precommit-install:
	pre-commit install

precommit-run-all:
	pre-commit run --all

run:
	uv run ./src/main.py
