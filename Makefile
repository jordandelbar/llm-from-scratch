.PHONY: precommit-install precommit-run-all

precommit-install:
	@pre-commit install

precommit-run-all:
	@pre-commit run --all

download-the-verdict:
	@uv run src/download_text.py
