.PHONY: precommit-install precommit-run-all

precommit-install:
	@pre-commit install

precommit-run-all:
	@pre-commit run --all
