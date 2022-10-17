.PHONY: check-docs install-from-source clean

check-docs:
	bash ./bin/tests/check_docs.sh

install-from-source:
	pip uninstall catalyst -y && pip install -e ./
