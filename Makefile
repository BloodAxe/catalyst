.PHONY: check-docs docker docker-fp16 docker-dev docker-dev-fp16 install-from-source clean

check-docs:
	bash ./bin/tests/check_docs.sh

install-from-source:
	pip uninstall catalyst -y && pip install -e ./
