.PHONY: test compile preflight audit

compile:
	python -m compileall -q src tools

test:
	pytest -q

preflight: compile test
	python tools/preflight/environment_preflight.py

audit:
	bash scripts/01_audit_all.sh
