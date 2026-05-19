.PHONY: audit audit-strict audit-run

audit:
	conda run -n dl python scripts/audit_figures.py

audit-strict:
	conda run -n dl python scripts/audit_figures.py --strict

audit-run:
	conda run -n dl python scripts/audit_figures.py --run-id $(RUN_ID)
