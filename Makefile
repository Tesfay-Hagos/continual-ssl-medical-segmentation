PY_NOTEBOOK  := src/notebooks/kaggle_run.py
IPYNB        := src/notebooks/kaggle_run.ipynb

.PHONY: notebook nb lint push-nb help

## Convert kaggle_run.py → kaggle_run.ipynb (primary target)
notebook nb: $(PY_NOTEBOOK)
	jupytext --to notebook $(PY_NOTEBOOK) -o $(IPYNB)
	@echo "✅  $(IPYNB) updated ($(shell python3 -c \
	  "import json; nb=json.load(open('$(IPYNB)')); \
	   cc=[c for c in nb['cells'] if c['cell_type']=='code']; \
	   mc=[c for c in nb['cells'] if c['cell_type']=='markdown']; \
	   print(f'{len(cc)} code + {len(mc)} markdown cells')"))"

## Convert kaggle_run.ipynb → kaggle_run.py (sync outputs back if needed)
sync-py: $(IPYNB)
	jupytext --to py:percent $(IPYNB) -o $(PY_NOTEBOOK)
	@echo "✅  $(PY_NOTEBOOK) synced from notebook"

## Run syntax + logic dry-run on kaggle_run.py without executing
lint:
	python3 src/scripts/lint_notebook.py $(PY_NOTEBOOK)

## Commit and push the .py source + regenerated .ipynb
push-nb: notebook
	git add $(PY_NOTEBOOK) $(IPYNB)
	git commit -m "Update Kaggle notebook (auto-generated from kaggle_run.py)"
	git push

help:
	@echo ""
	@echo "  make notebook   Convert kaggle_run.py → kaggle_run.ipynb"
	@echo "  make sync-py    Sync .ipynb outputs back to .py"
	@echo "  make lint       Dry-run syntax + logic checks on .py"
	@echo "  make push-nb    lint + notebook + git commit + push"
	@echo ""
