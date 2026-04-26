PY_NOTEBOOK  := src/notebooks/kaggle_run.py
IPYNB        := src/notebooks/kaggle_run.ipynb

SSL_KD_PY    := src/notebooks/SSL_KD.py
SSL_KD_IPYNB := src/notebooks/SSL_KD.ipynb

-include .env
export

.PHONY: notebook nb ssl-kd lint push-nb env-check help

## Convert kaggle_run.py -> kaggle_run.ipynb (primary target)
notebook nb: $(PY_NOTEBOOK)
	jupytext --update --to notebook $(PY_NOTEBOOK) -o $(IPYNB)
	@echo "OK  $(IPYNB) updated ($(shell python3 -c \
	  "import json; nb=json.load(open('$(IPYNB)')); \
	   cc=[c for c in nb['cells'] if c['cell_type']=='code']; \
	   mc=[c for c in nb['cells'] if c['cell_type']=='markdown']; \
	   print(f'{len(cc)} code + {len(mc)} markdown cells')"))"

## Convert SSL_KD.py -> SSL_KD.ipynb
ssl-kd: $(SSL_KD_PY)
	jupytext --update --to notebook $(SSL_KD_PY) -o $(SSL_KD_IPYNB)
	@echo "OK  $(SSL_KD_IPYNB) updated ($(shell python3 -c \
	  "import json; nb=json.load(open('$(SSL_KD_IPYNB)')); \
	   cc=[c for c in nb['cells'] if c['cell_type']=='code']; \
	   mc=[c for c in nb['cells'] if c['cell_type']=='markdown']; \
	   print(f'{len(cc)} code + {len(mc)} markdown cells')"))"

## Convert kaggle_run.ipynb -> kaggle_run.py (sync outputs back if needed)
sync-py: $(IPYNB)
	jupytext --to py:percent $(IPYNB) -o $(PY_NOTEBOOK)
	@echo "OK  $(PY_NOTEBOOK) synced from notebook"

## Run syntax + logic dry-run on kaggle_run.py without executing
lint:
	python3 src/scripts/lint_notebook.py $(PY_NOTEBOOK)

## Commit and push all .py sources + regenerated .ipynb files
push-nb: notebook ssl-kd
	git add $(PY_NOTEBOOK) $(IPYNB) $(SSL_KD_PY) $(SSL_KD_IPYNB)
	git commit -m "Update Kaggle notebooks (auto-generated from .py sources)"
	git push

## Verify required env vars are set
env-check:
	@test -n "$$KAGGLE_API_TOKEN" || (echo "KAGGLE_API_TOKEN not set -- copy .env.example to .env"; exit 1)
	@echo "OK  KAGGLE_API_TOKEN set"
	@test -n "$$WANDB_API_KEY" && echo "OK  WANDB_API_KEY set" || echo "WARN  WANDB_API_KEY not set (WandB logging disabled)"

help:
	@echo ""
	@echo "  make notebook   Convert kaggle_run.py -> kaggle_run.ipynb"
	@echo "  make ssl-kd     Convert SSL_KD.py -> SSL_KD.ipynb"
	@echo "  make sync-py    Sync .ipynb outputs back to kaggle_run.py"
	@echo "  make lint       Dry-run syntax + logic checks on kaggle_run.py"
	@echo "  make push-nb    notebook + ssl-kd + git commit + push"
	@echo "  make env-check  Verify .env credentials are loaded"
	@echo ""
