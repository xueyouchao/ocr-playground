# Everything the playground needs, in the order it needs it.
#
# The site build (`pages`, `site`, `serve`) is stdlib-only and runs on the system python. The upload
# service and the tests need the venv, because the service is wired with python-dependency-injector.
#
# Where things live: `data/` is what a run produced (the scan, the roster, the records), `site/` is
# what the build publishes (template, page images, and the two files it generates), `scripts/` holds
# the entry points (and `scripts/runners/`), `src/` the package itself, `deploy/` the two ways it ships.
# `src/ocr_playground/interfaces/site/layout.py` is the single place that names the two data directories.
PY        ?= python3
VENV      ?= .venv
VENV_PY   ?= $(VENV)/bin/python
PYTEST    ?= $(VENV)/bin/pytest
PORT      ?= 8080
SITE_DIR  ?= /home/ubuntu/sites/ocr-playground
COCOINDEX ?= cocoindex

.PHONY: help venv test pages records site serve publish api index clean

help:
	@echo "make venv     create .venv and install the runtime + test dependencies"
	@echo "make test     run the suite (offline: no GPU, no LiteParse, no poppler)"
	@echo "make pages    render data/corpus/document.pdf -> /tmp/ocr-playground/pages + site/pages"
	@echo "make records  re-run a parser (see README); data/records/ is already populated"
	@echo "make site     data/records/ -> site/evidence.json -> site/index.html (stdlib-only)"
	@echo "make serve    serve site/ on :$(PORT)"
	@echo "make publish  copy site/index.html + site/pages/ to $(SITE_DIR) (see deploy/)"
	@echo "make api      run the upload service on :8090 (needs the venv; CHANDRA_OCR_BASE_URL=… for the default)"
	@echo "make index    read OCR_CORPUS into out/ incrementally (needs requirements-cocoindex.txt)"
	@echo "make clean    drop site/evidence.json and site/index.html"

venv:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements-dev.txt

test:
	@test -x $(PYTEST) || { echo "run 'make venv' first"; exit 1; }
	$(PYTEST) -q

pages:
	$(PY) scripts/prepare.py

# data/records/ is already populated. Re-running a parser is a per-parser command with its own
# environment (see README), so this target explains itself rather than exiting 0 having done nothing.
records:
	@echo "data/records/ is populated; re-run one parser from scripts/runners/ or deploy/modal/ (README 'Run it'), then 'make site'"

site:
	$(PY) scripts/collect.py
	$(PY) scripts/build.py

serve:
	$(PY) -m http.server $(PORT) --bind 0.0.0.0 --directory site

api:
	@test -x $(VENV_PY) || { echo "run 'make venv' first"; exit 1; }
	$(VENV_PY) scripts/api.py --port 8090

index:
	$(COCOINDEX) update scripts/index.py

publish: site
	# two explicit syncs: `rsync -a --delete site/index.html site/pages/ dest/` would flatten pages/
	# into the site root and then delete the real directory
	rsync -a site/index.html $(SITE_DIR)/
	rsync -a --delete site/pages/ $(SITE_DIR)/pages/

clean:
	rm -f site/evidence.json site/index.html
