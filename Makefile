ifneq ($(origin MAKEFILE_LIST),file)
$(error MAKEFILE_LIST must not be overridden)
endif
override ROOT := $(shell path='$(subst ','"'"',$(MAKEFILE_LIST))'; path=$$(printf '%s' "$$path" | /usr/bin/sed 's/^ //'); /usr/bin/dirname -- "$$path")

.PHONY: all audit build check lint lock lock-check lock-upgrade root-test run runtime-check smoke static-check test verify

PYTHON ?= python3
UV ?= uv
PYPI_INDEX := https://pypi.org/simple

# Build the app (compile maintained Python modules)
build:
	cd "$(ROOT)" && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -c "import pathlib; [compile(pathlib.Path(path).read_text(), path, 'exec') for path in ('components/common.py', 'components/recommendations.py', 'customer_cluster.py', 'pages/4_🤞_Recommendations.py', 'scripts/check-runtime-imports.py', 'scripts/smoke-streamlit.py', 'test_app.py', 'test_embedding_cache.py', 'utils/crawler.py', 'utils/embedding_cache.py', 'utils/generate.py', 'utils/token.py')]"

# Run the app locally
run:
	cd "$(ROOT)" && streamlit run 👋_Hello.py

# Test the app
test:
	cd "$(ROOT)" && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -q test_app.py test_crawler.py test_embedding_cache.py
	cd "$(ROOT)" && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/test-embedding-cache-mutations.py

static-check:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) "$(ROOT)/scripts/check-workshop-baseline.py"

root-test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) "$(ROOT)/scripts/test-makefile-root.py"

lock:
	cd "$(ROOT)" && UV_INDEX_URL="$(PYPI_INDEX)" $(UV) pip compile requirements.in --python-version 3.12 --universal --generate-hashes --quiet --output-file requirements.txt
	cd "$(ROOT)" && UV_INDEX_URL="$(PYPI_INDEX)" $(UV) pip compile requirements-test.in --python-version 3.12 --universal --generate-hashes --quiet --output-file requirements-test.txt

lock-upgrade:
	cd "$(ROOT)" && UV_INDEX_URL="$(PYPI_INDEX)" $(UV) pip compile requirements.in --python-version 3.12 --universal --generate-hashes --upgrade --quiet --output-file requirements.txt
	cd "$(ROOT)" && UV_INDEX_URL="$(PYPI_INDEX)" $(UV) pip compile requirements-test.in --python-version 3.12 --universal --generate-hashes --upgrade --quiet --output-file requirements-test.txt

lock-check: lock
	git -C "$(ROOT)" diff --exit-code -- requirements.txt requirements-test.txt

audit:
	cd "$(ROOT)" && PIP_INDEX_URL="$(PYPI_INDEX)" pip-audit --no-deps --disable-pip -r requirements-test.txt
	cd "$(ROOT)" && PIP_INDEX_URL="$(PYPI_INDEX)" pip-audit --no-deps --disable-pip -r requirements.txt

runtime-check:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) "$(ROOT)/scripts/check-runtime-imports.py"

smoke:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) "$(ROOT)/scripts/smoke-streamlit.py"

lint: static-check

verify: lint test root-test

check: verify

# Build test and run the app
all: build test run
