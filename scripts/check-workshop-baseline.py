#!/usr/bin/env python3
"""Static baseline checks for the OpenAI 102 workshop."""

from pathlib import Path
import ast
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MAKEFILE = """ifneq ($(origin MAKEFILE_LIST),file)
$(error MAKEFILE_LIST must not be overridden)
endif
override ROOT := $(shell path='$(subst ','"'"',$(MAKEFILE_LIST))'; path=$$(printf '%s' "$$path" | /usr/bin/sed 's/^ //'); /usr/bin/dirname -- "$$path")

.PHONY: all audit build check lint lock lock-check lock-upgrade root-test run runtime-check smoke static-check test verify

PYTHON ?= python3
UV ?= uv
PYPI_INDEX := https://pypi.org/simple

# Build the app (compile maintained Python modules)
build:
\tcd "$(ROOT)" && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -c "import pathlib; [compile(pathlib.Path(path).read_text(), path, 'exec') for path in ('components/common.py', 'components/recommendations.py', 'customer_cluster.py', 'pages/4_🤞_Recommendations.py', 'scripts/check-runtime-imports.py', 'scripts/smoke-streamlit.py', 'test_app.py', 'test_embedding_cache.py', 'utils/crawler.py', 'utils/embedding_cache.py', 'utils/generate.py', 'utils/token.py')]"

# Run the app locally
run:
\tcd "$(ROOT)" && streamlit run 👋_Hello.py

# Test the app
test:
\tcd "$(ROOT)" && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -q test_app.py test_crawler.py test_embedding_cache.py

static-check:
\tPYTHONDONTWRITEBYTECODE=1 $(PYTHON) "$(ROOT)/scripts/check-workshop-baseline.py"

lock:
\tcd "$(ROOT)" && UV_INDEX_URL="$(PYPI_INDEX)" $(UV) pip compile requirements.in --python-version 3.12 --universal --generate-hashes --quiet --output-file requirements.txt
\tcd "$(ROOT)" && UV_INDEX_URL="$(PYPI_INDEX)" $(UV) pip compile requirements-test.in --python-version 3.12 --universal --generate-hashes --quiet --output-file requirements-test.txt

lock-upgrade:
\tcd "$(ROOT)" && UV_INDEX_URL="$(PYPI_INDEX)" $(UV) pip compile requirements.in --python-version 3.12 --universal --generate-hashes --upgrade --quiet --output-file requirements.txt
\tcd "$(ROOT)" && UV_INDEX_URL="$(PYPI_INDEX)" $(UV) pip compile requirements-test.in --python-version 3.12 --universal --generate-hashes --upgrade --quiet --output-file requirements-test.txt

lock-check: lock
\tgit -C "$(ROOT)" diff --exit-code -- requirements.txt requirements-test.txt

audit:
\tcd "$(ROOT)" && PIP_INDEX_URL="$(PYPI_INDEX)" pip-audit --no-deps --disable-pip -r requirements-test.txt
\tcd "$(ROOT)" && PIP_INDEX_URL="$(PYPI_INDEX)" pip-audit --no-deps --disable-pip -r requirements.txt

runtime-check:
\tPYTHONDONTWRITEBYTECODE=1 $(PYTHON) "$(ROOT)/scripts/check-runtime-imports.py"

smoke:
\tPYTHONDONTWRITEBYTECODE=1 $(PYTHON) "$(ROOT)/scripts/smoke-streamlit.py"

root-test:
\tPYTHONDONTWRITEBYTECODE=1 $(PYTHON) "$(ROOT)/scripts/test-makefile-root.py"

lint: static-check

verify: lint test root-test

check: verify

# Build test and run the app
all: build test run
"""
CI_PLAN = "docs/plans/2026-06-10-hosted-workshop-validation.md"
DEPENDENCY_PLAN = "docs/plans/2026-06-12-supported-python-dependency-graph.md"
TRANSITIVE_SECURITY_PLAN = "docs/plans/2026-06-16-transitive-dependency-security-update.md"
UNIVERSAL_LOCK_AUDIT_PLAN = "docs/plans/2026-06-16-universal-lock-audit.md"
HASH_VERIFIED_LOCK_PLAN = "docs/plans/2026-06-17-hash-verified-universal-locks.md"
EMBEDDING_CACHE_PLAN = "docs/plans/2026-06-13-json-embedding-cache.md"
API_COMPATIBILITY_PLAN = "docs/plans/2026-06-13-openai-api-compatibility-notes.md"
LOCATION_INDEPENDENT_MAKE_PLAN = "docs/plans/2026-06-13-location-independent-make.md"
EMBEDDING_PAYLOAD_PLAN = "docs/plans/2026-06-14-embedding-payload-validation.md"
CUSTOMER_RECOMMENDATION_PLAN = "docs/plans/2026-06-15-customer-industry-recommendation.md"
PRODUCT_BACKED_RECOMMENDATION_PLAN = "docs/plans/2026-06-15-product-backed-recommendation.md"
PRODUCT_NAME_VALIDATION_PLAN = "docs/plans/2026-06-15-product-name-validation.md"
MALFORMED_CUSTOMER_PLAN = "docs/plans/2026-06-15-malformed-customer-entry-guard.md"
CUSTOMER_INDUSTRY_NAME_PLAN = "docs/plans/2026-06-15-customer-industry-name-validation.md"
RECOMMENDATION_CONTAINER_PLAN = "docs/plans/2026-06-15-recommendation-container-validation.md"
RECOMMENDATION_EMBEDDING_PLAN = "docs/plans/2026-06-15-recommendation-embedding-validation.md"
RECOMMENDATION_TIE_BREAK_PLAN = "docs/plans/2026-06-16-recommendation-tie-breaking.md"
FINETUNING_RETRY_PLAN = "docs/plans/2026-06-16-finetuning-rate-limit-retry.md"
STARLETTE_LOCK_PLAN = "docs/plans/2026-06-16-starlette-lock-floor-reproducibility.md"
SAFE_JSON_FIXTURE_PLAN = "docs/plans/2026-06-16-safe-json-embedding-fixtures.md"
CRAWLER_SSRF_PLAN = "docs/plans/2026-06-19-crawler-ssrf-boundary.md"
SAFE_MAKE_ROOT_PLAN = "docs/plans/2026-06-21-safe-make-root.md"
REQUIRED = [
    ".github/CODEOWNERS",
    ".github/workflows/check.yml",
    ".gitignore",
    "CHANGES.md",
    "Dockerfile",
    "Makefile",
    "README.md",
    "SECURITY.md",
    "VISION.md",
    "components/common.py",
    "components/recommendations.py",
    "customer_cluster.py",
    "pages/4_🤞_Recommendations.py",
    "docs/plans/2026-06-08-openai-102-workshop-baseline.md",
    "docs/plans/2026-06-09-vector-math-validation.md",
    "docs/plans/2026-06-09-small-embedding-fixtures.md",
    "docs/plans/2026-06-09-empty-embedding-fixtures.md",
    "docs/plans/2026-06-09-malformed-embedding-fixtures.md",
    "docs/plans/2026-06-09-embedding-metadata-text.md",
    "docs/plans/2026-06-09-finite-embedding-values.md",
    "docs/plans/2026-06-10-numeric-embedding-values.md",
    "docs/plans/2026-06-10-query-embedding-validation.md",
    "docs/plans/2026-06-12-vector-value-validation.md",
    DEPENDENCY_PLAN,
    TRANSITIVE_SECURITY_PLAN,
    UNIVERSAL_LOCK_AUDIT_PLAN,
    HASH_VERIFIED_LOCK_PLAN,
    EMBEDDING_CACHE_PLAN,
    API_COMPATIBILITY_PLAN,
    LOCATION_INDEPENDENT_MAKE_PLAN,
    CUSTOMER_RECOMMENDATION_PLAN,
    PRODUCT_BACKED_RECOMMENDATION_PLAN,
    PRODUCT_NAME_VALIDATION_PLAN,
    MALFORMED_CUSTOMER_PLAN,
    CUSTOMER_INDUSTRY_NAME_PLAN,
    RECOMMENDATION_CONTAINER_PLAN,
    RECOMMENDATION_EMBEDDING_PLAN,
    RECOMMENDATION_TIE_BREAK_PLAN,
    FINETUNING_RETRY_PLAN,
    STARLETTE_LOCK_PLAN,
    SAFE_JSON_FIXTURE_PLAN,
    CRAWLER_SSRF_PLAN,
    SAFE_MAKE_ROOT_PLAN,
    "docs/openai-api-compatibility.md",
    CI_PLAN,
    "docs/plans/2026-06-09-make-gate-aliases.md",
    "docs/plans/2026-06-09-bytecode-free-tests.md",
    "docs/readme-overview.svg",
    "requirements.in",
    "requirements.txt",
    "requirements-test.in",
    "requirements-test.txt",
    "scripts/check-runtime-imports.py",
    "scripts/smoke-streamlit.py",
    "scripts/test-makefile-root.py",
    "scripts/check-workshop-baseline.py",
    "test_app.py",
    "test_embedding_cache.py",
    "test_embeddings.json",
    "utils/crawler.py",
    "utils/embedding_cache.py",
    "utils/generate.py",
    "utils/token.py",
]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8", errors="replace")


def markdown_section(text, heading):
    match = re.search(
        rf"(?ms)^## {re.escape(heading)}\s*$\n(.*?)(?=^## |\Z)",
        text,
    )
    return match.group(1).strip() if match else ""


def tracked(paths):
    result = subprocess.run(
        ["git", "ls-files", *paths],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def main():
    failures = []
    for path in REQUIRED:
        if not (ROOT / path).is_file():
            failures.append(f"required file missing: {path}")

    for path in [
        "components/common.py",
        "components/recommendations.py",
        "customer_cluster.py",
        "pages/4_🤞_Recommendations.py",
        "test_app.py",
        "test_embedding_cache.py",
        "utils/crawler.py",
        "utils/embedding_cache.py",
        "utils/generate.py",
    ]:
        try:
            ast.parse(read(path), filename=path)
        except SyntaxError as error:
            failures.append(f"{path} must parse as Python: {error}")

    common = read("components/common.py")
    if 'api_token = "sk-"' in common or 'value="sk-"' in common:
        failures.append("sidebar token input must not default to an sk- prefix")
    if 'type="password"' not in common:
        failures.append("sidebar token input must be password typed")
    if 'os.environ["OPENAI_API_KEY"] = api_token_input' not in common:
        failures.append("sidebar token input must update OPENAI_API_KEY locally")

    recommendations = read("components/recommendations.py")
    for phrase in [
        "from collections.abc import Iterable, Mapping",
        "from utils.generate import cosine_similarity",
        "if isinstance(item, Mapping)",
        "customer_industry = customer.get(\"industry\")",
        "if not isinstance(customer_industry, str) or not customer_industry.strip():",
        "customer_scores = similarity_scores.get(customer_industry, {})",
        "for industry, score in customer_scores.items()",
        "if not isinstance(products, list):",
        "if isinstance(product, str) and product.strip()",
        "validated_products[industry] = product_names",
        "if not product_scores:",
        "top_industry = max(",
        "choose_product(validated_products[top_industry])",
        "return None, similarity_scores",
        "if not isinstance(industry_embeddings, Mapping):",
        "not isinstance(customer_data, Iterable)",
        "isinstance(customer_data, (str, bytes, Mapping))",
        "not isinstance(industry_products, Mapping)",
        "except (TypeError, ValueError, OverflowError):",
    ]:
        if phrase not in recommendations:
            failures.append(f"customer recommendation logic must retain {phrase}")
    if "list(sorted_scores.keys())[0]" in recommendations:
        failures.append("customer recommendations must not select the first mapping key")
    for phrase in [
        "product_scores[industry]",
        "industry == customer_industry",
    ]:
        if phrase not in recommendations:
            failures.append(f"recommendation tie breaking must retain {phrase}")
    tie_break = recommendations.split("top_industry = max(", 1)[-1].split(")\n    return", 1)[0]
    score_index = tie_break.find("product_scores[industry]")
    own_industry_index = tie_break.find("industry == customer_industry")
    if not 0 <= score_index < own_industry_index:
        failures.append("recommendation tie breaking must rank score before own-industry preference")

    tests = read("test_app.py")
    for phrase in [
        "def test_recommend_product_skips_malformed_customer_entries",
        '[None, "invalid", []]',
    ]:
        if phrase not in tests:
            failures.append(f"malformed customer coverage must retain {phrase}")

    malformed_customer_guidance = {
        "README.md": "Malformed customer-list members are ignored",
        "SECURITY.md": "ignore malformed customer-list members",
        "VISION.md": "Keep malformed customer-list members",
        "CHANGES.md": "Ignored malformed customer-list members",
    }
    for path, phrase in malformed_customer_guidance.items():
        if phrase not in read(path):
            failures.append(f"{path} must retain malformed customer guidance")

    for phrase in [
        "def test_recommend_product_rejects_invalid_customer_industry_names",
        '@pytest.mark.parametrize("industry", [None, [], {}, "", "   "])',
    ]:
        if phrase not in tests:
            failures.append(f"customer industry validation coverage must retain {phrase}")
    for path in ["README.md", "SECURITY.md", "VISION.md", "CHANGES.md"]:
        if "nonempty string customer industry" not in read(path).lower():
            failures.append(f"{path} must document nonempty string customer industry validation")
        if "recommendation container validation" not in read(path).lower():
            failures.append(f"{path} must document recommendation container validation")
    for phrase in [
        "def test_recommend_product_rejects_invalid_top_level_containers",
        "expected_score_keys",
        "(None, {\"Healthcare\"",
        "([], None, {}, set())",
        "([], {\"Healthcare\": [{\"embedding\": [1.0, 0.0]}]}, None",
    ]:
        if phrase not in tests:
            failures.append(f"recommendation container coverage must retain {phrase}")
    for phrase in [
        "def test_recommend_product_skips_invalid_embedding_pairs",
        '{"Healthcare": [{"embedding": [0.0, 0.0]}]}',
        '"Retail": [{"embedding": [1.0]}]',
        '{"Healthcare": [{"embedding": ["invalid", 1.0]}]}',
        '{"Healthcare": 1.0}',
        'scores.get("Healthcare", {}) == expected_healthcare_scores',
    ]:
        if phrase not in tests:
            failures.append(f"recommendation embedding coverage must retain {phrase}")
    for phrase in [
        "def test_recommend_product_prefers_customer_industry_on_equal_score",
        '"Retail": [{"embedding": [1.0, 0.0]}]',
        '"Healthcare": [{"embedding": [1.0, 0.0]}]',
        'assert product == "healthcare product"',
    ]:
        if phrase not in tests:
            failures.append(f"recommendation tie-break coverage must retain {phrase}")

    recommendation_page = read("pages/4_🤞_Recommendations.py")
    for phrase in [
        "from components.recommendations import recommend_product",
        "INDUSTRY_PRODUCTS = {",
        "customer_data,",
        "industry_embeddings,",
        "INDUSTRY_PRODUCTS,",
    ]:
        if phrase not in recommendation_page:
            failures.append(f"Recommendations page must retain {phrase}")

    generate = read("utils/generate.py")
    for phrase in [
        "def get_cache_file",
        "def validate_saved_embeddings",
        "def validate_query_embedding",
        "def validate_embedding_response",
        "hashlib.sha256",
        "os.path.commonpath",
        "get_cache_file(cache_folder, query)",
        "def _record_estimated_cost",
        "def cosine_similarity",
        "def _validate_vector_pair",
        "n_neighbors = min(5, len(embeddings_array))",
        "Cosine similarity is undefined for zero vectors.",
        "At least one embedding fixture row is required.",
        "Embedding fixture rows must have the same dimensionality.",
        "metadata must include text",
        "np.number",
        "numeric finite numbers",
        "math.isfinite",
        "finite numbers",
        "Query embedding must match the trained model dimensionality.",
        "Vectors must be non-empty numeric finite sequences.",
        "Embedding response items must have the same dimensionality.",
        "Embedding cache must be valid UTF-8 JSON.",
        'with open(json_file_path, encoding="utf-8") as file:',
        "saved_embeddings = json.load(file)",
        "Embedding fixture must be valid UTF-8 JSON.",
        "if not isinstance(saved_embeddings, list):",
        "Embedding fixture must be a JSON array.",
        "Embedding fixture not found:",
    ]:
        if phrase not in generate:
            failures.append(f"utils/generate.py must include {phrase}")
    if 'os.path.join(cache_folder, f"{query}.json")' in generate:
        failures.append("text embedding cache names must not use raw queries")
    if 'st_state[\'cost\'] = f"${0:.10f}"' in generate:
        failures.append("cost tracking must record the first request cost")
    embedding_validator = generate.split("def validate_embedding_response", 1)[-1].split(
        "def get_embeddings", 1
    )[0]
    for phrase in [
        "if not isinstance(data, list) or not data:",
        "isinstance(value, (bool, complex, np.complexfloating))",
        "math.isfinite(numeric_value)",
    ]:
        if phrase not in embedding_validator:
            failures.append(f"embedding response validator must retain {phrase}")
    cache_validation = generate.find("validate_embedding_response(cached_response)")
    cache_return = generate.find("return cached_response")
    response_validation = generate.find("validate_embedding_response(response_data)")
    response_write = generate.find("json.dump(response_data, f)")
    if cache_validation == -1 or cache_return == -1 or cache_validation > cache_return:
        failures.append("cached embeddings must be validated before return")
    if response_validation == -1 or response_write == -1 or response_validation > response_write:
        failures.append("API embeddings must be validated before cache write")
    if "pickle" in generate:
        failures.append("shared embedding fixture loading must not retain pickle")

    crawler = read("utils/crawler.py")
    for phrase in [
        "_IPV4_NON_GLOBAL_NETWORKS",
        "_IPV6_NON_GLOBAL_NETWORKS",
        "socket.socket(family, socket.SOCK_STREAM)",
        "server_hostname=self._target.hostname",
        "sock.getpeername()",
        "peer_address != self._target.address",
        "_response_html_encoding(response)",
        "decompressor.decompress(chunk, remaining + 1)",
        "deadline.timeout(READ_TIMEOUT)",
        "_get_response_with_deadline(connection, deadline)",
        "sock.shutdown(socket.SHUT_RDWR)",
        "response.close()",
        "abandoned.set()",
        "urljoin(current_url, location)",
    ]:
        if phrase not in crawler:
            failures.append(f"crawler public-network boundary must retain {phrase}")
    for forbidden in [".is_global", "requests.Session", "allow_redirects=True"]:
        if forbidden in crawler:
            failures.append(f"crawler public-network boundary must reject {forbidden}")

    text_search = read("pages/3_🔍_Text_Search.py")
    if "crawler.get_texts(url_list)" not in text_search:
        failures.append("text-search page must use the bounded batch crawler")

    langchain = read("pages/9_⛓️_Langchain.py")
    if "requests.get(url, timeout=15)" not in langchain:
        failures.append("LangChain tutorial requests must use a timeout")

    customer_cluster = read("customer_cluster.py")
    for forbidden in ["pickle.load", "pickle.dump", "embedding_cache.pkl"]:
        if forbidden in customer_cluster:
            failures.append(f"customer clustering must not retain {forbidden}")
    for phrase in [
        "EMBEDDING_CACHE_FILE",
        "load_embedding_cache(embedding_cache_file)",
        "save_embedding_cache(embedding_cache, embedding_cache_file)",
    ]:
        if phrase not in customer_cluster:
            failures.append(f"customer clustering must use {phrase}")

    embedding_cache = read("utils/embedding_cache.py")
    for phrase in [
        'Path("embedding_cache.json")',
        'read_text(encoding="utf-8")',
        "json.loads(serialized)",
        "isinstance(value, dict)",
        "isinstance(key, str) and isinstance(item, str)",
        'raise ValueError("embedding cache must be valid UTF-8 JSON") from None',
        'raise ValueError("embedding cache must contain string keys and values")',
        'path.with_suffix(path.suffix + ".tmp")',
        "temporary_path.replace(path)",
    ]:
        if phrase not in embedding_cache:
            failures.append(f"embedding cache helper must retain {phrase}")

    makefile = read("Makefile")
    if makefile != EXPECTED_MAKEFILE:
        failures.append(
            "Makefile must exactly preserve rooted workshop tooling and command overrides"
        )

    readme = read("README.md")
    location_independent_make_plan = read(LOCATION_INDEPENDENT_MAKE_PLAN)
    if "make -f /path/to/openai-102-workshop/Makefile check" not in readme:
        failures.append("README must document location-independent Makefile invocation")
    safe_make_root_plan = read(SAFE_MAKE_ROOT_PLAN)
    if not all(
        evidence in safe_make_root_plan
        for evidence in [
            "all fifteen public Make targets",
            "command-line and environment `ROOT`",
            "command-line and environment `MAKEFILE_LIST`",
        ]
    ):
        failures.append(
            "safe Make root plan must cover aliases and both override channels"
        )
    if not all(
        evidence in location_independent_make_plan.lower()
        for evidence in [
            "status: completed",
            "root and external-directory",
            "ten isolated hostile mutations",
        ]
    ):
        failures.append(
            "location-independent Make plan must record completed root, external, and mutation verification"
        )

    direct_requirements = {
        line for line in read("requirements.in").splitlines()
        if line and not line.startswith("#")
    }
    expected_direct_requirements = {
        "beautifulsoup4==4.14.3",
        "langchain-text-splitters==1.1.2",
        "langsmith==0.8.18",
        "matplotlib==3.10.9",
        "numpy==2.4.6",
        "openai==0.28.1",
        "pandas==3.0.3",
        "requests==2.34.2",
        "scikit-learn==1.9.0",
        "seaborn==0.13.2",
        "spacy==3.8.14",
        "starlette==1.3.1",
        "streamlit==1.58.0",
        "tiktoken==0.11.0",
    }
    if direct_requirements != expected_direct_requirements:
        failures.append("requirements.in must keep the reviewed direct application graph")

    direct_test_requirements = {
        line for line in read("requirements-test.in").splitlines()
        if line and not line.startswith("#")
    }
    expected_direct_test_requirements = {
        "langchain-text-splitters==1.1.2",
        "langsmith==0.8.18",
        "msgpack==1.2.1",
        "numpy==2.4.6",
        "openai==0.28.1",
        "pip-audit==2.10.0",
        "pytest==9.0.3",
        "requests==2.34.2",
        "scikit-learn==1.9.0",
        "tiktoken==0.11.0",
        "uv==0.11.19",
    }
    if direct_test_requirements != expected_direct_test_requirements:
        failures.append("requirements-test.in must keep the reviewed verification graph")

    application_lock = read("requirements.txt")
    test_lock = read("requirements-test.txt")
    for lock_name, lock in [("requirements.txt", application_lock), ("requirements-test.txt", test_lock)]:
        pin_count = 0
        hashed_pin_count = 0
        for line in lock.splitlines():
            if line and not line.startswith(("#", " ")):
                pin_count += 1
                if not re.match(r"^[A-Za-z0-9_.-]+==[^; ]+(?:\s*;.*?)? \\$", line):
                    failures.append(f"{lock_name} must contain only hash-annotated exact generated pins: {line}")
        hashed_pin_count = len(re.findall(
            r"(?m)^[A-Za-z0-9_.-]+==.* \\\n    --hash=sha256:[0-9a-f]{64}",
            lock,
        ))
        if pin_count == 0 or hashed_pin_count != pin_count:
            failures.append(f"{lock_name} must hash every exact generated pin")
        if "--python-version 3.12 --universal --generate-hashes" not in lock:
            failures.append(f"{lock_name} must record the hash-generating Python 3.12 universal compile contract")
        if "aiohttp==3.14.1" not in lock or "aiohttp==3.14.0" in lock:
            failures.append(f"{lock_name} must retain the reviewed aiohttp security update")
        if "langsmith==0.8.18" not in lock or "langsmith==0.8.9" in lock:
            failures.append(f"{lock_name} must retain the reviewed langsmith security update")

    if "msgpack==1.2.1" not in test_lock or "msgpack==1.1.2" in test_lock:
        failures.append("requirements-test.txt must retain the reviewed msgpack security update")

    for removed_package in ["torch", "transformers", "sentencepiece", "virtualenv", "python-dotenv"]:
        if re.search(rf"^{re.escape(removed_package)}==", application_lock, re.MULTILINE | re.IGNORECASE):
            failures.append(f"requirements.txt must not restore unused {removed_package}")
    for safe_pin in ["jinja2==3.1.6", "pyarrow==24.0.0", "pygments==2.20.0", "requests==2.34.2", "streamlit==1.58.0"]:
        if safe_pin not in application_lock:
            failures.append(f"requirements.txt must retain reviewed pin {safe_pin}")
    if "starlette==1.3.1" not in application_lock or "starlette==1.2.1" in application_lock:
        failures.append("requirements.txt must retain the reviewed starlette security update")

    for document, phrases in {
        "README.md": ["aiohttp==3.14.1", "starlette==1.3.1", "langsmith==0.8.18", "msgpack==1.2.1"],
        "SECURITY.md": ["`aiohttp` at 3.14.1", "`starlette` at 1.3.1", "`langsmith` at 0.8.18", "`msgpack` at 1.2.1"],
        "CHANGES.md": ["`aiohttp==3.14.1`", "`starlette==1.3.1`", "`langsmith==0.8.18`", "`msgpack==1.2.1`"],
    }.items():
        content = read(document)
        for phrase in phrases:
            if phrase not in content:
                failures.append(f"{document} must document the reviewed security floor {phrase}")

    token_helper = read("utils/token.py")
    if "from langchain_text_splitters import RecursiveCharacterTextSplitter" not in token_helper:
        failures.append("token helper must use the supported splitter package")
    if "from langchain.text_splitter" in token_helper:
        failures.append("token helper must not restore the full legacy LangChain import")

    runtime_import_check = read("scripts/check-runtime-imports.py")
    for distribution, module in {
        "beautifulsoup4": "bs4",
        "langchain-text-splitters": "langchain_text_splitters",
        "matplotlib": "matplotlib",
        "numpy": "numpy",
        "openai": "openai",
        "pandas": "pandas",
        "requests": "requests",
        "scikit-learn": "sklearn",
        "seaborn": "seaborn",
        "spacy": "spacy",
        "streamlit": "streamlit",
        "tiktoken": "tiktoken",
    }.items():
        if f'"{distribution}": "{module}"' not in runtime_import_check:
            failures.append(f"runtime import check must cover {distribution}")
    for phrase in ["sys.version_info[:2] != (3, 12)", "import_module(module)", "metadata.version(distribution)"]:
        if phrase not in runtime_import_check:
            failures.append(f"runtime import check must retain {phrase}")

    streamlit_smoke = read("scripts/smoke-streamlit.py")
    for phrase in [
        'environment.pop("OPENAI_API_KEY", None)',
        '"--server.headless=true"',
        '"--server.address=127.0.0.1"',
        'f"http://127.0.0.1:{port}/_stcore/health"',
        "TIMEOUT_SECONDS = 20",
        "process.terminate()",
    ]:
        if phrase not in streamlit_smoke:
            failures.append(f"Streamlit smoke must retain {phrase}")

    workflow = read(".github/workflows/check.yml")
    codeowners = read(".github/CODEOWNERS")
    for phrase in [
        "permissions:\n  contents: read",
        "cancel-in-progress: true",
        "runs-on: ubuntu-24.04",
        "timeout-minutes: 15",
        "actions/checkout@df4cb1c069e1874edd31b4311f1884172cec0e10",
        "persist-credentials: false",
        "actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405",
        'python-version: "3.12"',
        "python -m pip install -r requirements-test.txt",
        "python -m pip check",
        "make check",
        "make lock-check",
        "make audit",
        "python -m pip install -r requirements.txt",
        "make runtime-check",
        "make smoke",
    ]:
        if phrase not in workflow:
            failures.append(f"Check workflow must keep {phrase}")
    workflow_files = sorted(str(path.relative_to(ROOT)) for path in (ROOT / ".github/workflows").rglob("*") if path.is_file())
    if workflow_files != [".github/workflows/check.yml"]:
        failures.append("check.yml must be the repository's only hosted workflow")
    if codeowners.strip() != "* @garethpaul":
        failures.append("CODEOWNERS must assign the repository to @garethpaul")

    dockerfile = read("Dockerfile")
    for phrase in ["FROM python:3.12-slim", "COPY . .", "python -m pip install --no-cache-dir --require-hashes -r requirements.txt"]:
        if phrase not in dockerfile:
            failures.append(f"Dockerfile must include {phrase}")
    for forbidden in ["EMBEDDINGS_URL", "embeddings.pkl", "wget"]:
        if forbidden in dockerfile:
            failures.append(f"Dockerfile must not retain hidden fixture download {forbidden}")

    demo = read("embeddings_demo_step4.py")
    for phrase in [
        'os.environ.get("EMBEDDINGS_FILE_PATH", "embeddings.json")',
        "load_embeddings_and_train_model(embedding_file_path)",
    ]:
        if phrase not in demo:
            failures.append(f"step-4 demo must include {phrase}")
    if demo.find("load_embeddings_and_train_model(embedding_file_path)") > demo.find(
        "openai.Embedding.create"
    ):
        failures.append("step-4 demo must validate its local fixture before API use")
    active_fixture_sources = "\n".join(
        read(path) for path in [
            "utils/generate.py",
            "embeddings_demo_step4.py",
            "create.py",
            "pages/3_🔍_Text_Search.py",
        ]
    )
    for forbidden in ["import pickle", "pickle.load", "pickle.dump", "embeddings.pkl"]:
        if forbidden in active_fixture_sources:
            failures.append(f"active embedding fixture paths must not retain {forbidden}")

    gitignore = read(".gitignore")
    for expected in [".env", ".env.*", "cache/", "url_cache/", "query_cache/", "embedding_cache.pkl", "embedding_cache.json", "embedding_cache.json.tmp", "embeddings.pkl", "embeddings.json"]:
        if expected not in gitignore:
            failures.append(f".gitignore must include {expected}")
    generated_tracks = tracked(["embedding_cache.pkl", "embedding_cache.json", "embedding_cache.json.tmp", "__pycache__", ".pytest_cache"])
    if generated_tracks:
        failures.append("generated local caches must not be tracked: " + ", ".join(generated_tracks))
    if (ROOT / "test_embeddings.pkl").exists():
        failures.append("legacy executable test_embeddings.pkl fixture must not be tracked")
    bytecode_paths = sorted(
        str(path.relative_to(ROOT))
        for pattern in ("__pycache__", "*.pyc")
        for path in ROOT.rglob(pattern)
    )
    if bytecode_paths:
        failures.append("generated Python bytecode must not remain after gates: " + ", ".join(bytecode_paths[:5]))

    tests = read("test_app.py")
    for phrase in [
        "fake_streamlit",
        "test_get_cache_file_does_not_escape_cache_dir",
        "test_get_embeddings_reads_cache_without_api_call",
        "test_get_embeddings_rejects_invalid_cache_without_api_call",
        "test_get_embeddings_rejects_malformed_json_without_api_call",
        "test_get_embeddings_rejects_invalid_api_data_before_cache_write",
        "distances.shape == (1, 2)",
        "test_distance_dimension_mismatch",
        "test_cosine_similarity_dimension_mismatch",
        "test_cosine_similarity_zero_vector",
        "test_record_estimated_cost_adds_first_and_subsequent_values",
        "test_load_embeddings_and_train_model_rejects_empty_fixtures",
        'Path(__file__).with_name("test_embeddings.json")',
        "test_load_embeddings_and_train_model_rejects_malformed_rows",
        "test_load_embeddings_and_train_model_rejects_dimension_mismatch",
        "test_load_embeddings_and_train_model_rejects_metadata_without_text",
        "test_load_embeddings_and_train_model_rejects_non_finite_embedding_values",
        "test_load_embeddings_and_train_model_rejects_missing_fixture",
        "test_load_embeddings_and_train_model_rejects_malformed_json",
        "test_load_embeddings_and_train_model_rejects_invalid_utf8",
        "test_load_embeddings_and_train_model_rejects_non_array_json",
        "test_step4_demo_rejects_missing_fixture_before_api",
        "test_recommend_product_uses_customer_relative_nearest_industry",
        "test_recommend_product_falls_back_to_product_backed_industry",
        "test_recommend_product_filters_malformed_product_names",
        "test_recommend_product_returns_none_for_unavailable_inputs",
        "test_get_top_k_metadata_rejects_invalid_query_embeddings",
        "test_get_top_k_metadata_rejects_dimension_mismatch",
        "numeric finite numbers",
        "test_recursive_text_splitter_preserves_token_overlap",
        "word240",
        "word480",
    ]:
        if phrase not in tests:
            failures.append(f"test_app.py must include {phrase}")

    cache_tests = read("test_embedding_cache.py")
    for phrase in [
        "test_embedding_cache_missing_file_is_empty",
        "test_embedding_cache_round_trip_uses_json",
        "test_embedding_cache_rejects_malformed_or_invalid_data",
        "test_embedding_cache_rejects_invalid_utf8",
    ]:
        if phrase not in cache_tests:
            failures.append(f"test_embedding_cache.py must include {phrase}")

    pipfile = read("Pipfile").replace(" ", "")
    for package in [
        'openai="==0.28.1"',
        'numpy="==2.4.6"',
        'streamlit="==1.58.0"',
        'tiktoken="==0.11.0"',
        'python_version="3.12"',
    ]:
        if package not in pipfile:
            failures.append(f"Pipfile must retain {package}")

    docs = "\n".join(read(path) for path in ["README.md", "SECURITY.md", "VISION.md"])
    for phrase in [
        "make lint",
        "make test",
        "make build",
        "make check",
        "OPENAI_API_KEY",
        "generated caches",
        "legacy OpenAI SDK examples",
        "vector math",
        "small embedding fixtures",
        "empty embedding fixtures",
        "malformed embedding fixtures",
        "metadata text",
        "finite embedding values",
        "numeric embedding values",
        "query embedding validation",
        "vector value validation",
        "Python bytecode",
        "hosted Linux",
        "requirements-test.txt",
        "Python 3.12",
        "requirements.in",
        "make audit",
        "Streamlit health",
        "JSON embedding cache",
        "product-backed",
        "nonempty string product name",
    ]:
        if phrase.lower() not in docs.lower():
            failures.append(f"docs must mention {phrase}")
    guidance_documents = [
        read(path).lower() for path in ["README.md", "SECURITY.md", "VISION.md"]
    ]
    if not all("product-backed" in document for document in guidance_documents):
        failures.append("all guidance must document product-backed recommendations")
    if not all("nonempty string product name" in document for document in guidance_documents):
        failures.append("all guidance must document string product-name validation")
    changes = read("CHANGES.md")
    for phrase in ["make lint", "make test", "make build", "make check"]:
        if phrase not in changes:
            failures.append(f"CHANGES must mention {phrase}")

    plan = read("docs/plans/2026-06-08-openai-102-workshop-baseline.md")
    if "status: completed" not in plan or "make check" not in plan:
        failures.append("completed plan must record status and verification")
    vector_plan = read("docs/plans/2026-06-09-vector-math-validation.md")
    if "status: completed" not in vector_plan or "cosine_similarity" not in vector_plan:
        failures.append("vector validation plan must record status and verification")
    small_fixture_plan = read("docs/plans/2026-06-09-small-embedding-fixtures.md")
    if "status: completed" not in small_fixture_plan or "n_neighbors" not in small_fixture_plan:
        failures.append("small fixture plan must record status and verification")
    empty_fixture_plan = read("docs/plans/2026-06-09-empty-embedding-fixtures.md")
    if "status: completed" not in empty_fixture_plan or "embedding fixture row" not in empty_fixture_plan:
        failures.append("empty fixture plan must record status and verification")
    malformed_fixture_plan = read("docs/plans/2026-06-09-malformed-embedding-fixtures.md")
    if "status: completed" not in malformed_fixture_plan or "same dimensionality" not in malformed_fixture_plan:
        failures.append("malformed fixture plan must record status and verification")
    metadata_text_plan = read("docs/plans/2026-06-09-embedding-metadata-text.md")
    if "status: completed" not in metadata_text_plan or "metadata text" not in metadata_text_plan:
        failures.append("metadata text plan must record status and verification")
    finite_embedding_plan = read("docs/plans/2026-06-09-finite-embedding-values.md")
    if "status: completed" not in finite_embedding_plan or "finite embedding values" not in finite_embedding_plan:
        failures.append("finite embedding values plan must record status and verification")
    numeric_embedding_plan = read("docs/plans/2026-06-10-numeric-embedding-values.md")
    if "status: completed" not in numeric_embedding_plan or "numeric embedding values" not in numeric_embedding_plan:
        failures.append("numeric embedding values plan must record status and verification")
    query_embedding_plan = read("docs/plans/2026-06-10-query-embedding-validation.md")
    if "status: completed" not in query_embedding_plan or "nearest-neighbor lookup" not in query_embedding_plan:
        failures.append("query embedding validation plan must record status and verification")
    vector_value_plan = read("docs/plans/2026-06-12-vector-value-validation.md")
    if "status: completed" not in vector_value_plan or "numeric finite vector-pair validator" not in vector_value_plan:
        failures.append("vector value validation plan must record status and verification")
    make_gate_plan_path = ROOT / "docs/plans/2026-06-09-make-gate-aliases.md"
    make_gate_plan = make_gate_plan_path.read_text(encoding="utf-8") if make_gate_plan_path.exists() else ""
    if "status: completed" not in make_gate_plan or "make lint" not in make_gate_plan or "make build" not in make_gate_plan:
        failures.append("make gate alias plan must record status and verification")
    bytecode_plan = read("docs/plans/2026-06-09-bytecode-free-tests.md")
    if "status: completed" not in bytecode_plan or "Python bytecode" not in bytecode_plan:
        failures.append("bytecode-free test plan must record status and verification")
    ci_plan = read(CI_PLAN)
    if "status: completed" not in ci_plan or "make check" not in ci_plan:
        failures.append("hosted workshop validation plan must record status and verification")
    dependency_plan = read(DEPENDENCY_PLAN)
    if dependency_plan.count("status: completed") != 1:
        failures.append("dependency graph plan must record one completed status")
    for phrase in [
        "## Work Completed",
        "## Verification Results",
        "no known vulnerabilities",
        "55",
        "Streamlit",
        "9459cbe007b2fe9bac9a6dd95e10745a46497d98",
        "27430175764",
        "27430176870",
        "27430174890",
    ]:
        if phrase not in dependency_plan:
            failures.append(f"dependency graph plan must record {phrase}")
    if re.search(r"\b(?:planned|pending|todo|tbd)\b", dependency_plan, re.IGNORECASE):
        failures.append("dependency graph plan must not retain incomplete status markers")
    transitive_security_plan = read(TRANSITIVE_SECURITY_PLAN)
    transitive_security_verification = markdown_section(
        transitive_security_plan, "Verification Completed"
    )
    if (
        transitive_security_plan.count("status: completed") != 1
        or not transitive_security_verification
        or re.search(
            r"(?i)\b(?:pending|todo|tbd|not run|to be recorded)\b",
            transitive_security_verification,
        )
    ):
        failures.append("transitive dependency security plan must record completed verification")
    for phrase in [
        "aiohttp==3.14.1",
        "starlette==1.3.1",
        "requirements.txt",
        "requirements-test.txt",
        "pip-audit",
        "make lock-check",
        "no known vulnerabilities",
    ]:
        if phrase not in transitive_security_verification:
            failures.append(f"transitive dependency security verification must record {phrase}")
    universal_lock_audit_plan = read(UNIVERSAL_LOCK_AUDIT_PLAN)
    universal_lock_audit_verification = markdown_section(
        universal_lock_audit_plan, "Verification Completed"
    )
    if (
        universal_lock_audit_plan.count("status: completed") != 1
        or not universal_lock_audit_verification
        or re.search(
            r"(?i)\b(?:pending|todo|tbd|not run|to be recorded)\b",
            universal_lock_audit_verification,
        )
    ):
        failures.append("universal lock audit plan must record completed verification")
    for phrase in [
        "--no-deps --disable-pip",
        "make audit",
        "make lock-check",
        "no known vulnerabilities",
        "application-smoke",
        "hostile mutations",
    ]:
        if phrase not in universal_lock_audit_verification:
            failures.append(f"universal lock audit verification must record {phrase}")
    hash_verified_lock_plan = read(HASH_VERIFIED_LOCK_PLAN)
    hash_verified_lock_verification = markdown_section(
        hash_verified_lock_plan, "Verification Completed"
    )
    if (
        hash_verified_lock_plan.count("status: completed") != 1
        or not hash_verified_lock_verification
        or re.search(
            r"(?i)\b(?:pending|todo|tbd|not run|to be recorded)\b",
            hash_verified_lock_verification,
        )
    ):
        failures.append("hash-verified universal lock plan must record completed verification")
    for phrase in [
        "All four Make gates passed",
        "Six isolated hostile mutations were rejected",
        "external directory",
        "27658071031",
        "27658078238",
        "application-smoke",
    ]:
        if phrase not in hash_verified_lock_verification:
            failures.append(f"hash-verified universal lock verification must record {phrase}")
    for path in ["README.md", "SECURITY.md", "VISION.md", "CHANGES.md"]:
        if "artifact hashes" not in read(path).lower():
            failures.append(f"{path} must document universal lock artifact hashes")
    prepared_ci_plan = read("docs/plans/2026-06-10-ci-baseline.md")
    if "status: completed" not in prepared_ci_plan or "make check" not in prepared_ci_plan:
        failures.append("CI baseline plan must record status and verification")
    embedding_cache_plan = read(EMBEDDING_CACHE_PLAN)
    for phrase in [
        "status: completed",
        "make check",
        "test_embedding_cache.py",
        "six hostile mutations",
    ]:
        if phrase not in embedding_cache_plan:
            failures.append(f"JSON embedding cache plan must record {phrase}")
    embedding_payload_plan = read(EMBEDDING_PAYLOAD_PLAN)
    for phrase in [
        "status: completed",
        "make check",
        "invalid cache",
        "invalid API data",
        "hostile mutations",
    ]:
        if phrase not in embedding_payload_plan:
            failures.append(f"embedding payload validation plan must record {phrase}")
    customer_recommendation_plan = read(CUSTOMER_RECOMMENDATION_PLAN)
    customer_recommendation_verification = markdown_section(
        customer_recommendation_plan, "Verification Completed"
    )
    if (
        "status: completed" not in customer_recommendation_plan
        or not customer_recommendation_verification
        or re.search(
            r"(?i)\b(?:pending|todo|tbd|not run|to be recorded)\b",
            customer_recommendation_verification,
        )
    ):
        failures.append("customer recommendation plan must record completed verification")
    for phrase in [
        "four focused recommendation tests",
        "make check",
        "external working directory",
        "Six isolated hostile mutations",
        "git diff --check",
    ]:
        if phrase not in customer_recommendation_verification:
            failures.append(f"customer recommendation verification must record {phrase}")
    product_backed_plan = read(PRODUCT_BACKED_RECOMMENDATION_PLAN)
    product_backed_status = re.findall(
        r"(?mi)^status:\s*(.+?)\s*$", product_backed_plan
    )
    product_backed_work = markdown_section(product_backed_plan, "Work Completed")
    product_backed_verification = markdown_section(
        product_backed_plan, "Verification Completed"
    )
    if (product_backed_status != ["completed"] or not product_backed_work or
            not product_backed_verification or re.search(
                r"(?i)\b(?:pending|todo|tbd|not run|to be recorded)\b",
                product_backed_verification,
            )):
        failures.append(
            "product-backed recommendation plan must record completed verification"
        )
    for phrase in [
        "Six focused recommendation cases",
        "80-test no-network suite",
        "make lint",
        "make test",
        "make build",
        "make check",
        "external working directory",
        "Runtime imports",
        "Streamlit health smoke",
        "no known vulnerabilities",
        "Six isolated hostile mutations",
        "git diff --check",
    ]:
        if phrase not in product_backed_verification:
            failures.append(f"product-backed recommendation verification must record {phrase}")
    product_name_plan = read(PRODUCT_NAME_VALIDATION_PLAN)
    product_name_status = re.findall(r"(?mi)^status:\s*(.+?)\s*$", product_name_plan)
    product_name_work = markdown_section(product_name_plan, "Work Completed")
    product_name_verification = markdown_section(product_name_plan, "Verification Completed")
    if (product_name_status != ["completed"] or not product_name_work or
            not product_name_verification or re.search(
                r"(?i)\b(?:pending|todo|tbd|not run|to be recorded)\b",
                product_name_verification,
            )):
        failures.append("product name validation plan must record completed verification")
    for phrase in [
        "Eight focused recommendation cases",
        "82-test no-network suite",
        "make lint",
        "make test",
        "make build",
        "make check",
        "external working directory",
        "Runtime imports",
        "Streamlit health smoke",
        "no known vulnerabilities",
        "Six isolated hostile mutations",
        "git diff --check",
    ]:
        if phrase not in product_name_verification:
            failures.append(f"product name validation verification must record {phrase}")

    malformed_customer_plan = read(MALFORMED_CUSTOMER_PLAN)
    malformed_customer_status = re.findall(
        r"(?mi)^status:\s*(.+?)\s*$", malformed_customer_plan
    )
    malformed_customer_work = markdown_section(
        malformed_customer_plan, "Work Completed"
    )
    malformed_customer_verification = markdown_section(
        malformed_customer_plan, "Verification Completed"
    )
    if (malformed_customer_status != ["completed"] or not malformed_customer_work or
            not malformed_customer_verification or re.search(
                r"(?i)\b(?:pending|todo|tbd|not run|to be recorded)\b",
                malformed_customer_verification,
            )):
        failures.append("malformed customer guard plan must record completed verification")
    for phrase in [
        "focused recommendation cases",
        "no-network suite",
        "make lint",
        "make test",
        "make build",
        "make check",
        "external working directory",
        "isolated hostile mutations",
        "git diff --check",
    ]:
        if phrase not in malformed_customer_verification:
            failures.append(f"malformed customer verification must record {phrase}")

    customer_industry_plan = read(CUSTOMER_INDUSTRY_NAME_PLAN)
    customer_industry_status = re.findall(
        r"(?mi)^status:\s*(.+?)\s*$", customer_industry_plan
    )
    customer_industry_verification = markdown_section(
        customer_industry_plan, "Verification Completed"
    )
    if (customer_industry_status != ["completed"] or
            "focused recommendation cases" not in customer_industry_verification or
            "complete no-network suite" not in customer_industry_verification or
            "All four Make gates passed" not in customer_industry_verification or
            "external directory" not in customer_industry_verification or
            "Six isolated hostile mutations were rejected" not in customer_industry_verification or
            re.search(r"(?i)\b(?:pending|todo|tbd|not run|to be recorded)\b",
                      customer_industry_verification)):
        failures.append("customer industry name plan must record completed verification")

    recommendation_container_plan = read(RECOMMENDATION_CONTAINER_PLAN)
    recommendation_container_status = re.findall(
        r"(?mi)^status:\s*(.+?)\s*$", recommendation_container_plan
    )
    recommendation_container_verification = markdown_section(
        recommendation_container_plan, "Verification Completed"
    )
    if (recommendation_container_status != ["completed"] or
            "focused recommendation cases" not in recommendation_container_verification or
            "complete no-network suite" not in recommendation_container_verification or
            "All four Make gates passed" not in recommendation_container_verification or
            "external directory" not in recommendation_container_verification or
            "Seven isolated hostile mutations were rejected" not in recommendation_container_verification or
            re.search(r"(?i)\b(?:pending|todo|tbd|not run|to be recorded)\b",
                      recommendation_container_verification)):
        failures.append("recommendation container validation plan must record completed verification")

    recommendation_embedding_plan = read(RECOMMENDATION_EMBEDDING_PLAN)
    recommendation_embedding_status = re.findall(
        r"(?mi)^status:\s*(.+?)\s*$", recommendation_embedding_plan
    )
    recommendation_embedding_verification = markdown_section(
        recommendation_embedding_plan, "Verification Completed"
    )
    if (recommendation_embedding_status != ["completed"] or
            "focused recommendation cases" not in recommendation_embedding_verification or
            "complete no-network suite" not in recommendation_embedding_verification or
            "All four Make gates passed" not in recommendation_embedding_verification or
            "external directory" not in recommendation_embedding_verification or
            "Six isolated hostile mutations were rejected" not in recommendation_embedding_verification or
            re.search(r"(?i)\b(?:pending|todo|tbd|not run|to be recorded)\b",
                      recommendation_embedding_verification)):
        failures.append("recommendation embedding validation plan must record completed verification")

    for path in ["README.md", "SECURITY.md", "VISION.md", "CHANGES.md"]:
        if "invalid recommendation embedding pairs" not in read(path).lower():
            failures.append(f"{path} must document invalid recommendation embedding pairs")
        if "customer-industry recommendation tie break" not in read(path).lower():
            failures.append(f"{path} must document customer-industry recommendation tie break")

    recommendation_tie_break_plan = read(RECOMMENDATION_TIE_BREAK_PLAN)
    recommendation_tie_break_status = re.findall(
        r"(?mi)^status:\s*(.+?)\s*$", recommendation_tie_break_plan
    )
    recommendation_tie_break_verification = markdown_section(
        recommendation_tie_break_plan, "Verification Completed"
    )
    if (recommendation_tie_break_status != ["completed"] or
            "Three focused recommendation cases passed" not in recommendation_tie_break_verification or
            "complete no-network suite passed with 99 tests" not in recommendation_tie_break_verification or
            "All four Make gates passed" not in recommendation_tie_break_verification or
            "external directory" not in recommendation_tie_break_verification or
            "Eight isolated hostile mutations were rejected" not in recommendation_tie_break_verification or
            re.search(r"(?i)\b(?:pending|todo|tbd|not run|to be recorded)\b",
                      recommendation_tie_break_verification)):
        failures.append("recommendation tie-breaking plan must record completed verification")

    compatibility = " ".join(read("docs/openai-api-compatibility.md").split())
    for phrase in [
        "Review date: 2026-06-13",
        "Compatibility status: historical workshop examples; not current integration guidance",
        "openai.Completion.create",
        "openai.ChatCompletion.create",
        "openai.Embedding.create",
        "openai api fine_tunes.create",
        "text-embedding-ada-002",
        "text-davinci-003",
        "gpt-3.5-turbo",
        "https://developers.openai.com/api/docs/models",
        "https://developers.openai.com/api/docs/models/all",
        "https://developers.openai.com/api/docs/models/text-embedding-3-small",
        "https://developers.openai.com/api/docs/models/davinci-002",
        "They do not select a replacement model",
        "preserve the default no-network `make check` gate",
    ]:
        if phrase not in compatibility:
            failures.append(f"OpenAI API compatibility note must include {phrase}")

    warning = (
        "Historical OpenAI API example: this workshop preserves openai==0.28.1 "
        "and legacy model identifiers. Review docs/openai-api-compatibility.md "
        "before building a new integration."
    )
    warning_pages = [
        "pages/1_🧐_Getting_Started.py",
        "pages/2_⚡️_API.py",
        "pages/2_📝_Embeddings.py",
        "pages/3_🔍_Text_Search.py",
        "pages/8_🦾_FineTuning.py",
    ]
    for page in warning_pages:
        if read(page).count(warning) != 1:
            failures.append(f"{page} must show one OpenAI API compatibility warning")
    if "[`docs/openai-api-compatibility.md`](docs/openai-api-compatibility.md)" not in read("README.md"):
        failures.append("README must link the OpenAI API compatibility inventory")

    finetuning = read("pages/8_🦾_FineTuning.py")
    retry_loop = finetuning.split("for j in range(10):", 1)[1].split(
        "new_row =", 1
    )[0]
    for contract in [
        "except openai.error.RateLimitError:",
        "if j == 9:\n                        raise",
        "sleep_time = (2 ** j) + random.random()",
    ]:
        if contract not in retry_loop:
            failures.append(f"fine-tuning retry example must retain {contract}")
    if "except:" in retry_loop:
        failures.append("fine-tuning retry example must not catch every exception")
    if "else:\n                    raise" in retry_loop:
        failures.append("fine-tuning retry example must not retain unreachable rethrow logic")

    if "test_finetuning_example_retries_only_rate_limits" not in read("test_app.py"):
        failures.append("fine-tuning rate-limit retry regression must remain registered")

    for path in ["AGENTS.md", "README.md", "SECURITY.md", "VISION.md", "CHANGES.md"]:
        if "fine-tuning retry example" not in read(path).lower():
            failures.append(f"{path} must document the fine-tuning retry example")

    finetuning_retry_plan = read(FINETUNING_RETRY_PLAN)
    finetuning_retry_status = re.findall(
        r"(?mi)^status:\s*(.+?)\s*$", finetuning_retry_plan
    )
    finetuning_retry_verification = markdown_section(
        finetuning_retry_plan, "Verification Completed"
    )
    if (finetuning_retry_status != ["completed"] or
            "focused fine-tuning retry regression passed" not in finetuning_retry_verification or
            "complete no-network suite passed with 101 tests" not in finetuning_retry_verification or
            "All four Make gates passed" not in finetuning_retry_verification or
            "external directory" not in finetuning_retry_verification or
            "Six isolated hostile mutations were rejected" not in finetuning_retry_verification or
            re.search(r"(?i)\b(?:pending|todo|tbd|not run|to be recorded)\b",
                      finetuning_retry_verification)):
        failures.append("fine-tuning retry plan must record completed verification")

    if "test_starlette_security_floor_is_resolver_input" not in read("test_app.py"):
        failures.append("Starlette resolver-floor regression must remain registered")

    crawler_tests = read("test_crawler.py")
    for test_name in [
        "test_crawler_rejects_non_public_dns_answers",
        "test_crawler_rejects_every_iana_non_global_special_range",
        "test_crawler_retains_iana_globally_reachable_exceptions",
        "test_crawler_dns_resolution_obeys_total_deadline",
        "test_crawler_pins_request_to_validated_public_address",
        "test_crawler_pinned_https_preserves_sni_and_certificate_hostname",
        "test_crawler_revalidates_redirect_destination",
        "test_crawler_limits_redirect_chain",
        "test_crawler_streams_gzip_without_expanding_zip_bomb",
        "test_crawler_total_deadline_stops_slow_chunked_response",
        "test_crawler_total_deadline_bounds_status_and_header_parsing",
        "test_crawler_limits_url_count_and_aggregate_decoded_bytes",
    ]:
        if test_name not in crawler_tests:
            failures.append(f"crawler SSRF regression must remain registered: {test_name}")

    for path in ["AGENTS.md", "README.md", "SECURITY.md", "VISION.md", "CHANGES.md"]:
        if "starlette resolver floor" not in read(path).lower():
            failures.append(f"{path} must document the Starlette resolver floor")

    starlette_lock_plan = read(STARLETTE_LOCK_PLAN)
    starlette_lock_status = re.findall(
        r"(?mi)^status:\s*(.+?)\s*$", starlette_lock_plan
    )
    starlette_lock_verification = markdown_section(
        starlette_lock_plan, "Verification Completed"
    )
    if (starlette_lock_status != ["completed"] or
            "make lock-check passed twice" not in starlette_lock_verification or
            "complete no-network suite passed with 101 tests" not in starlette_lock_verification or
            "Both exact locks audited with no known vulnerabilities" not in starlette_lock_verification or
            "runtime imports and credential-free Streamlit smoke passed" not in starlette_lock_verification or
            "Six isolated hostile mutations were rejected" not in starlette_lock_verification or
            re.search(r"(?i)\b(?:pending|todo|tbd|not run|to be recorded)\b",
                      starlette_lock_verification)):
        failures.append("Starlette lock-floor plan must record completed verification")

    for path in ["AGENTS.md", "README.md", "SECURITY.md", "VISION.md", "CHANGES.md"]:
        if "safe json embedding fixtures" not in read(path).lower():
            failures.append(f"{path} must document safe JSON embedding fixtures")

    safe_json_fixture_plan = read(SAFE_JSON_FIXTURE_PLAN)
    safe_json_fixture_status = re.findall(
        r"(?mi)^status:\s*(.+?)\s*$", safe_json_fixture_plan
    )
    safe_json_fixture_verification = markdown_section(
        safe_json_fixture_plan, "Verification Completed"
    )
    for phrase in [
        "14 focused JSON fixture cases passed",
        "complete no-network suite passed with 107 tests",
        "repository and external-directory `make check` passed",
        "Six isolated hostile mutations were rejected",
        "git diff --check",
    ]:
        if phrase not in safe_json_fixture_verification:
            failures.append(
                f"safe JSON fixture verification must record {phrase}"
            )
    if (safe_json_fixture_status != ["completed"] or
            re.search(r"(?i)\b(?:pending|todo|tbd|not run|to be recorded)\b",
                      safe_json_fixture_verification)):
        failures.append("safe JSON fixture plan must record completed verification")

    for path in ["AGENTS.md", "README.md", "SECURITY.md", "VISION.md", "CHANGES.md"]:
        if "globally routable" not in read(path).lower():
            failures.append(f"{path} must document the crawler public-address boundary")

    crawler_ssrf_plan = read(CRAWLER_SSRF_PLAN)
    crawler_ssrf_status = re.findall(
        r"(?mi)^status:\s*(.+?)\s*$", crawler_ssrf_plan
    )
    crawler_ssrf_verification = markdown_section(
        crawler_ssrf_plan, "Verification Completed"
    )
    for phrase in [
        "108 focused crawler cases cover",
        "complete no-network suite passed with 219 tests",
        "exact Python 3.12.0",
        "certificate-verified pinned HTTPS request",
        "Alert #8",
        "make build",
        "make verify",
        "make lock-check",
        "make audit",
        "make runtime-check",
        "make smoke",
        "git diff --check",
    ]:
        if phrase not in crawler_ssrf_verification:
            failures.append(f"crawler SSRF verification must record {phrase}")
    if (crawler_ssrf_status != ["completed"] or
            re.search(r"(?i)\b(?:pending|todo|tbd|not run|to be recorded)\b",
                      crawler_ssrf_verification)):
        failures.append("crawler SSRF plan must record completed verification")

    for path, phrases in {
        "requirements.in": ["openai==0.28.1"],
        "requirements-test.in": ["openai==0.28.1"],
        "Pipfile": ['openai = "==0.28.1"'],
        "utils/generate.py": [
            "openai.Embedding.create",
            "openai.ChatCompletion.create",
            "openai.Completion.create",
            'engine="text-embedding-ada-002"',
            'model="gpt-3.5-turbo"',
            'model="text-davinci-003"',
        ],
    }.items():
        text = read(path)
        for phrase in phrases:
            if phrase not in text:
                failures.append(f"{path} must retain inventoried legacy API surface {phrase}")

    compatibility_plan = read(API_COMPATIBILITY_PLAN)
    compatibility_status = re.findall(r"(?mi)^status:\s*(.+?)\s*$", compatibility_plan)
    compatibility_work = markdown_section(compatibility_plan, "Work Completed")
    compatibility_verification = markdown_section(compatibility_plan, "Verification Completed")
    if compatibility_status != ["completed"] or not compatibility_work:
        failures.append("OpenAI API compatibility plan must record completed status and work")
    if not compatibility_verification or re.search(
        r"(?i)\b(?:pending|todo|tbd|not run)\b", compatibility_verification
    ):
        failures.append("OpenAI API compatibility plan must record completed verification")
    for evidence in [
        "python3 -m py_compile scripts/check-workshop-baseline.py",
        "make lint",
        "make test",
        "make build",
        "make check",
        "external working directory",
        "hostile mutations rejected",
        "legacy API behavior paths had no diff",
        "git diff --check",
    ]:
        if evidence not in compatibility_verification:
            failures.append(f"OpenAI API compatibility verification must record {evidence}")

    try:
        ET.parse(ROOT / "docs/readme-overview.svg")
    except ET.ParseError as error:
        failures.append(f"docs/readme-overview.svg must parse as XML: {error}")

    try:
        json.loads(read("query_cache/what is twilio?.json"))
    except Exception as error:
        failures.append(f"query_cache fixture must stay readable JSON: {error}")

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    print("openai-102-workshop baseline checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
