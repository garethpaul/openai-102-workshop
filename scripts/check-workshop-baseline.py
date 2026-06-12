#!/usr/bin/env python3
"""Static baseline checks for the OpenAI 102 workshop."""

from pathlib import Path
import ast
import json
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
CI_PLAN = "docs/plans/2026-06-10-hosted-workshop-validation.md"
REQUIRED = [
    ".github/workflows/check.yml",
    ".gitignore",
    "CHANGES.md",
    "Dockerfile",
    "Makefile",
    "README.md",
    "SECURITY.md",
    "VISION.md",
    "components/common.py",
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
    CI_PLAN,
    "docs/plans/2026-06-09-make-gate-aliases.md",
    "docs/plans/2026-06-09-bytecode-free-tests.md",
    "docs/readme-overview.svg",
    "requirements.txt",
    "requirements-test.txt",
    "scripts/check-workshop-baseline.py",
    "test_app.py",
    "utils/crawler.py",
    "utils/generate.py",
]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8", errors="replace")


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
        "test_app.py",
        "utils/crawler.py",
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

    generate = read("utils/generate.py")
    for phrase in [
        "def get_cache_file",
        "def validate_saved_embeddings",
        "def validate_query_embedding",
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
    ]:
        if phrase not in generate:
            failures.append(f"utils/generate.py must include {phrase}")
    if 'os.path.join(cache_folder, f"{query}.json")' in generate:
        failures.append("text embedding cache names must not use raw queries")
    if 'st_state[\'cost\'] = f"${0:.10f}"' in generate:
        failures.append("cost tracking must record the first request cost")

    crawler = read("utils/crawler.py")
    if "timeout=15" not in crawler or "raise_for_status()" not in crawler:
        failures.append("crawler requests must use timeout and raise_for_status")

    makefile = read("Makefile")
    for phrase in [
        ".PHONY: all build check lint run static-check test verify",
        "PYTHON ?= python3",
        "PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -c \"import pathlib; [compile(pathlib.Path(path).read_text(), path, 'exec')",
        "PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -q test_app.py",
        "PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/check-workshop-baseline.py",
        "lint: static-check",
        "verify: lint test",
        "check: verify",
    ]:
        if phrase not in makefile:
            failures.append(f"Makefile must include {phrase}")

    test_requirements = read("requirements-test.txt")
    for requirement in [
        "numpy==1.26.4",
        "openai==0.28.1",
        "pytest==7.4.4",
        "requests==2.31.0",
        "scikit-learn==1.3.2",
    ]:
        if requirement not in test_requirements:
            failures.append(f"test requirements must pin {requirement}")

    workflow = read(".github/workflows/check.yml")
    for phrase in [
        "permissions:\n  contents: read",
        "cancel-in-progress: true",
        "runs-on: ubuntu-24.04",
        "timeout-minutes: 15",
        "actions/checkout@df4cb1c069e1874edd31b4311f1884172cec0e10",
        "actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405",
        'python-version: "3.10"',
        "python -m pip install -r requirements-test.txt",
        "python -m pip check",
        "make check",
    ]:
        if phrase not in workflow:
            failures.append(f"Check workflow must keep {phrase}")

    dockerfile = read("Dockerfile")
    for phrase in ["ARG EMBEDDINGS_URL", "--no-install-recommends", "wget --https-only"]:
        if phrase not in dockerfile:
            failures.append(f"Dockerfile must include {phrase}")

    gitignore = read(".gitignore")
    for expected in [".env", ".env.*", "cache/", "url_cache/", "query_cache/", "embedding_cache.pkl", "embeddings.pkl"]:
        if expected not in gitignore:
            failures.append(f".gitignore must include {expected}")
    generated_tracks = tracked(["embedding_cache.pkl", "__pycache__", ".pytest_cache"])
    if generated_tracks:
        failures.append("generated local caches must not be tracked: " + ", ".join(generated_tracks))
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
        "distances.shape == (1, 2)",
        "test_distance_dimension_mismatch",
        "test_cosine_similarity_dimension_mismatch",
        "test_cosine_similarity_zero_vector",
        "test_record_estimated_cost_adds_first_and_subsequent_values",
        "test_load_embeddings_and_train_model_rejects_empty_fixtures",
        "test_load_embeddings_and_train_model_rejects_malformed_rows",
        "test_load_embeddings_and_train_model_rejects_dimension_mismatch",
        "test_load_embeddings_and_train_model_rejects_metadata_without_text",
        "test_load_embeddings_and_train_model_rejects_non_finite_embedding_values",
        "test_get_top_k_metadata_rejects_invalid_query_embeddings",
        "test_get_top_k_metadata_rejects_dimension_mismatch",
        "numeric finite numbers",
    ]:
        if phrase not in tests:
            failures.append(f"test_app.py must include {phrase}")

    requirements = read("requirements.txt").replace(" ", "")
    if "openai<1.0" not in requirements:
        failures.append("requirements.txt must pin legacy examples to openai<1.0")
    if "numpy<2" not in requirements:
        failures.append("requirements.txt must pin legacy examples to numpy<2")
    if "pytest" not in requirements:
        failures.append("requirements.txt must include pytest for make check")
    pipfile = read("Pipfile").replace(" ", "")
    if 'openai="<1.0"' not in pipfile:
        failures.append('Pipfile must pin legacy examples with openai = "<1.0"')
    if 'numpy="<2"' not in pipfile:
        failures.append('Pipfile must pin legacy examples with numpy = "<2"')
    if "pytest" not in pipfile:
        failures.append("Pipfile must include pytest in dev-packages")
    for package in ["streamlit==", "python-dotenv==", "tiktoken=="]:
        if package not in requirements:
            failures.append(f"requirements.txt must include {package}")

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
    ]:
        if phrase.lower() not in docs.lower():
            failures.append(f"docs must mention {phrase}")
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
