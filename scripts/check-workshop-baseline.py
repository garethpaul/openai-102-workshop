#!/usr/bin/env python3
"""Static baseline checks for the OpenAI 102 workshop."""

from pathlib import Path
import ast
import json
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ".gitignore",
    "CHANGES.md",
    "Dockerfile",
    "Makefile",
    "README.md",
    "SECURITY.md",
    "VISION.md",
    "components/common.py",
    "docs/plans/2026-06-08-openai-102-workshop-baseline.md",
    "docs/readme-overview.svg",
    "requirements.txt",
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
        "hashlib.sha256",
        "os.path.commonpath",
        "get_cache_file(cache_folder, query)",
    ]:
        if phrase not in generate:
            failures.append(f"utils/generate.py must include {phrase}")
    if 'os.path.join(cache_folder, f"{query}.json")' in generate:
        failures.append("text embedding cache names must not use raw queries")

    crawler = read("utils/crawler.py")
    if "timeout=15" not in crawler or "raise_for_status()" not in crawler:
        failures.append("crawler requests must use timeout and raise_for_status")

    makefile = read("Makefile")
    for phrase in ["python3 -m pytest -q test_app.py", "static-check", "check: static-check test"]:
        if phrase not in makefile:
            failures.append(f"Makefile must include {phrase}")

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

    tests = read("test_app.py")
    for phrase in [
        "fake_streamlit",
        "test_get_cache_file_does_not_escape_cache_dir",
        "test_get_embeddings_reads_cache_without_api_call",
        "test_distance_dimension_mismatch",
    ]:
        if phrase not in tests:
            failures.append(f"test_app.py must include {phrase}")

    requirements = read("requirements.txt")
    for package in ["openai", "streamlit==", "python-dotenv==", "tiktoken=="]:
        if package not in requirements:
            failures.append(f"requirements.txt must include {package}")

    docs = "\n".join(read(path) for path in ["README.md", "SECURITY.md", "VISION.md"])
    for phrase in ["make check", "OPENAI_API_KEY", "generated caches", "legacy OpenAI SDK examples"]:
        if phrase.lower() not in docs.lower():
            failures.append(f"docs must mention {phrase}")

    plan = read("docs/plans/2026-06-08-openai-102-workshop-baseline.md")
    if "status: completed" not in plan or "make check" not in plan:
        failures.append("completed plan must record status and verification")

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
