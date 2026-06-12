#!/usr/bin/env python3
"""Import the reviewed direct application dependency surface."""

from importlib import import_module, metadata
import sys


DIRECT_IMPORTS = {
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
}


def main():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("runtime dependency checks require Python 3.12")

    imported = []
    for distribution, module in DIRECT_IMPORTS.items():
        import_module(module)
        imported.append(f"{distribution}=={metadata.version(distribution)}")

    print("runtime imports passed: " + ", ".join(imported))


if __name__ == "__main__":
    main()
