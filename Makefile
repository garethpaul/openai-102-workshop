.PHONY: build run test static-check check all

# Build the app (install dependencies)
build:
	python3 -m pip install -r requirements.txt

# Run the app locally
run:
	streamlit run 👋_Hello.py

# Test the app
test:
	python3 -m pytest -q test_app.py

static-check:
	python3 scripts/check-workshop-baseline.py

check: static-check test

# Build test and run the app
all: build test run
