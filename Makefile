.PHONY: build run

# Build the app (install dependencies)
build:
	pip install -r requirements.txt

# Run the app locally
run:
	streamlit run 👋_Hello.py

# Test the app
test:
	pytest

# Build test and run the app
all: build test run
