.PHONY: help tests lint types complexity all clean install docs-serve docs-build

help:
	@echo "Available targets:"
	@echo "  make tests      - Run pytest test suite"
	@echo "  make lint       - Run ruff linter and formatter"
	@echo "  make types      - Run pyright type checker"
	@echo "  make complexity - Run radon code complexity analysis"
	@echo "  make all        - Run all checks (tests, lint, types)"
	@echo "  make clean      - Clean build artifacts and cache"
	@echo "  make install    - Install dependencies with uv"
	@echo "  make docs-serve - Start MkDocs dev server with live reload"
	@echo "  make docs-build - Build MkDocs static site for production"

tests:
	@echo "Running pytest..."
	uv run pytest

lint:
	@echo "Running ruff linter..."
	uv run ruff check src/ tests/
	@echo "Checking ruff formatting..."
	uv run ruff format src/ tests/

types:
	@echo "Running pyright type checker..."
	uv run pyright src/

complexity:
	@echo "Running radon cyclomatic complexity analysis..."
	@echo "=== Cyclomatic Complexity (A-F scale) ==="
	uv run radon cc src/ -a -s
	@echo ""
	@echo "=== Maintainability Index (A-C scale) ==="
	uv run radon mi src/ -s

all: lint types tests
	@echo "All checks passed!"

clean:
	@echo "Cleaning build artifacts and cache..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true

install:
	@echo "Installing dependencies..."
	uv sync --dev

docs-serve:
	@echo "Starting MkDocs development server..."
	uv run mkdocs serve

docs-build:
	@echo "Building MkDocs documentation..."
	uv run mkdocs build
