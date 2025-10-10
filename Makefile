.PHONY: help tests lint types complexity deadcode deadcode-aggressive all clean install

help:
	@echo "Available targets:"
	@echo "  make tests              - Run pytest test suite"
	@echo "  make lint               - Run ruff linter and formatter"
	@echo "  make types              - Run pyright type checker"
	@echo "  make complexity         - Run radon code complexity analysis"
	@echo "  make deadcode           - Find unused code with vulture (80% confidence)"
	@echo "  make deadcode-aggressive - Find unused code (60% confidence, more false positives)"
	@echo "  make all                - Run all checks (tests, lint, types)"
	@echo "  make clean              - Clean build artifacts and cache"
	@echo "  make install            - Install dependencies with uv"

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
	@RADON_CFG="" uv run radon cc src/ -a -s
	@echo ""
	@echo "=== Maintainability Index (A-C scale) ==="
	@RADON_CFG="" uv run radon mi src/ -s

deadcode:
	@echo "Scanning for unused code with vulture..."
	@echo "=== Dead Code Detection ==="
	@echo ""
	@uv run vulture src/ .vulture_whitelist.py --min-confidence 80 --sort-by-size || true
	@echo ""
	@echo "Note: Review results carefully - some code may be used dynamically or by external tools."
	@echo "Tip: Add false positives to .vulture_whitelist.py"
	@echo "     Adjust sensitivity with: make deadcode CONFIDENCE=60 (default: 80, range: 0-100)"

deadcode-aggressive:
	@echo "Scanning for unused code (aggressive mode)..."
	@echo "=== Dead Code Detection (60% confidence) ==="
	@echo ""
	@uv run vulture src/ .vulture_whitelist.py --min-confidence 60 --sort-by-size || true
	@echo ""
	@echo "Note: Lower confidence = more false positives, but catches more potential dead code."

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
