#!/bin/bash

# Quality Check Script
# This script runs code quality checks without modifying files

set -e

echo "================================"
echo "Running Code Quality Checks"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track if any check fails
FAILED=0

echo ""
echo "${YELLOW}[1/3] Checking code formatting with Black...${NC}"
if uv run black --check backend/ main.py; then
    echo "${GREEN}✓ Black formatting check passed${NC}"
else
    echo "${RED}✗ Black formatting check failed${NC}"
    echo "  Run './format.sh' to auto-fix formatting issues"
    FAILED=1
fi

echo ""
echo "${YELLOW}[2/3] Checking import sorting with isort...${NC}"
if uv run isort --check-only backend/ main.py; then
    echo "${GREEN}✓ isort import check passed${NC}"
else
    echo "${RED}✗ isort import check failed${NC}"
    echo "  Run './format.sh' to auto-fix import sorting"
    FAILED=1
fi

echo ""
echo "${YELLOW}[3/3] Running flake8 linter...${NC}"
if uv run flake8 backend/ main.py; then
    echo "${GREEN}✓ flake8 linting passed${NC}"
else
    echo "${RED}✗ flake8 linting failed${NC}"
    echo "  Review the errors above and fix manually"
    FAILED=1
fi

echo ""
echo "================================"
if [ $FAILED -eq 0 ]; then
    echo "${GREEN}All quality checks passed! ✨${NC}"
    echo "================================"
    exit 0
else
    echo "${RED}Some quality checks failed${NC}"
    echo "================================"
    exit 1
fi
