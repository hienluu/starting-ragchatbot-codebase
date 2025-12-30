#!/bin/bash

# Format Script
# This script automatically formats code using Black and isort

set -e

echo "================================"
echo "Auto-formatting Code"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "${YELLOW}[1/2] Formatting code with Black...${NC}"
uv run black backend/ main.py
echo "${GREEN}✓ Black formatting completed${NC}"

echo ""
echo "${YELLOW}[2/2] Sorting imports with isort...${NC}"
uv run isort backend/ main.py
echo "${GREEN}✓ Import sorting completed${NC}"

echo ""
echo "================================"
echo "${GREEN}Code formatting complete! ✨${NC}"
echo "================================"
echo ""
echo "Run './quality-check.sh' to verify all quality checks pass"
