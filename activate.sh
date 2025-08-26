#!/bin/bash
# Activation script for AI Filesystem Finder development environment

echo "Activating AI Filesystem Finder virtual environment..."
source venv/bin/activate

echo "Virtual environment activated!"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo ""
echo "Available development commands:"
echo "  pytest          - Run tests"
echo "  black .         - Format code"
echo "  isort .         - Sort imports"
echo "  flake8 .        - Lint code"
echo "  mypy src/      - Type checking"
echo ""
echo "To deactivate, run: deactivate"