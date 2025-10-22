#!/bin/bash

# Setup script for PoCCINQualityEval
# This script sets up the Python environment and installs all dependencies using uv

echo "========================================"
echo "PoCCINQualityEval Setup"
echo "========================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Check if uv is installed (fast Python package manager)
if ! command -v uv &> /dev/null
then
    echo "uv not found. Attempting to install uv..."
    if command -v brew &> /dev/null; then
        echo "Installing uv via Homebrew..."
        brew install uv
        if [ $? -ne 0 ]; then
            echo "❌ Failed to install uv via Homebrew."
            exit 1
        fi
    else
        echo "Installing uv via official installer script..."
        curl -fsSL https://astral.sh/uv/install.sh | sh
        if [ $? -ne 0 ]; then
            echo "❌ Failed to install uv via installer script."
            exit 1
        fi
        # Ensure ~/.local/bin is in PATH for current session (uv default install path)
        if [ -d "$HOME/.local/bin" ]; then
            export PATH="$HOME/.local/bin:$PATH"
        fi
    fi
fi

echo "✓ uv found: $(uv --version)"
echo ""

# Create virtual environment with uv
echo "Creating virtual environment with uv..."
uv venv venv

if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✓ Virtual environment activated"
echo ""

# Install dependencies with uv
echo ""
echo "Installing dependencies with uv (this may take a few minutes)..."
uv pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ Setup complete!"
echo "========================================"
echo ""
echo "To use the analyzer:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the analyzer:"
echo "     python id_card_analyzer.py <image_path>"
echo ""
echo "Example:"
echo "  python id_card_analyzer.py sample_id_card.jpg"
echo ""
