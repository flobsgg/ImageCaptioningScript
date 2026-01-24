#!/bin/bash

# Setup script for Image Captioning Tool
# Automatically installs Python dependencies and creates virtual environment

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================"
echo "Image Captioning Tool - Setup"
echo "================================"
echo ""

# Check for Python 3
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        # Check if python is Python 3
        if python --version 2>&1 | grep -q "Python 3"; then
            PYTHON_CMD="python"
        else
            return 1
        fi
    else
        return 1
    fi

    echo "Found Python: $($PYTHON_CMD --version)"
    return 0
}

# Install Python if not found (macOS only)
install_python_mac() {
    echo ""
    echo "Python 3 is required but not installed."
    echo ""

    if command -v brew &> /dev/null; then
        echo "Homebrew detected. Installing Python..."
        brew install python3
        PYTHON_CMD="python3"
    else
        echo "To install Python, you can either:"
        echo "  1. Install Homebrew first: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo "     Then run this script again."
        echo ""
        echo "  2. Download Python directly from: https://www.python.org/downloads/"
        echo ""
        exit 1
    fi
}

# Main setup
if ! check_python; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        install_python_mac
        check_python || { echo "Python installation failed"; exit 1; }
    else
        echo "Python 3 is required. Please install it:"
        echo "  Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
        echo "  Fedora: sudo dnf install python3"
        echo "  Or download from: https://www.python.org/downloads/"
        exit 1
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate and install dependencies
echo ""
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip --quiet
pip install . --quiet
deactivate

# Create config file from example if it doesn't exist
if [ ! -f "config.yaml" ] && [ -f "config.yaml.example" ]; then
    cp config.yaml.example config.yaml
    echo "Created config.yaml from template"
fi

echo ""
echo "================================"
echo "Setup complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "  1. Edit config.yaml to set your preferences"
echo "  2. Run: ./run_captioning.sh /path/to/images"
echo ""
echo "For help:"
echo "  ./run_captioning.sh --help"
echo ""
