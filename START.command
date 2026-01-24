#!/bin/bash

# =========================================
#   Image Captioning Tool for LoRA
#   Just double-click to start!
# =========================================

cd "$(dirname "$0")/.src"

# Check if setup is needed
if [ ! -d "venv" ]; then
    echo ""
    echo "  ┌─────────────────────────────────────┐"
    echo "  │   First time setup - please wait   │"
    echo "  └─────────────────────────────────────┘"
    echo ""
    ./setup.sh
    echo ""
    sleep 1
fi

# Run the tool
./run_captioning.sh

# Keep window open
echo ""
read -p "Press Enter to close..."
