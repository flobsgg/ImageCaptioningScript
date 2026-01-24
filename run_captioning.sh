#!/bin/bash

# Activate virtual environment and run the captioning script
cd "$(dirname "$0")"
source venv/bin/activate
python ImageCaptioning.py "$@"
