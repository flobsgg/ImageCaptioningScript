#!/bin/bash

# Run the captioning script with venv Python
cd "$(dirname "$0")"
./venv/bin/python ImageCaptioning.py "$@"
