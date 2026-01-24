# Image Captioning for Character LoRA Training

Automatically generate image captions for **character LoRA** training using Google Gemini AI.

> Note: This tool is optimized for character/person LoRAs. It describes clothing, poses, and environment while ignoring identity features (hair, skin, eyes) that the LoRA should learn.

## Quick Start

1. Get your free API key: [Google AI Studio](https://aistudio.google.com/apikey)
2. Double-click `START.command`
3. Follow the prompts - done!

## What It Does

- Scans your image folder for JPG, PNG, WEBP files
- Creates `.txt` caption files for each image
- Describes: clothing, pose, expression, background, lighting
- Ignores: hair, skin, eyes, facial features (perfect for character LoRAs)

## Example Caption

```
ohwx man, medium shot, wearing navy blue suit, confident expression,
standing with hands clasped, modern office, soft natural lighting
```

## Project Structure

```
ImageCaptionScript/
├── HOW TO START.txt    ← Quick instructions
├── START.command       ← Double-click to run (macOS)
└── .src/               ← Source code (hidden)
```

## Features

- **Interactive Setup**: No config files - just follow the prompts
- **Batch Processing**: Multiple images per API call for efficiency
- **Auto-Save Settings**: Your preferences are saved for next time
- **Skip Existing**: Already captioned images are skipped
- **Progress Bar**: Visual progress tracking

## CLI Options (Advanced)

```bash
./.src/run_captioning.sh /path/to/images
./.src/run_captioning.sh /path/to/images --trigger "ohwx woman"
./.src/run_captioning.sh /path/to/images --model gemini-2.0-flash
```

| Option | Description |
|--------|-------------|
| `--trigger` | Trigger word for captions |
| `--style-tags` | Style tags to append |
| `--batch-size` | Images per API call |
| `--model` | Gemini model to use |
| `--api-key` | Gemini API key |

## Requirements

- macOS
- Python 3.9+ (auto-installed if missing)
- Free Google Gemini API key

## License

MIT
