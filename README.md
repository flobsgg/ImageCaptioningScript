# Image Captioning for LoRA Training

A command-line tool that generates detailed image captions using Google's Gemini API. Designed specifically for character LoRA training datasets where identity features (hair, skin, eyes, facial structure) should be excluded from descriptions.

## Features

- **Batch Processing**: Process multiple images in a single API call for efficiency
- **Progress Bar**: Visual progress tracking with tqdm
- **Config File**: Store settings in `config.yaml` instead of passing CLI args
- **Automatic Fallback**: Falls back to single-image processing if batch fails
- **Skip Existing**: Automatically skips images that already have caption files
- **Detailed Reports**: Generates processing reports with statistics

## Quick Start

### 1. Get a Gemini API Key

Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

### 2. Download & Setup

```bash
git clone https://github.com/YOUR_USERNAME/ImageCaptionScript.git
cd ImageCaptionScript
chmod +x setup.sh run_captioning.sh
./setup.sh
```

The setup script will:
- Check for Python 3.9+ (and guide installation if missing)
- Create a virtual environment
- Install all dependencies
- Create `config.yaml` from template

### 3. Configure

Edit `config.yaml` to set your preferences:

```yaml
trigger_word: "ohwx man"
model: "gemini-2.5-pro"
batch_size: 8
# api_key: "your-key-here"  # Or use environment variable
```

### 4. Run

```bash
./run_captioning.sh /path/to/images
```

## Configuration

Settings can be configured in three ways (in order of priority):

1. **CLI arguments** (highest priority)
2. **config.yaml** file
3. **Default values** (lowest priority)

### config.yaml

```yaml
trigger_word: "ohwx man"      # Prepended to all captions
style_tags: ""                 # Appended to captions (optional)
model: "gemini-2.5-pro"        # Gemini model to use
batch_size: 8                  # Images per API call
log_level: "INFO"              # DEBUG, INFO, WARNING, ERROR
# api_key: "your-key"          # Or use GEMINI_API_KEY env var
```

### Environment Variable

```bash
export GEMINI_API_KEY="your-api-key"
./run_captioning.sh /path/to/images
```

### CLI Options

```bash
./run_captioning.sh /path/to/images --trigger "ohwx woman" --batch-size 4
```

| Option | Description |
|--------|-------------|
| `--trigger` | Trigger word prepended to captions |
| `--style-tags` | Style tags appended to captions |
| `--batch-size` | Images per API call |
| `--model` | Gemini model to use |
| `--api-key` | Gemini API key |
| `--config` | Path to config file |
| `--log-level` | Logging verbosity |

## How It Works

1. Scans the directory for `.jpg`, `.jpeg`, `.png`, and `.webp` images
2. Skips images that already have a `.txt` caption file
3. Processes images in batches using Gemini's multimodal API
4. Shows progress with a visual progress bar
5. Generates captions describing:
   - Camera angle and composition
   - Clothing and accessories
   - Facial expression and emotion
   - Pose and action
   - Environment and background
   - Lighting and atmosphere
6. Saves captions as `.txt` files next to each image

## Caption Format

```
ohwx man, medium shot, wearing navy blue suit and white shirt, confident expression, standing with hands clasped, modern office with glass windows, soft natural lighting
```

## Development

### Code Quality

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Check code
ruff check .

# Format code
ruff format .
```

### Manual Installation

```bash
python -m venv venv
source venv/bin/activate
pip install .
python ImageCaptioning.py /path/to/images
```

## Requirements

- Python 3.9+
- Google Gemini API key (free)

## License

MIT License - see [LICENSE](LICENSE) for details.
