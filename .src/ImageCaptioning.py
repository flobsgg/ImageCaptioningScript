#!/usr/bin/env python3
"""
Image Captioning Tool for LoRA Training

Generates detailed captions for images using Google's Gemini API.
Designed for character LoRA training datasets where identity features
should be excluded from descriptions.
"""

# Suppress deprecation warnings from Google libraries
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*urllib3.*")

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import google.generativeai as genai
import questionary
from dotenv import load_dotenv, set_key
from PIL import Image
from questionary import Style
from tqdm import tqdm

# --- DEFAULT CONFIGURATION ---
DEFAULTS = {
    "trigger_word": "ohwx man",
    "style_tags": "",
    "model": "gemini-2.5-pro",
    "batch_size": 8,
    "log_level": "INFO",
}

# Custom style for questionary prompts
PROMPT_STYLE = Style([
    ("qmark", "fg:cyan bold"),
    ("question", "fg:white bold"),
    ("answer", "fg:green"),
    ("pointer", "fg:cyan bold"),
    ("highlighted", "fg:cyan bold"),
    ("selected", "fg:green"),
    ("instruction", "fg:gray"),
])

# --- PROMPTS ---
SINGLE_IMAGE_INSTRUCTION = """You are analyzing an image for a character LoRA training dataset. The goal is to describe EVERYTHING EXCEPT the character's permanent physical identity features. Your analysis must be objective, descriptive, and detailed, avoiding subjective words like 'beautiful' or 'amazing'. Write concise, objective, and descriptive sentences.

**CRITICAL RULES - DO NOT DESCRIBE:**
1.  **Hair:** Do not mention hair color, style, length, or texture
2.  **Skin:** Do not mention skin tone, complexion, or skin characteristics
3.  **Eyes:** Do not mention eye color, shape, or eye features
4.  **Facial Features:** Do not describe facial structure, bone structure, nose, lips, ears, or any permanent facial characteristics
5.  **Body Type:** Do not describe height, build, or body proportions

**SUBJECT REFERENCE:**
Refer to the subject using gendered terms with an age descriptor (e.g., "a young woman", "a middle-aged man", "an elderly woman"). Use appropriate pronouns (she/her or he/his) throughout the description. Do NOT use neutral terms like "the subject", "the person", or "they".

**ALWAYS DESCRIBE:**
1.  **Camera & Composition:** Shot type, angle, framing (e.g., "A close-up shot of a young woman framed from the shoulders up", "A wide-angle view of an older man from a low angle")
2.  **Clothing & Accessories:** All clothing items, jewelry, watches, hats, glasses, bags, etc. (e.g., "She wears a black leather jacket over a white t-shirt", "He has aviator sunglasses and a silver necklace")
3.  **Facial Expression & Emotion:** Current emotion, mood, expression (e.g., "Her expression is warm and relaxed, with a gentle smile", "He appears focused and serious")
4.  **Pose & Action:** Body position, gestures, activities (e.g., "She stands with arms crossed and weight shifted to one side", "He is seated at a desk, leaning forward")
5.  **Background & Environment:** Location, setting, surroundings (e.g., "The background shows an urban street with graffiti-covered walls", "A modern office interior is visible behind her")
6.  **Lighting & Atmosphere:** Light direction, quality, mood (e.g., "Natural daylight illuminates the scene from the left, creating soft shadows", "Dramatic side lighting produces high contrast")
7.  **Focus & Depth of Field:** Background sharpness, bokeh quality (e.g., "The background is softly blurred with pleasant bokeh", "Everything remains sharp and in focus throughout the frame")

**Output Structure (follow this order precisely):**
1.  **Shot Type & Composition:** Camera angle and framing
2.  **Clothing & Accessories:** What the person is wearing
3.  **Expression & Emotion:** Facial expression and mood
4.  **Pose & Action:** Body position and activity
5.  **Environment & Background:** Location and setting
6.  **Lighting & Atmosphere:** Light quality and mood
7.  **Focus & Depth of Field:** Background sharpness and blur

**Example Output:**
A close-up shot frames a young man from the shoulders up. He wears a white t-shirt layered under a denim jacket. His expression is warm and friendly, with a relaxed smile. He stands casually with hands tucked into his pockets. The background features an urban setting with a colorful graffiti wall. Natural daylight creates soft, even shadows across the scene. The background remains sharp and fully in focus."""

BATCH_INSTRUCTION = """You are analyzing multiple images for a character LoRA training dataset. The goal is to describe EVERYTHING EXCEPT the character's permanent physical identity features. You will analyze multiple images and provide individual detailed descriptions for each one.

**CRITICAL REQUIREMENTS:**
1. Analyze each image INDIVIDUALLY and SEPARATELY - do not reference other images
2. Each analysis must be as detailed as if you were analyzing just that single image
3. Output must be valid JSON with image filenames as keys
4. Each description must be objective, descriptive, and detailed
5. Avoid subjective words like 'beautiful' or 'amazing'
6. Each description should be written as concise, objective, and descriptive sentences

**CRITICAL RULES - DO NOT DESCRIBE:**
1.  **Hair:** Do not mention hair color, style, length, or texture
2.  **Skin:** Do not mention skin tone, complexion, or skin characteristics
3.  **Eyes:** Do not mention eye color, shape, or eye features
4.  **Facial Features:** Do not describe facial structure, bone structure, nose, lips, ears, or any permanent facial characteristics
5.  **Body Type:** Do not describe height, build, or body proportions

**SUBJECT REFERENCE:**
Refer to the subject using gendered terms with an age descriptor (e.g., "a young woman", "a middle-aged man", "an elderly woman"). Use appropriate pronouns (she/her or he/his) throughout the description. Do NOT use neutral terms like "the subject", "the person", or "they".

**ALWAYS DESCRIBE (for each image individually):**
1.  **Camera & Composition:** Shot type, angle, framing (e.g., "A close-up shot of a young woman framed from the shoulders up", "A wide-angle view of an older man from a low angle")
2.  **Clothing & Accessories:** All clothing items, jewelry, watches, hats, glasses, bags, etc. (e.g., "She wears a black leather jacket over a white t-shirt", "He has aviator sunglasses and a silver necklace")
3.  **Facial Expression & Emotion:** Current emotion, mood, expression (e.g., "Her expression is warm and relaxed, with a gentle smile", "He appears focused and serious")
4.  **Pose & Action:** Body position, gestures, activities (e.g., "She stands with arms crossed and weight shifted to one side", "He is seated at a desk, leaning forward")
5.  **Background & Environment:** Location, setting, surroundings (e.g., "The background shows an urban street with graffiti-covered walls", "A modern office interior is visible behind her")
6.  **Lighting & Atmosphere:** Light direction, quality, mood (e.g., "Natural daylight illuminates the scene from the left, creating soft shadows", "Dramatic side lighting produces high contrast")
7.  **Focus & Depth of Field:** Background sharpness, bokeh quality (e.g., "The background is softly blurred with pleasant bokeh", "Everything remains sharp and in focus throughout the frame")

**Output Structure (follow this order precisely for each image):**
1.  **Shot Type & Composition:** Camera angle and framing
2.  **Clothing & Accessories:** What the person is wearing
3.  **Expression & Emotion:** Facial expression and mood
4.  **Pose & Action:** Body position and activity
5.  **Environment & Background:** Location and setting
6.  **Lighting & Atmosphere:** Light quality and mood
7.  **Focus & Depth of Field:** Background sharpness and blur

**OUTPUT FORMAT:**
Return a JSON object where each key is the image filename and each value is the detailed description:

{
  "image1.jpg": "A close-up shot frames a young man from the shoulders up. He wears a white t-shirt layered under a denim jacket. His expression is warm and friendly, with a relaxed smile. He stands casually with hands tucked into his pockets. The background features an urban setting with a colorful graffiti wall. Natural daylight creates soft, even shadows across the scene. The background remains sharp and fully in focus.",
  "image2.jpg": "A medium shot captures a middle-aged woman from the waist up. She wears a tailored black suit with a matching tie. Her expression is serious and focused. She sits at a wooden desk, leaning slightly forward. A modern office interior with glass partitions is visible behind her. Soft overhead lighting provides even illumination. The background is gently blurred with a shallow depth of field.",
  "image3.jpg": "A wide-angle shot shows the full figure of an older man. He wears a casual grey hoodie and dark jeans. He is laughing openly with genuine amusement. He walks along a city sidewalk mid-stride. The background reveals a busy urban street with storefronts. Golden hour lighting casts warm tones and long shadows. The entire scene remains sharp with a deep depth of field."
}

IMPORTANT: Each description must be complete and detailed as if analyzing only that single image. Do not reference or compare to other images in the batch."""

# Setup logging
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with the specified level."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )


def get_env_path() -> Path:
    """Get the path to the .env file in the script directory."""
    return Path(__file__).parent / ".env"


def load_saved_settings() -> dict[str, Any]:
    """Load settings from .env file."""
    env_path = get_env_path()
    settings = {}

    if env_path.exists():
        load_dotenv(env_path)
        settings["api_key"] = os.getenv("GEMINI_API_KEY", "")
        settings["trigger_word"] = os.getenv("TRIGGER_WORD", DEFAULTS["trigger_word"])
        settings["style_tags"] = os.getenv("STYLE_TAGS", DEFAULTS["style_tags"])
        settings["model"] = os.getenv("MODEL", DEFAULTS["model"])
        settings["batch_size"] = int(os.getenv("BATCH_SIZE", str(DEFAULTS["batch_size"])))

    return settings


def save_settings(settings: dict[str, Any]) -> None:
    """Save settings to .env file."""
    env_path = get_env_path()

    # Create .env file if it doesn't exist
    if not env_path.exists():
        env_path.touch()

    # Save each setting
    if settings.get("api_key"):
        set_key(env_path, "GEMINI_API_KEY", settings["api_key"])
    if settings.get("trigger_word"):
        set_key(env_path, "TRIGGER_WORD", settings["trigger_word"])
    if "style_tags" in settings:
        set_key(env_path, "STYLE_TAGS", settings["style_tags"])
    if settings.get("model"):
        set_key(env_path, "MODEL", settings["model"])
    if settings.get("batch_size"):
        set_key(env_path, "BATCH_SIZE", str(settings["batch_size"]))


def print_banner() -> None:
    """Print the application banner."""
    print()
    print("  ┌─────────────────────────────────────┐")
    print("  │   Image Captioning Tool             │")
    print("  │   for Character LoRA Training       │")
    print("  └─────────────────────────────────────┘")
    print()


def run_interactive_setup(saved_settings: dict[str, Any], need_directory: bool = True) -> dict[str, Any]:
    """Run interactive setup wizard and return settings."""
    settings = {}

    # API Key
    saved_key = saved_settings.get("api_key", "")
    if saved_key:
        key_preview = f"{saved_key[:4]}...{saved_key[-4:]}"
        use_saved = questionary.confirm(
            f"Use saved API key ({key_preview})?",
            default=True,
            style=PROMPT_STYLE,
        ).ask()

        if use_saved is None:  # User pressed Ctrl+C
            sys.exit(0)

        if use_saved:
            settings["api_key"] = saved_key
        else:
            api_key = questionary.password(
                "Enter your Gemini API key:",
                style=PROMPT_STYLE,
            ).ask()
            if api_key is None:
                sys.exit(0)
            settings["api_key"] = api_key
    else:
        print("  Get your free API key at: https://aistudio.google.com/apikey")
        print()
        api_key = questionary.password(
            "Enter your Gemini API key:",
            style=PROMPT_STYLE,
        ).ask()
        if api_key is None:
            sys.exit(0)
        settings["api_key"] = api_key

    # Trigger Word
    saved_trigger = saved_settings.get("trigger_word", DEFAULTS["trigger_word"])
    trigger_word = questionary.text(
        "Trigger word for your LoRA:",
        default=saved_trigger,
        style=PROMPT_STYLE,
    ).ask()
    if trigger_word is None:
        sys.exit(0)
    settings["trigger_word"] = trigger_word or DEFAULTS["trigger_word"]

    # Style Tags (optional)
    saved_style = saved_settings.get("style_tags", "")
    style_tags = questionary.text(
        "Style tags to append (optional):",
        default=saved_style,
        style=PROMPT_STYLE,
    ).ask()
    if style_tags is None:
        sys.exit(0)
    settings["style_tags"] = style_tags or ""

    # Model selection
    saved_model = saved_settings.get("model", DEFAULTS["model"])
    model = questionary.select(
        "Select Gemini model:",
        choices=[
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ],
        default=saved_model if saved_model in [
            "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash",
            "gemini-1.5-pro", "gemini-1.5-flash"
        ] else DEFAULTS["model"],
        style=PROMPT_STYLE,
    ).ask()
    if model is None:
        sys.exit(0)
    settings["model"] = model

    # Batch size
    saved_batch = saved_settings.get("batch_size", DEFAULTS["batch_size"])
    batch_size = questionary.select(
        "Images per API call (batch size):",
        choices=["4", "6", "8", "10", "12"],
        default=str(saved_batch) if str(saved_batch) in ["4", "6", "8", "10", "12"] else "8",
        style=PROMPT_STYLE,
    ).ask()
    if batch_size is None:
        sys.exit(0)
    settings["batch_size"] = int(batch_size)

    # Image directory
    if need_directory:
        print()
        print("  Tip: Drag & drop folder into terminal, or type path")
        image_dir = questionary.text(
            "Path to image folder:",
            style=PROMPT_STYLE,
        ).ask()
        if image_dir is None:
            sys.exit(0)
        # Clean up path (remove quotes and trailing spaces from drag & drop)
        image_dir = image_dir.strip().strip("'\"")
        settings["image_directory"] = image_dir

    # Save settings
    save_settings(settings)
    print()
    print("  ✓ Settings saved to .env")
    print()

    return settings


def process_single_image(
    model: genai.GenerativeModel,
    image_path: Path,
    trigger_word: str,
    style_tags: str,
) -> Optional[str]:
    """Process a single image and return the caption."""
    try:
        img = Image.open(image_path)
        response = model.generate_content([SINGLE_IMAGE_INSTRUCTION, img])
        base_caption = response.text.strip().replace("\n", " ")

        if style_tags:
            return f"{trigger_word}, {base_caption}, {style_tags}"
        return f"{trigger_word}, {base_caption}"

    except Exception as e:
        logger.error(f"Failed to process {image_path.name}: {e}")
        return None


def process_batch_images(
    model: genai.GenerativeModel,
    image_paths: list[Path],
    trigger_word: str,
    style_tags: str,
) -> dict[Path, Optional[str]]:
    """Process multiple images in a single API call."""
    try:
        images = []
        filenames = []

        for image_path in image_paths:
            img = Image.open(image_path)
            images.append(img)
            filenames.append(image_path.name)

        content = [BATCH_INSTRUCTION]
        for i, (img, filename) in enumerate(zip(images, filenames)):
            content.append(f"Image {i + 1} (filename: {filename}):")
            content.append(img)

        response = model.generate_content(content)
        response_text = response.text.strip()

        # Clean up response text - remove code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            parsed_response = json.loads(response_text)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                parsed_response = json.loads(json_match.group())
            else:
                raise ValueError("Could not parse JSON response") from None

        results: dict[Path, Optional[str]] = {}
        for image_path in image_paths:
            filename = image_path.name
            if filename in parsed_response:
                base_caption = parsed_response[filename].strip().replace("\n", " ")
                if style_tags:
                    final_caption = f"{trigger_word}, {base_caption}, {style_tags}"
                else:
                    final_caption = f"{trigger_word}, {base_caption}"
                results[image_path] = final_caption
            else:
                logger.warning(f"No caption for {filename} in batch response")
                results[image_path] = None

        return results

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return {path: None for path in image_paths}


def main() -> None:
    """Main function with batch processing."""
    parser = argparse.ArgumentParser(
        description="Generate image captions for LoRA training using Google Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Interactive mode
  %(prog)s /path/to/images                    # Process with saved settings
  %(prog)s /path/to/images --trigger "ohwx woman"
  %(prog)s /path/to/images --api-key YOUR_KEY --model gemini-2.0-flash

Settings are saved in .env and reused automatically.
        """,
    )

    parser.add_argument(
        "image_directory",
        nargs="?",
        help="Path to the directory containing images",
    )
    parser.add_argument(
        "--api-key",
        help="Gemini API key (or set via interactive setup / .env)",
    )
    parser.add_argument(
        "--trigger",
        help=f"Trigger word to prepend to captions (default: {DEFAULTS['trigger_word']})",
    )
    parser.add_argument(
        "--style-tags",
        help="Style tags to append to captions",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help=f"Number of images per API call (default: {DEFAULTS['batch_size']})",
    )
    parser.add_argument(
        "--model",
        help=f"Gemini model to use (default: {DEFAULTS['model']})",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Load saved settings from .env
    saved_settings = load_saved_settings()

    # Determine if we need interactive mode
    has_api_key = args.api_key or saved_settings.get("api_key")
    has_directory = args.image_directory is not None

    # If no API key available, or no arguments at all -> interactive mode
    if not has_api_key or (not has_directory and len(sys.argv) == 1):
        print_banner()
        settings = run_interactive_setup(saved_settings, need_directory=not has_directory)

        # Merge CLI args over interactive settings
        api_key = args.api_key or settings.get("api_key")
        trigger_word = args.trigger or settings.get("trigger_word", DEFAULTS["trigger_word"])
        style_tags = args.style_tags if args.style_tags is not None else settings.get("style_tags", "")
        batch_size = args.batch_size or settings.get("batch_size", DEFAULTS["batch_size"])
        model_name = args.model or settings.get("model", DEFAULTS["model"])
        image_directory = args.image_directory or settings.get("image_directory")
    else:
        # Use CLI args with saved settings as fallback
        api_key = args.api_key or saved_settings.get("api_key")
        trigger_word = args.trigger or saved_settings.get("trigger_word", DEFAULTS["trigger_word"])
        style_tags = args.style_tags if args.style_tags is not None else saved_settings.get("style_tags", "")
        batch_size = args.batch_size or saved_settings.get("batch_size", DEFAULTS["batch_size"])
        model_name = args.model or saved_settings.get("model", DEFAULTS["model"])
        image_directory = args.image_directory

    # Validate required values
    if not api_key:
        logger.error("API key is required. Run without arguments for interactive setup.")
        return

    if not image_directory:
        logger.error("Image directory is required.")
        return

    # Validate directory
    if not Path(image_directory).is_dir():
        logger.error(f"'{image_directory}' is not a valid directory")
        return

    logger.info(f"Starting batch captioning with {model_name}")
    logger.info(f"Trigger word: {trigger_word}")
    if style_tags:
        logger.info(f"Style tags: {style_tags}")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        logger.error(f"Could not initialize Gemini client: {e}")
        return

    # Collect image paths
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    image_paths: list[Path] = []
    for ext in image_extensions:
        image_paths.extend(Path(image_directory).rglob(ext))
    image_paths.sort()

    if not image_paths:
        logger.error(f"No images found in '{image_directory}'")
        return

    # Filter already processed images
    unprocessed_images: list[Path] = []
    skipped_count = 0

    for image_path in image_paths:
        caption_path = image_path.with_suffix(".txt")
        if caption_path.exists():
            skipped_count += 1
        else:
            unprocessed_images.append(image_path)

    if not unprocessed_images:
        logger.info("All images have already been processed!")
        return

    logger.info(f"Found {len(image_paths)} images, {skipped_count} already processed")
    logger.info(f"Processing {len(unprocessed_images)} images in batches of {batch_size}")

    processed_count = 0
    api_calls_made = 0
    single_fallback_count = 0
    failed_images: list[str] = []

    # Create progress bar for batch progress
    total_batches = (len(unprocessed_images) + batch_size - 1) // batch_size
    pbar = tqdm(total=total_batches, desc="Processing", unit="batch")

    # Batch processing
    for i in range(0, len(unprocessed_images), batch_size):
        batch = unprocessed_images[i : i + batch_size]
        batch_num = (i // batch_size) + 1

        pbar.set_description(f"Batch {batch_num}/{total_batches} ({len(batch)} imgs)")

        batch_results = process_batch_images(model, batch, trigger_word, style_tags)
        api_calls_made += 1

        for image_path in batch:
            caption = batch_results.get(image_path)

            if caption:
                caption_path = image_path.with_suffix(".txt")
                try:
                    with open(caption_path, "w", encoding="utf-8") as f:
                        f.write(caption)
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Failed to save {image_path.name}: {e}")
                    failed_images.append(image_path.name)
            else:
                # Fallback to single image processing
                logger.debug(f"Fallback for {image_path.name}")
                single_caption = process_single_image(model, image_path, trigger_word, style_tags)
                api_calls_made += 1
                single_fallback_count += 1

                if single_caption:
                    caption_path = image_path.with_suffix(".txt")
                    try:
                        with open(caption_path, "w", encoding="utf-8") as f:
                            f.write(single_caption)
                        processed_count += 1
                    except Exception as e:
                        logger.error(f"Failed to save {image_path.name}: {e}")
                        failed_images.append(image_path.name)
                else:
                    failed_images.append(image_path.name)

        pbar.update(1)

        # Pause between API calls
        if i + batch_size < len(unprocessed_images):
            time.sleep(3)

    pbar.close()

    # Summary
    print()  # Empty line after progress bar
    logger.info("=" * 50)
    logger.info("Captioning complete!")
    logger.info(f"Processed: {processed_count}/{len(unprocessed_images)} images")
    logger.info(f"API calls: {api_calls_made} (saved {len(unprocessed_images) - api_calls_made})")

    if failed_images:
        logger.warning(f"Failed: {len(failed_images)} images")
        for name in failed_images[:5]:
            logger.warning(f"  - {name}")
        if len(failed_images) > 5:
            logger.warning(f"  ... and {len(failed_images) - 5} more")


if __name__ == "__main__":
    main()
