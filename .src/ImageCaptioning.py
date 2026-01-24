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
from datetime import datetime
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
SINGLE_IMAGE_INSTRUCTION = """You are analyzing an image for a character LoRA training dataset. The goal is to describe EVERYTHING EXCEPT the character's permanent physical identity features. Your analysis must be objective, descriptive, and detailed, avoiding subjective words like 'beautiful' or 'amazing'. Provide the output as a single, continuous line of comma-separated keywords and phrases.

**CRITICAL RULES - DO NOT DESCRIBE:**
1.  **Hair:** Do not mention hair color, style, length, or texture
2.  **Skin:** Do not mention skin tone, complexion, or skin characteristics
3.  **Eyes:** Do not mention eye color, shape, or eye features
4.  **Facial Features:** Do not describe facial structure, bone structure, nose, lips, ears, or any permanent facial characteristics
5.  **Body Type:** Do not describe height, build, or body proportions

**ALWAYS DESCRIBE:**
1.  **Camera & Composition:** Shot type, angle, framing (e.g., `close-up shot`, `wide angle`, `low angle view`, `over-the-shoulder`)
2.  **Clothing & Accessories:** All clothing items, jewelry, watches, hats, glasses, bags, etc. (e.g., `wearing black leather jacket`, `blue jeans`, `silver necklace`, `aviator sunglasses`)
3.  **Facial Expression & Emotion:** Current emotion, mood, expression (e.g., `smiling`, `serious expression`, `laughing`, `looking concerned`, `neutral expression`)
4.  **Pose & Action:** Body position, gestures, activities (e.g., `standing with arms crossed`, `sitting on chair`, `walking`, `hands in pockets`, `looking over shoulder`)
5.  **Background & Environment:** Location, setting, surroundings (e.g., `outdoor park`, `urban street`, `modern office`, `blurred background`, `brick wall behind`)
6.  **Lighting & Atmosphere:** Light direction, quality, mood (e.g., `natural daylight`, `dramatic side lighting`, `soft shadows`, `golden hour lighting`, `high contrast`)

**Output Structure (follow this order precisely):**
1.  **Shot Type & Composition:** Camera angle and framing
2.  **Clothing & Accessories:** What the person is wearing
3.  **Expression & Emotion:** Facial expression and mood
4.  **Pose & Action:** Body position and activity
5.  **Environment & Background:** Location and setting
6.  **Lighting & Atmosphere:** Light quality and mood

**Example Output:**
close-up shot, wearing white t-shirt and denim jacket, smiling warmly, standing with hands in pockets, outdoor urban setting with graffiti wall, natural daylight with soft shadows"""

BATCH_INSTRUCTION = """You are analyzing multiple images for a character LoRA training dataset. The goal is to describe EVERYTHING EXCEPT the character's permanent physical identity features. You will analyze multiple images and provide individual detailed descriptions for each one.

**CRITICAL REQUIREMENTS:**
1. Analyze each image INDIVIDUALLY and SEPARATELY - do not reference other images
2. Each analysis must be as detailed as if you were analyzing just that single image
3. Output must be valid JSON with image filenames as keys
4. Each description must be objective, descriptive, and detailed
5. Avoid subjective words like 'beautiful' or 'amazing'
6. Each description should be a single, continuous line of comma-separated keywords and phrases

**CRITICAL RULES - DO NOT DESCRIBE:**
1.  **Hair:** Do not mention hair color, style, length, or texture
2.  **Skin:** Do not mention skin tone, complexion, or skin characteristics
3.  **Eyes:** Do not mention eye color, shape, or eye features
4.  **Facial Features:** Do not describe facial structure, bone structure, nose, lips, ears, or any permanent facial characteristics
5.  **Body Type:** Do not describe height, build, or body proportions

**ALWAYS DESCRIBE (for each image individually):**
1.  **Camera & Composition:** Shot type, angle, framing (e.g., `close-up shot`, `wide angle`, `low angle view`, `over-the-shoulder`)
2.  **Clothing & Accessories:** All clothing items, jewelry, watches, hats, glasses, bags, etc. (e.g., `wearing black leather jacket`, `blue jeans`, `silver necklace`, `aviator sunglasses`)
3.  **Facial Expression & Emotion:** Current emotion, mood, expression (e.g., `smiling`, `serious expression`, `laughing`, `looking concerned`, `neutral expression`)
4.  **Pose & Action:** Body position, gestures, activities (e.g., `standing with arms crossed`, `sitting on chair`, `walking`, `hands in pockets`, `looking over shoulder`)
5.  **Background & Environment:** Location, setting, surroundings (e.g., `outdoor park`, `urban street`, `modern office`, `blurred background`, `brick wall behind`)
6.  **Lighting & Atmosphere:** Light direction, quality, mood (e.g., `natural daylight`, `dramatic side lighting`, `soft shadows`, `golden hour lighting`, `high contrast`)

**Output Structure (follow this order precisely for each image):**
1.  **Shot Type & Composition:** Camera angle and framing
2.  **Clothing & Accessories:** What the person is wearing
3.  **Expression & Emotion:** Facial expression and mood
4.  **Pose & Action:** Body position and activity
5.  **Environment & Background:** Location and setting
6.  **Lighting & Atmosphere:** Light quality and mood

**OUTPUT FORMAT:**
Return a JSON object where each key is the image filename and each value is the detailed description:

{
  "image1.jpg": "close-up shot, wearing white t-shirt and denim jacket, smiling warmly, standing with hands in pockets, outdoor urban setting with graffiti wall, natural daylight with soft shadows",
  "image2.jpg": "medium shot, wearing black suit and tie, serious expression, sitting at desk, modern office interior, soft overhead lighting",
  "image3.jpg": "wide angle shot, wearing casual hoodie and jeans, laughing, walking on sidewalk, city street background, golden hour lighting"
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


def save_batch_report(
    dataset_path: str,
    total_images: int,
    processed_count: int,
    skipped_count: int,
    api_calls_made: int,
    single_fallback_count: int,
    batch_size: int,
    start_time: datetime,
    model_name: str,
    trigger_word: str,
    style_tags: str,
) -> None:
    """Save detailed batch processing report to data folder."""
    try:
        dataset_path_obj = Path(dataset_path)
        dataset_name = dataset_path_obj.name
        parent_dir = dataset_path_obj.parent
        data_folder = parent_dir / f"{dataset_name}_data"

        data_folder.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        report_filename = f"{timestamp}_{dataset_name}_Batch_Processing_Report.txt"
        report_path = data_folder / report_filename

        end_time = datetime.now()
        processing_time = end_time - start_time
        estimated_single_calls = processed_count + single_fallback_count
        api_savings = max(0, estimated_single_calls - api_calls_made)
        efficiency = (
            (processed_count - single_fallback_count) / api_calls_made if api_calls_made > 0 else 0
        )
        batch_success_rate = (
            (api_calls_made - single_fallback_count) / api_calls_made * 100
            if api_calls_made > 0
            else 0
        )

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Batch Caption Generation Report\n")
            f.write("=" * 35 + "\n\n")
            f.write(f"Generated: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Source Path: {dataset_path}\n")
            f.write(f"Processing Time: {processing_time}\n\n")

            f.write("Processing Configuration:\n")
            f.write(f"  Batch Size: {batch_size} images per API call\n")
            f.write(f"  Model: {model_name}\n\n")

            f.write("Processing Results:\n")
            f.write(f"  Total Images Found: {total_images}\n")
            f.write(f"  Images Skipped (already processed): {skipped_count}\n")
            f.write(f"  Images to Process: {processed_count + single_fallback_count}\n")
            f.write(f"  Images Successfully Processed: {processed_count}\n\n")

            f.write("API Usage Statistics:\n")
            f.write(f"  Total API Calls Made: {api_calls_made}\n")
            f.write(f"  Batch Processing Calls: {api_calls_made - single_fallback_count}\n")
            f.write(f"  Single Image Fallback Calls: {single_fallback_count}\n")
            f.write(f"  Estimated Single-Image Processing: {estimated_single_calls} calls\n")
            savings_pct = (
                (api_savings / estimated_single_calls * 100) if estimated_single_calls > 0 else 0
            )
            f.write(f"  API Calls Saved: {api_savings} ({savings_pct:.1f}% reduction)\n\n")

            f.write("Efficiency Metrics:\n")
            f.write(f"  Batch Success Rate: {batch_success_rate:.1f}%\n")
            f.write(f"  Average Images per API Call: {efficiency:.1f}\n")
            rate = (
                processed_count / processing_time.total_seconds() * 60
                if processing_time.total_seconds() > 0
                else 0
            )
            f.write(f"  Processing Rate: {rate:.1f} images/minute\n\n")

            f.write("Quality Assurance:\n")
            f.write(f"  Trigger Word: {trigger_word}\n")
            f.write(f"  Style Tags: {style_tags}\n")
            f.write("  Fallback Strategy: Single-image processing for failed batches\n")

        logger.info(f"Report saved to {report_path}")

    except Exception as e:
        logger.warning(f"Could not save batch report: {e}")


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

    start_time = datetime.now()

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

    save_batch_report(
        dataset_path=image_directory,
        total_images=len(image_paths),
        processed_count=processed_count,
        skipped_count=skipped_count,
        api_calls_made=api_calls_made,
        single_fallback_count=single_fallback_count,
        batch_size=batch_size,
        start_time=start_time,
        model_name=model_name,
        trigger_word=trigger_word,
        style_tags=style_tags,
    )


if __name__ == "__main__":
    main()
