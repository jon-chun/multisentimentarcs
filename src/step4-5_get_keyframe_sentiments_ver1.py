import os
import logging
from datetime import datetime
import ollama
from typing import List, Dict
from collections import defaultdict

# Constants
MAX_CALL_OLLAMA = 5
ROOT_INPUT_DIRECTORY = "../data/keyframes"
ROOT_OUTPUT_DIRECTORY = "../data/keyframes_sentiments"
MIN_DESCRIPTION_LEN = 5

# Set up logging to both terminal and the generated log file
current_time = datetime.now().strftime("%Y%m%d_%H%M")
log_file = os.path.join("..", "data", f"keyframe_sentiment_analysis_{current_time}.log")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])

def find_leaf_directories(base_path: str) -> List[str]:
    """Find all terminal leaf directories within the base path."""
    leaf_dirs = [root for root, dirs, _ in os.walk(base_path) if not dirs]
    leaf_dirs.sort()
    logging.info(f"Leaf directories found: {leaf_dirs}")
    return leaf_dirs

def get_image_sentiment_description(image_path: str, frame_number: int) -> str:
    """
    Describe the sentiment or emotions evoked by film-maker elements in a movie still image.

    Parameters:
        image_path (str): The path to the image file.
        frame_number (int): The frame number being processed.

    Returns:
        str: The description of the image sentiment, or a neutral sentiment if unsuccessful.
    """
    if not os.path.exists(image_path):
        logging.error(f"Image file does not exist: {image_path}")
        return "This text has a neutral sentimental"

    for attempt in range(1, MAX_CALL_OLLAMA + 1):
        logging.info(f"Attempt {attempt}: Sending image to Ollama for sentiment analysis: {image_path}")
        try:
            res = ollama.chat(
                model="llava-llama3",
                messages=[
                    {
                        'role': 'user',
                        'content': 'Describe this movie still and the sentiment or emotions evoked by film-maker elements including Facial Expression, Camera Angle, Lighting, Framing and Composition, Setting and Background, Color, Body Language and Gestures, Props and Costumes, Depth of Field, Character Positioning and Interaction, Visual Effects and Post-Processing.',
                        'images': [image_path],
                        'temperature': 0.5,
                        'top_p': 0.5
                    }
                ]
            )

            if 'message' in res and 'content' in res['message']:
                description = res['message']['content']
                if len(description) >= MIN_DESCRIPTION_LEN:
                    logging.info(f"Description for frame #{frame_number}: {description}")
                    return description
                else:
                    logging.warning(f"Attempt {attempt}: Description too short: {description}")

            logging.error(f"Attempt {attempt}: Unexpected API response format: {res}")

        except Exception as e:
            logging.error(f"Attempt {attempt}: Error during sentiment analysis for image {image_path}: {e}")

    logging.error(f"All {MAX_CALL_OLLAMA} attempts failed for image {image_path}. Returning neutral sentiment.")
    return "This text has a neutral sentimental"

def save_description(leaf_dir: str, keyframe: str, description: str) -> None:
    """Save the sentiment description to the corresponding output directory."""
    relative_path = os.path.relpath(leaf_dir, ROOT_INPUT_DIRECTORY)
    output_dir = os.path.join(ROOT_OUTPUT_DIRECTORY, relative_path)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{os.path.splitext(keyframe)[0]}_description.txt")
    with open(output_file, 'w') as f:
        f.write(description)

    logging.info(f"Saved description for {keyframe} at {output_file}")

def process_keyframes(leaf_dirs: List[str]) -> None:
    """Process all keyframe images in the leaf directories."""
    frame_number = 1
    for leaf_dir in sorted(leaf_dirs):
        keyframes = sorted(f for f in os.listdir(leaf_dir) if os.path.isfile(os.path.join(leaf_dir, f)))
        keyframes_by_scene = defaultdict(list)
        for keyframe in keyframes:
            scene_prefix = keyframe.split('_')[0]  # Assuming scene prefix is the part before the first underscore
            keyframes_by_scene[scene_prefix].append(keyframe)

        for scene_prefix in sorted(keyframes_by_scene.keys()):
            for keyframe in sorted(keyframes_by_scene[scene_prefix]):
                keyframe_path = os.path.join(leaf_dir, keyframe)
                relative_path = os.path.relpath(leaf_dir, ROOT_INPUT_DIRECTORY)
                output_file = os.path.join(ROOT_OUTPUT_DIRECTORY, relative_path, f"{os.path.splitext(keyframe)[0]}_description.txt")
                if os.path.exists(output_file):
                    logging.info(f"Skipping already processed keyframe: {keyframe_path}")
                    continue

                logging.info(f"Processing {keyframe_path}...")
                description = get_image_sentiment_description(keyframe_path, frame_number)
                save_description(leaf_dir, keyframe, description)
                frame_number += 1

def main() -> None:
    """Main function to initiate the processing of keyframes."""
    leaf_dirs = find_leaf_directories(ROOT_INPUT_DIRECTORY)
    process_keyframes(leaf_dirs)
    logging.info("Sentiment analysis for keyframes completed.")

if __name__ == "__main__":
    main()
