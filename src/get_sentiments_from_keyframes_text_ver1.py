import os
import logging
import cv2
import requests
from datetime import datetime
import ollama



# Set up logging to both terminal and the generated log file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Constants
MAX_CALL_OLLAMA = 5
MIN_IMAGE_DESCRIPTION = 100  # Set a minimum length for the description text

ROOT_INPUT_DIRECTORY = "../data/keyframes"
ROOT_OUTPUT_DIRECTORY = "../data/sentiments_keyframes"
OLLAMA_API_URL = "https://api.ollama.com/analyze_image"

# Set up logging to both terminal and the generated log file
current_time = datetime.now().strftime("%Y%m%d_%H%M")
log_file = os.path.join("..", "data", f"keyframe_sentiment_analysis_{current_time}.log")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])

def find_leaf_directories(base_path):
    """Find all terminal leaf directories within the base path."""
    leaf_dirs = []
    for root, dirs, files in os.walk(base_path):
        if not dirs:  # No subdirectories, it's a leaf
            leaf_dirs.append(root)
    logging.info(f"Leaf directories found: {leaf_dirs}")
    return leaf_dirs

def get_image_sentiment_description(image_path):
    """
    Describe the sentiment or emotions evoked by film-maker elements in a movie still image.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        str: The description of the image sentiment, or an empty string if unsuccessful.
    """
    if not os.path.exists(image_path):
        logging.error(f"Image file does not exist: {image_path}")
        return ""

    for attempt in range(1, MAX_CALL_OLLAMA + 1):
        try:
            logging.info(f"\n\nAttempt {attempt}: Sending image to Ollama for sentiment analysis: {image_path}")
            print(f"Processing image {image_path}, attempt #{attempt}")

            res = ollama.chat(
                model="llava-llama3-sentiment",
                messages=[
                    {
                        'role': 'user',
                        'content': 'Concisely describe only directly observable elements in this movie image that evokes sentiment or emotions using common film-maker elements like Facial Expression, Camera Angle, Lighting, Framing and Composition, Setting and Background, Color, Body Language and Gestures, Props and Costumes, Depth of Field, Character Positioning and Interaction, Visual Effects and Post-Processing. Do not provide any facts, context, descriptions, or other information not directly contained in the image itself (e.g. do not name specific film titles, roles, actors, etc).',
                        'images': [
                            image_path
                        ]
                    }
                ]
            )

            if 'message' in res and 'content' in res['message']:
                image_sentiment_description = res['message']['content']
                if len(image_sentiment_description) > MIN_IMAGE_DESCRIPTION:
                    logging.info(f"Received valid description for image {image_path}")
                    print(f"Received valid description for image {image_path}")
                    return image_sentiment_description
                else:
                    logging.warning(f"Attempt {attempt}: Description too short for image {image_path}")
                    print(f"Attempt {attempt}: Description too short for image {image_path}")
            else:
                logging.error(f"Attempt {attempt}: Unexpected API response format: {res}")

        except Exception as e:
            logging.error(f"Attempt {attempt}: Error during sentiment analysis for image {image_path}: {e}")
            print(f"Attempt {attempt}: Error during sentiment analysis for image {image_path}: {e}")

    logging.error(f"All {MAX_CALL_OLLAMA} attempts failed for image {image_path}. Returning empty description.")
    print(f"All {MAX_CALL_OLLAMA} attempts failed for image {image_path}. Returning empty description.")
    return ""

def process_keyframes(leaf_dirs):
    """Process all keyframe images in the leaf directories."""
    frame_number = 1
    for leaf_dir in leaf_dirs:
        keyframes = [f for f in os.listdir(leaf_dir) if os.path.isfile(os.path.join(leaf_dir, f))]
        for keyframe in keyframes:
            keyframe_path = os.path.join(leaf_dir, keyframe)
            description = get_image_sentiment_description(keyframe_path)
            save_description(leaf_dir, keyframe, description)
            frame_number += 1

def save_description(leaf_dir, keyframe, description):
    """Save the sentiment description to the corresponding output directory."""
    relative_path = os.path.relpath(leaf_dir, ROOT_INPUT_DIRECTORY)
    output_dir = os.path.join(ROOT_OUTPUT_DIRECTORY, relative_path)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{os.path.splitext(keyframe)[0]}_description.txt")
    with open(output_file, 'w') as f:
        f.write(description)

    logging.info(f"Saved description for {keyframe} at {output_file}")
    print(f"Saved description for {keyframe} at {output_file}")

def main():
    """Main function to initiate the processing of keyframes."""
    leaf_dirs = find_leaf_directories(ROOT_INPUT_DIRECTORY)
    process_keyframes(leaf_dirs)
    logging.info("Sentiment analysis for keyframes completed.")
    print("Sentiment analysis for keyframes completed.")

if __name__ == "__main__":
    main()