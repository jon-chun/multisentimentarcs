import os
import logging
import ollama
from datetime import datetime

# Constants
MAX_CALL_OLLAMA = 5
ROOT_INPUT_DIRECTORY = "../data/keyframes"
ROOT_OUTPUT_DIRECTORY = "../data/sentiments_keyframes"

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

def get_image_sentiment_description(image_path, frame_number):
    """
    Describe the sentiment or emotions evoked by film-maker elements in a movie still image.

    Parameters:
        image_path (str): The path to the image file.
        frame_number (int): The frame number being processed.

    Returns:
        float: The description of the image sentiment as a float, or 0.0 if unsuccessful.
    """
    if not os.path.exists(image_path):
        logging.error(f"Image file does not exist: {image_path}")
        return 0.0

    for attempt in range(1, MAX_CALL_OLLAMA + 1):
        print(f"Processing frame #{frame_number}, attempt #{attempt}")
        try:
            logging.info(f"Attempt {attempt}: Sending image to Ollama for sentiment analysis: {image_path}")

            res = ollama.chat(
                model="llava-llama3-sentiment",
                messages=[
                    {
                        'role': 'user',
                        'content': 'Respond with one floating point number between [-1.0 to 1.0] for the sentiment polarity of this movie still based on the sentiment evoked in this image by aspects like Facial Expression, Camera Angle, Lighting, Framing and Composition, Setting and Background, Color, Body Language and Gestures, Props and Costumes, Depth of Field, Character Positioning and Interaction, Visual Effects and Post-Processing. Do not give a description, greeting, summary, etc. Only respond with one floating point number representing the sentiment polarity of the input image.',
                        'images': [
                            image_path
                        ]
                    }
                ]
            )

            if 'message' in res and 'content' in res['message']:
                image_sentiment_float_str = res['message']['content']
                print(f"  OLLAMA RESPONSE: {image_sentiment_float_str}")
                try:
                    image_sentiment_float = float(image_sentiment_float_str)
                    logging.info(f"Received description for image {image_path} and successfully converted to float")
                    return image_sentiment_float
                except ValueError:
                    print(f"Attempt {attempt}: Could not convert response to float: {image_sentiment_float_str}")
                    logging.warning(f"Attempt {attempt}: Could not convert response to float: {image_sentiment_float_str}")
            else:
                logging.error(f"Attempt {attempt}: Unexpected API response format: {res}")

        except Exception as e:
            logging.error(f"Attempt {attempt}: Error during sentiment analysis for image {image_path}: {e}")

    logging.error(f"All {MAX_CALL_OLLAMA} attempts failed for image {image_path}. Returning 0.0")
    return 0.0

def process_keyframes(leaf_dirs):
    """Process all keyframe images in the leaf directories."""
    frame_number = 1
    for leaf_dir in leaf_dirs:
        keyframes = [f for f in os.listdir(leaf_dir) if os.path.isfile(os.path.join(leaf_dir, f))]
        for keyframe in keyframes:
            keyframe_path = os.path.join(leaf_dir, keyframe)
            description = get_image_sentiment_description(keyframe_path, frame_number)
            save_description(leaf_dir, keyframe, description)
            frame_number += 1

def save_description(leaf_dir, keyframe, description):
    """Save the sentiment description to the corresponding output directory."""
    relative_path = os.path.relpath(leaf_dir, ROOT_INPUT_DIRECTORY)
    output_dir = os.path.join(ROOT_OUTPUT_DIRECTORY, relative_path)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{os.path.splitext(keyframe)[0]}_description.txt")
    with open(output_file, 'w') as f:
        f.write(str(description))

    logging.info(f"Saved description for {keyframe} at {output_file}")

def main():
    """Main function to initiate the processing of keyframes."""
    leaf_dirs = find_leaf_directories(ROOT_INPUT_DIRECTORY)
    process_keyframes(leaf_dirs)
    logging.info("Sentiment analysis for keyframes completed.")

if __name__ == "__main__":
    main()
