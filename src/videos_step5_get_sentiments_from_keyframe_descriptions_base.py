import os
import logging
import ollama
from datetime import datetime

# Constants
MAX_CALL_OLLAMA = 5
ROOT_INPUT_DIRECTORY = "../data/keyframes_descriptions"
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
        image_sentiment_text_description = "This text has a neutral sentimental"
        return image_sentiment_text_description

    for attempt in range(1, MAX_CALL_OLLAMA + 1):
        print(f"Processing frame #{frame_number}, attempt #{attempt}")
        try:
            logging.info(f"Attempt {attempt}: Sending image to Ollama for sentiment analysis: {image_path}")


            res = ollama.chat(
                model="llava-llama3",
                messages=[
                    {
                        'role': 'user',
                        'content': 'Describe this movie still and the sentiment or emotions evoked by film-maker elements including Facial Expression, Camera Angle, Lighting, Framing and Composition, Setting and Background, Color, Body Language and Gestures, Props and Costumes, Depth of Field, Character Positioning and Interaction, Visual Effects and Post-Processing.',
                        'images': [
                            image_path
                        ],
                        'temperature': 0.5,  # https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683
                        'top_p': 0.5  # Adjust this value as needed
                    }
                ]
            )

            if 'message' in res and 'content' in res['message']:
                image_sentiment_text_description = res['message']['content']
                if len(image_sentiment_text_description) < MIN_DESCRIPTION_LEN:
                    image_sentiment_text_description = "This text has a neutral sentimental"
                print(f"  image_sentiment_text_description:\n{image_sentiment_text_description}\n\n")
                return image_sentiment_text_description
            
            else:
                image_sentiment_text_description = "This text has a neutral sentimental"
                logging.error(f"Attempt {attempt}: Unexpected API response format: {res}")

        except Exception as e:
            logging.error(f"Attempt {attempt}: Error during sentiment analysis for image {image_path}: {e}")

    logging.error(f"All {MAX_CALL_OLLAMA} attempts failed for image {image_path}. Returning 0.0")
    image_sentiment_text_description = "This text has a neutral sentimental"
    return image_sentiment_text_description


def process_keyframes(leaf_dirs):
    """Process all keyframe images in the leaf directories."""
    frame_number = 1
    leaf_sorted_dirs = sorted(leaf_dirs)
    for leaf_dir in leaf_sorted_dirs:
        keyframes = [f for f in os.listdir(leaf_dir) if os.path.isfile(os.path.join(leaf_dir, f))]
        keyframes_sorted = sorted(keyframes)
        keyframes_total = len(keyframes_sorted)
        for keyframe_index, keyframe in enumerate(keyframes_sorted):
            # Check if the output file already exists
            relative_path = os.path.relpath(leaf_dir, ROOT_INPUT_DIRECTORY)
            output_dir = os.path.join(ROOT_OUTPUT_DIRECTORY, relative_path)
            output_file = os.path.join(output_dir, f"{os.path.splitext(keyframe)[0]}_description.txt")
            if os.path.exists(output_file):
                logging.info(f"Skipping {keyframe} as description already exists at {output_file}")
                continue

            print(f"  PROCESSING #{keyframe_index}/{keyframes_total}...")
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
    # leaf_dirs.remove('../data/stills/deleted')
    leaf_sorted_dirs = sorted(leaf_dirs)
    print(f'leaf_dirs: {leaf_sorted_dirs}\n\n\n')
    process_keyframes(leaf_sorted_dirs)
    logging.info("Sentiment analysis for keyframes completed.")

if __name__ == "__main__":
    main()
