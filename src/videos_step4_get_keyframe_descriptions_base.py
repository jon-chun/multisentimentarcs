import os
import logging
from datetime import datetime
import ollama
from typing import List, Dict
from collections import defaultdict
import re
import time
import subprocess
import random
import gc
import tracemalloc
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Constants 
MAX_CALL_OLLAMA = 5
MAX_WAIT_OLLAMA_API_SEC = 3000  # 50 minutes
MAX_WAIT_OLLAMA_RESTART_SEC = 60  # 1 minute
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

tracemalloc.start()

def sanitize_filename(filename: str) -> str:
    """Sanitize filenames to remove special characters."""
    return re.sub(r'[<>:"/\\|?*.,;]', '-', filename).replace(' ', '_')

def find_leaf_directories(base_path: str) -> List[str]:
    """Find all terminal leaf directories within the base path."""
    leaf_dirs = [root for root, dirs, _ in os.walk(base_path) if not dirs]
    leaf_dirs.sort()
    logging.info(f"Leaf directories found: {leaf_dirs}")
    return leaf_dirs



tracemalloc.start()

def restart_ollama_server() -> None:
    """Restart the Ollama server using the CLI command."""
    try:
        subprocess.run(["ollama", "serve"], check=True)
        logging.info("Successfully restarted Ollama server.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to restart Ollama server: {e}")

def log_memory_usage():
    current, peak = tracemalloc.get_traced_memory()
    logging.info(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.reset_peak()




import requests

def fetch_image_sentiment(image_path: str) -> Dict:
    return ollama.chat(
        model="llava-llama3sentiment",
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

    with ThreadPoolExecutor() as executor:
        for attempt in range(1, MAX_CALL_OLLAMA + 1):
            logging.info(f"Attempt {attempt}: Sending image to Ollama for sentiment analysis: {image_path}")
            future = executor.submit(fetch_image_sentiment, image_path)
            try:
                res = future.result(timeout=MAX_WAIT_OLLAMA_API_SEC)
                if 'message' in res and 'content' in res['message']:
                    description = res['message']['content']
                    if len(description) >= MIN_DESCRIPTION_LEN:
                        logging.info(f"Description for frame #{frame_number}: {description}")
                        return description
                    else:
                        logging.warning(f"Attempt {attempt}: Description too short: {description}")
                logging.error(f"Attempt {attempt}: Unexpected API response format: {res}")
            except TimeoutError:
                logging.error(f"Attempt {attempt}: API call timed out. Restarting Ollama server.")
                restart_ollama_server()
                logging.info(f"Waiting for {MAX_WAIT_OLLAMA_RESTART_SEC} seconds after server restart.")
                time.sleep(MAX_WAIT_OLLAMA_RESTART_SEC)
            except Exception as e:
                logging.error(f"Attempt {attempt}: Error during sentiment analysis for image {image_path}: {e}")

            # Pause between API calls
            time.sleep(random.uniform(1, 3))

    logging.error(f"All {MAX_CALL_OLLAMA} attempts failed for image {image_path}. Returning neutral sentiment.")
    return "This text has a neutral sentimental"


def save_description(genre: str, film_name: str, keyframe: str, description: str) -> None:
    """Save the sentiment description to the corresponding output directory."""
    sanitized_film_name = sanitize_filename(film_name)
    output_dir = os.path.join(ROOT_OUTPUT_DIRECTORY, genre, sanitized_film_name)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{os.path.splitext(keyframe)[0]}_description.txt")
    with open(output_file, 'w') as f:
        f.write(description)

    logging.info(f"Saved description for {keyframe} at {output_file}")

def extract_scene_number(keyframe: str) -> int:
    """Extract the numeric part of the scene prefix from the keyframe filename."""
    match = re.search(r'\d+', keyframe)
    return int(match.group()) if match else float('inf')

def log_timing(message: str) -> None:
    """Log the timing information and echo to terminal with newlines for emphasis."""
    logging.info(message)
    print(f"\n\n{message}\n\n")


def process_keyframes(leaf_dirs: List[str]) -> None:
    """Process all keyframe images in the leaf directories."""
    frame_number = 1
    genre_start_time = time.time()
    for leaf_dir in sorted(leaf_dirs):
        # Assuming leaf_dir is formatted as 'genre/film_name'
        relative_path = os.path.relpath(leaf_dir, ROOT_INPUT_DIRECTORY)
        parts = relative_path.split(os.sep)
        if len(parts) < 2:
            logging.warning(f"Skipping invalid directory structure: {leaf_dir}")
            continue
        genre, film_name = parts[0], parts[1]

        film_start_time = time.time()
        keyframes = sorted(f for f in os.listdir(leaf_dir) if os.path.isfile(os.path.join(leaf_dir, f)))
        keyframes_by_scene = defaultdict(list)
        for keyframe in keyframes:
            scene_prefix = keyframe.split('_')[0]  # Assuming scene prefix is the part before the first underscore
            keyframes_by_scene[scene_prefix].append(keyframe)

        for scene_prefix in sorted(keyframes_by_scene.keys(), key=extract_scene_number):
            for keyframe in sorted(keyframes_by_scene[scene_prefix]):
                keyframe_path = os.path.join(leaf_dir, keyframe)
                output_file = os.path.join(ROOT_OUTPUT_DIRECTORY, genre, sanitize_filename(film_name), f"{os.path.splitext(keyframe)[0]}_description.txt")
                if os.path.exists(output_file):
                    logging.info(f"Skipping already processed keyframe: {keyframe_path}")
                    continue

                logging.info(f"Processing {keyframe_path}...")
                try:
                    description = get_image_sentiment_description(keyframe_path, frame_number)
                except Exception as e:
                    logging.error(f"Error processing {keyframe_path}: {e}")
                    description = "This text has a neutral sentimental"  # Create a dummy output

                save_description(genre, film_name, keyframe, description)
                frame_number += 1

        film_end_time = time.time()
        film_elapsed_time = film_end_time - film_start_time
        log_timing(f"FILM PROCESSING COMPLETED: {leaf_dir}\nStart Time: {film_start_time}\nEnd Time: {film_end_time}\nElapsed Time: {film_elapsed_time:.2f} seconds")

    genre_end_time = time.time()
    genre_elapsed_time = genre_end_time - genre_start_time
    log_timing(f"GENRE PROCESSING COMPLETED\nStart Time: {genre_start_time}\nEnd Time: {genre_end_time}\nElapsed Time: {genre_elapsed_time:.2f} seconds")



def main() -> None:
    """Main function to initiate the processing of keyframes."""
    overall_start_time = time.time()
    leaf_dirs = find_leaf_directories(ROOT_INPUT_DIRECTORY)
    while leaf_dirs:
        process_keyframes(leaf_dirs)
        leaf_dirs = find_leaf_directories(ROOT_INPUT_DIRECTORY)
        log_memory_usage()
        gc.collect()
    overall_end_time = time.time()
    overall_elapsed_time = overall_end_time - overall_start_time
    log_timing(f"TOTAL PROCESSING COMPLETED\nStart Time: {overall_start_time}\nEnd Time: {overall_end_time}\nElapsed Time: {overall_elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
