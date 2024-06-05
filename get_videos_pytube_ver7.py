# NOTE: add parallel processes with pauses between each fetch/reponse pair

import os
import re
import pandas as pd
from pytube import YouTube
import logging
from datetime import datetime
from pytube.exceptions import AgeRestrictedError
import threading
import time
import random

# Generate a log file name with a datetime stamp
current_time = datetime.now().strftime("%Y%m%d_%H%M")
log_file = os.path.join("..", "data", f"video_download_errors_{current_time}.log")

# Set up logging to both terminal and the generated log file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])

# Define file paths
dir_path = os.path.join("..", "data")
input_file = "dataset_film_small_details.csv"
input_file_path = os.path.join(dir_path, input_file)

# Define constants
MAX_VIDEO_SIZE_MB = 500
MAX_VIDEO_SIZE_BYTES = MAX_VIDEO_SIZE_MB * 1024 * 1024
MAX_DOWNLOAD_TIMESECS = 300
MAX_RAND_WAITSECS = 5

# Function to check if a file with the same base name exists in the directory
def file_base_exists(directory, base_name):
    for file in os.listdir(directory):
        if os.path.splitext(file)[0] == base_name:
            return True
    return False

# Function to download YouTube videos
def download_video(link, download_path, filename_root):
    # Check if the file with the same base name already exists
    if file_base_exists(download_path, filename_root):
        logging.info(f"File with base name '{filename_root}' already exists in {download_path}, skipping download.")
        return None
    
    try:
        # Instantiate YouTube object and get available streams
        youtube_object = YouTube(link)
        logging.info(f"Instantiated YouTube object for {link}")
        streams = youtube_object.streams.filter(progressive=True, file_extension='mp4').all()
        logging.info(f"Found {len(streams)} streams for {link}")
        
        # Select the largest stream <= MAX_VIDEO_SIZE_BYTES
        selected_stream = None
        for stream in sorted(streams, key=lambda s: s.filesize, reverse=True):
            if stream.filesize <= MAX_VIDEO_SIZE_BYTES:
                selected_stream = stream
                break
        
        if not selected_stream:
            logging.warning(f"No suitable stream found for {link} within the size limit of {MAX_VIDEO_SIZE_MB} MB.")
            return None
        
        file_extension = selected_stream.mime_type.split('/')[-1]
        filename = f"{filename_root}.{file_extension}"
        file_path = os.path.join(download_path, filename)
        
        output_path = selected_stream.download(output_path=download_path, filename=filename)
        logging.info(f"Download completed successfully for {filename}")
        return output_path
    except AgeRestrictedError as e:
        logging.error(f"Video {link} is age restricted and can't be accessed without logging in: {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while downloading {filename}: {e}")
        return None

def download_with_timeout(link, download_path, filename_root, timeout=MAX_DOWNLOAD_TIMESECS):
    # Create a thread for the download_video function
    thread = threading.Thread(target=download_video, args=(link, download_path, filename_root))
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        logging.error(f"Download timed out for video {link}")
        return None

# Main logic
if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv(input_file_path)

    # Define a regex pattern to match illegal characters
    illegal_chars_pattern = re.compile(r'[<>:"/\\|?*.,;]')

    # Iterate over each film and download the video
    for index, row in df.iterrows():
        film_name = row['name']
        film_year = row['year']
        film_genre = row['genre']
        film_url = row['url']
        
        logging.info(f"Processing film: {film_name} (URL: {film_url})")

        # Sanitize the film name by replacing illegal characters with a hyphen
        sanitized_film_name = illegal_chars_pattern.sub('-', film_name)

        # Construct the video filename root
        filename_root = f"{sanitized_film_name.replace(' ', '_')}_{film_year}"

        # Define the directory to save the video
        video_dir = os.path.join(dir_path, "videos", film_genre)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        # Download the video with a timeout
        download_with_timeout(film_url, video_dir, filename_root)

        # Add a random wait time between 1 and MAX_RAND_WAITSECS seconds
        wait_time = random.randint(1, MAX_RAND_WAITSECS)
        logging.info(f"Waiting for {wait_time} seconds before next download.")
        time.sleep(wait_time)

    logging.info("All videos processed.")