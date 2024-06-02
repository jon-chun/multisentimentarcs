import os
import re
import pandas as pd
from pytube import YouTube
import logging
from datetime import datetime
from pytube.exceptions import AgeRestrictedError

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
        # Instantiate YouTube object and get stream for downloading
        youtube_object = YouTube(link)
        # Necessary Hack?
        # youtube_object.bypass_age_gate()  # Necessary hack?
        stream = youtube_object.streams.get_highest_resolution()
        file_extension = stream.mime_type.split('/')[-1]
        filename = f"{filename_root}.{file_extension}"
        file_path = os.path.join(download_path, filename)
        
        output_path = stream.download(output_path=download_path, filename=filename)
        logging.info(f"Download completed successfully for {filename}")
        return output_path
    except AgeRestrictedError as e:
        logging.error(f"Video {link} is age restricted and can't be accessed without logging in: {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while downloading {filename}: {e}")
        return None

# Main logic
if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv(input_file_path)

    # Iterate over each film and download the video
    for index, row in df.iterrows():
        film_name = row['name']
        film_year = row['year']
        film_genre = row['genre']
        film_url = row['url']
        
        logging.info(f"Processing film: {film_name} (URL: {film_url})")

        # Construct the video filename root
        filename_root = f"{film_name.replace(' ', '_')}_{film_year}"

        # Define the directory to save the video
        video_dir = os.path.join(dir_path, "videos", film_genre)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        # Download the video
        download_video(film_url, video_dir, filename_root)

    logging.info("All videos processed.")
