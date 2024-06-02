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
summary_file_path = os.path.join(dir_path, "summary_video_download.csv")

# Define constants
MAX_VIDEO_SIZE_MB = 500
MAX_VIDEO_SIZE_BYTES = MAX_VIDEO_SIZE_MB * 1024 * 1024
MIN_VIDEO_SIZE_MB = 10
MIN_VIDEO_SIZE_BYTES = MIN_VIDEO_SIZE_MB * 1024 * 1024
MAX_DOWNLOAD_TIMESECS = 300
MAX_RAND_WAITSECS = 5

# Initialize the DataFrame for storing the summary of downloads
download_summary_df = pd.DataFrame(columns=[
    "filename", "url", "outcome", "log_message", "language", "video_quality", "file_type", "file_size"
])

# Function to sanitize filenames
def sanitize_filename(filename):
    illegal_chars_pattern = re.compile(r'[<>:"/\\|?*.,;]')
    return illegal_chars_pattern.sub('-', filename)

# Function to check if a file with the same base name exists in the directory and meets the size criteria
def file_exists_and_valid(directory, base_name):
    sanitized_base_name = sanitize_filename(base_name)
    for file in os.listdir(directory):
        if os.path.splitext(file)[0] == sanitized_base_name:
            file_path = os.path.join(directory, file)
            if os.path.getsize(file_path) >= MIN_VIDEO_SIZE_BYTES:
                return True
    return False

# Function to download YouTube videos
def download_video(link, download_path, filename_root):
    log_messages = []
    try:
        # Instantiate YouTube object and get available streams
        youtube_object = YouTube(link)
        log_messages.append(f"Instantiated YouTube object for {link}")
        streams = youtube_object.streams.filter(progressive=True, file_extension='mp4').all()
        log_messages.append(f"Found {len(streams)} streams for {link}")
        
        # Select the largest stream <= MAX_VIDEO_SIZE_BYTES
        selected_stream = None
        for stream in sorted(streams, key=lambda s: s.filesize, reverse=True):
            if stream.filesize <= MAX_VIDEO_SIZE_BYTES:
                selected_stream = stream
                break
        
        if not selected_stream:
            message = f"No suitable stream found for {link} within the size limit of {MAX_VIDEO_SIZE_MB} MB."
            log_messages.append(message)
            logging.warning(message)
            return "failure", log_messages, None
        
        file_extension = selected_stream.mime_type.split('/')[-1]
        sanitized_filename_root = sanitize_filename(filename_root)
        filename = f"{sanitized_filename_root}.{file_extension}"
        file_path = os.path.join(download_path, filename)
        
        output_path = selected_stream.download(output_path=download_path, filename=filename)
        file_size = os.path.getsize(output_path)
        file_size_str = f"{file_size / 1024 / 1024:.2f} MB" if file_size > 1024 * 1024 else f"{file_size / 1024:.2f} KB"
        message = f"Download completed successfully for {filename}"
        log_messages.append(message)
        logging.info(message)
        
        metadata = {
            "language": "en",  # Assume English for now
            "video_quality": selected_stream.resolution,
            "file_type": file_extension,
            "file_size": file_size_str
        }
        
        return "success", log_messages, metadata
    except AgeRestrictedError as e:
        message = f"Video {link} is age restricted and can't be accessed without logging in: {e}"
        log_messages.append(message)
        logging.error(message)
        return "failure", log_messages, None
    except Exception as e:
        message = f"An error occurred while downloading {filename}: {e}"
        log_messages.append(message)
        logging.error(message)
        return "failure", log_messages, None

def download_with_timeout(link, download_path, filename_root, timeout=MAX_DOWNLOAD_TIMESECS):
    outcome = "failure"
    log_messages = []
    metadata = None

    def download_thread():
        nonlocal outcome, log_messages, metadata
        outcome, log_messages, metadata = download_video(link, download_path, filename_root)
    
    def print_elapsed_time(start_time):
        while thread.is_alive():
            elapsed_time = time.time() - start_time
            cumulative_mins = int(elapsed_time // 60)
            if elapsed_time % 60 == 0:  # Print every full minute
                print(f"      total time: {cumulative_mins} minutes")
            time.sleep(1)  # Check every second to avoid missing the exact minute
    
    start_time = time.time()
    thread = threading.Thread(target=download_thread)
    thread.start()

    timer_thread = threading.Thread(target=print_elapsed_time, args=(start_time,))
    timer_thread.start()

    thread.join(timeout)
    if thread.is_alive():
        message = f"Download timed out for video {link}"
        log_messages.append(message)
        logging.error(message)
        # Attempt to delete incomplete file if exists
        try:
            file_extension = 'mp4'  # Assuming the file extension for simplicity, adjust if needed
            sanitized_filename_root = sanitize_filename(filename_root)
            filename = f"{sanitized_filename_root}.{file_extension}"
            file_path = os.path.join(download_path, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                message = f"Deleted incomplete file: {file_path}"
                log_messages.append(message)
                logging.info(message)
        except Exception as e:
            message = f"Failed to delete incomplete file {filename_root}: {e}"
            log_messages.append(message)
            logging.error(message)
    else:
        outcome, log_messages, metadata = download_video(link, download_path, filename_root)  # Collect the return values from download_video
    
    return outcome, log_messages, metadata

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
        
        logging.info(f"\n\nProcessing film: {film_name} (URL: {film_url})")

        # Sanitize the film name by replacing illegal characters with a hyphen
        sanitized_film_name = sanitize_filename(film_name)

        # Construct the video filename root
        filename_root = f"{sanitized_film_name.replace(' ', '_')}_{film_year}"

        # Define the directory to save the video
        video_dir = os.path.join(dir_path, "videos", film_genre)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        # Check if the file exists and meets the size criteria
        if file_exists_and_valid(video_dir, filename_root):
            logging.info(f"File with base name '{filename_root}' already exists and is valid in {video_dir}, skipping download.")
            outcome, log_messages, metadata = "skipped", [f"File with base name '{filename_root}' already exists and is valid in {video_dir}, skipping download."], None
        else:
            # Add a random wait time between 1 and MAX_RAND_WAITSECS seconds
            wait_time = random.randint(1, MAX_RAND_WAITSECS)
            logging.info(f"Waiting for {wait_time} seconds before starting download.\n")
            time.sleep(wait_time)
            
            # Download the video with a timeout
            outcome, log_messages, metadata = download_with_timeout(film_url, video_dir, filename_root)

        # Prepare the row to be added to the DataFrame
        row_data = {
            "filename": filename_root,
            "url": film_url,
            "outcome": outcome,
            "log_message": " | ".join(log_messages)
        }
        
        if outcome == "success" and metadata:
            row_data.update(metadata)

        # Append the download details to the summary DataFrame using pd.concat
        download_summary_df = pd.concat([download_summary_df, pd.DataFrame([row_data])], ignore_index=True)

        # Write the updated summary DataFrame to CSV
        download_summary_df.to_csv(summary_file_path, index=False)

    logging.info("All videos processed.")
