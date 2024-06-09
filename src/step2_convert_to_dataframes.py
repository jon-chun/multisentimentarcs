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

# Constants
MAX_VIDEO_SIZE_MB = 500
MAX_VIDEO_SIZE_BYTES = MAX_VIDEO_SIZE_MB * 1024 * 1024
MIN_VIDEO_SIZE_MB = 10
MIN_VIDEO_SIZE_BYTES = MIN_VIDEO_SIZE_MB * 1024 * 1024
MAX_DOWNLOAD_TIME_SECS = 300
MAX_RAND_WAIT_SECS = 5
KEEP_OLD = True

# Paths
DATA_DIR = os.path.join("..", "data")
INPUT_FILE = os.path.join(DATA_DIR, "dataset_yt_videos_details.csv")
SUMMARY_FILE = os.path.join(DATA_DIR, "dataset_yt_videos_summary.csv")

# Logging setup
current_time = datetime.now().strftime("%Y%m%d_%H%M")
log_file = os.path.join(DATA_DIR, f"transcripts_step2_convert_to_dataframe_errors_{current_time}.log")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# DataFrame for summary
download_summary_df = pd.DataFrame(columns=[
    "filename", "url", "outcome", "log_message", "language", "video_quality", "file_type", "file_size"
])

def sanitize_filename(filename):
    """Sanitize filename by replacing illegal characters."""
    return re.sub(r'[<>:"/\\|?*.,;]', '-', filename)

def file_exists_and_valid(directory, base_name):
    """Check if a file exists and meets the size criteria."""
    sanitized_base_name = sanitize_filename(base_name)
    for file in os.listdir(directory):
        if os.path.splitext(file)[0] == sanitized_base_name:
            file_path = os.path.join(directory, file)
            if os.path.getsize(file_path) >= MIN_VIDEO_SIZE_BYTES:
                return True
    return False

def download_video(link, download_path, filename_root):
    """Download YouTube video and return outcome, log messages, and metadata."""
    log_messages = []
    try:
        youtube_object = YouTube(link)
        log_messages.append(f"Instantiated YouTube object for {link}")
        streams = list(youtube_object.streams.filter(progressive=True, file_extension='mp4'))
        log_messages.append(f"Found {len(streams)} streams for {link}")
        
        selected_stream = next((stream for stream in sorted(streams, key=lambda s: s.filesize, reverse=True)
                               if stream.filesize <= MAX_VIDEO_SIZE_BYTES), None)
        
        if not selected_stream:
            message = f"No suitable stream found for {link} within the size limit of {MAX_VIDEO_SIZE_MB} MB."
            log_messages.append(message)
            logging.warning(message)
            return "failure", log_messages, None
        
        filename = f"{sanitize_filename(filename_root)}.{selected_stream.mime_type.split('/')[-1]}"
        output_path = selected_stream.download(output_path=download_path, filename=filename)
        file_size = os.path.getsize(output_path)
        file_size_str = f"{file_size / 1024 / 1024:.2f} MB" if file_size > 1024 * 1024 else f"{file_size / 1024:.2f} KB"
        message = f"Download completed successfully for {filename}"
        log_messages.append(message)
        logging.info(message)
        
        metadata = {
            "language": "en",
            "video_quality": selected_stream.resolution,
            "file_type": selected_stream.mime_type.split('/')[-1],
            "file_size": file_size_str
        }
        return "success", log_messages, metadata
    except AgeRestrictedError as e:
        message = f"Video {link} is age restricted and can't be accessed without logging in: {e}"
        log_messages.append(message)
        logging.error(message)
        return "failure", log_messages, None
    except Exception as e:
        message = f"An error occurred while downloading {filename_root}: {e}"
        log_messages.append(message)
        logging.error(message)
        return "failure", log_messages, None

def download_with_timeout(link, download_path, filename_root, timeout=MAX_DOWNLOAD_TIME_SECS):
    """Download video with a timeout."""
    outcome = "failure"
    log_messages = []
    metadata = None

    def download_thread():
        nonlocal outcome, log_messages, metadata
        outcome, log_messages, metadata = download_video(link, download_path, filename_root)
    
    start_time = time.time()
    thread = threading.Thread(target=download_thread)
    thread.start()

    while thread.is_alive() and time.time() - start_time < timeout:
        time.sleep(1)

    if thread.is_alive():
        log_messages.append(f"Download timed out for video {link}")
        logging.error(log_messages[-1])
        if not KEEP_OLD:
            try:
                file_path = os.path.join(download_path, f"{sanitize_filename(filename_root)}.mp4")
                if os.path.exists(file_path) and os.path.getsize(file_path) < MIN_VIDEO_SIZE_BYTES:
                    os.remove(file_path)
                    log_messages.append(f"Deleted incomplete file: {file_path}")
                    logging.info(log_messages[-1])
            except Exception as e:
                log_messages.append(f"Failed to delete incomplete file {filename_root}: {e}")
                logging.error(log_messages[-1])
    return outcome, log_messages, metadata

def main():
    """Main function to process videos."""
    df = pd.read_csv(INPUT_FILE)

    for _, row in df.iterrows():
        film_name, film_year, film_genre, film_url = row['name'], row['year'], row['genre'], row['url']
        logging.info(f"Processing film: {film_name} (URL: {film_url})")

        filename_root = f"{sanitize_filename(film_name).replace(' ', '_')}_{film_year}"
        video_dir = os.path.join(DATA_DIR, "videos", film_genre)
        os.makedirs(video_dir, exist_ok=True)

        if file_exists_and_valid(video_dir, filename_root):
            log_messages = [f"File with base name '{filename_root}' already exists and is valid in {video_dir}, skipping download."]
            logging.info(log_messages[0])
            outcome, metadata = "skipped", None
        else:
            time.sleep(random.randint(1, MAX_RAND_WAIT_SECS))
            outcome, log_messages, metadata = download_with_timeout(film_url, video_dir, filename_root)

        row_data = {"filename": filename_root, "url": film_url, "outcome": outcome, "log_message": " | ".join(log_messages)}
        if metadata:
            row_data.update(metadata)
        download_summary_df.loc[len(download_summary_df)] = row_data
        download_summary_df.to_csv(SUMMARY_FILE, index=False)

    logging.info("All videos processed.")

if __name__ == "__main__":
    main()
