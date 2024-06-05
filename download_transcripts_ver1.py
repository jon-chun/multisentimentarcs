"""
This script processes YouTube transcripts for a set of films listed in a CSV file.
It logs the progress and any errors encountered during the process.

Inputs:
    - dataset_film_small_details.csv: A CSV file containing film details, including columns for 'name', 'year', 'genre', 'url', and optionally 'video_id'.

Outputs:
    - Transcript files: JSON files containing transcripts for each film, saved in a structured directory based on film genres, or in the batch directory if processing from TARGET_MOVIE_DICT.
    - Log file: A log file named with a datetime stamp, recording the process and any errors.
    - Failed logs file: A CSV file logging failed attempts to fetch transcripts.

Functions:
    1. save_transcript_to_file(transcript, directory, filename):
        - Saves a transcript to a JSON file in the specified directory.
        - Inputs: 
            - transcript: dict, the transcript data.
            - directory: str, the directory path to save the file.
            - filename: str, the name of the file.
        - Outputs: None

    2. extract_video_id(url):
        - Extracts the video ID from a YouTube URL.
        - Inputs:
            - url: str, the YouTube video URL.
        - Outputs:
            - video_id: str or None, the extracted video ID or None if not found.

    3. process_transcripts(df):
        - Processes transcripts for each film in the DataFrame.
        - Inputs:
            - df: pandas.DataFrame, the DataFrame containing film details.
        - Outputs:
            - failed_logs: list of dicts, logs of failed attempts to fetch transcripts.
        
    4. process_target_transcripts(target_dict):
        - Processes transcripts for films specified in the target dictionary.
        - Inputs:
            - target_dict: dict, the dictionary containing film names and video IDs.
        - Outputs:
            - failed_logs: list of dicts, logs of failed attempts to fetch transcripts.
"""

import os
import re
import pandas as pd
import json
from youtube_transcript_api import YouTubeTranscriptApi
import logging
from datetime import datetime

# Global target movie dictionary
TARGET_MOVIE_DICT = {
    "The_Mob_1951": "45Fq8FHWjgE",
    "Royal_Wedding_1951": "24A5h-ZVxQ8",
    "Rawhide_1951": "rEdH9yWhFfg"
}

# Generate a log file name with a datetime stamp
current_time = datetime.now().strftime("%Y%m%d_%H%M")
base_dir = os.path.dirname(os.path.abspath(__file__))
dir_path = os.path.join(base_dir, "data")
log_file = os.path.join(dir_path, f"transcript_errors_{current_time}.log")

# Set up logging to both terminal and the generated log file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])

# Define file paths
input_file = "dataset_film_small_details.csv"
input_file_path = os.path.join(dir_path, input_file)
output_failed_log = os.path.join(dir_path, "get_ssrt_yttapi_logs.csv")

def save_transcript_to_file(transcript, directory, filename):
    """Save the transcript to a JSON file in the specified directory."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=4)
    logging.info(f"Transcript saved to {file_path}")

def extract_video_id(url):
    """Extract the video ID from a YouTube URL."""
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

def process_transcripts(df):
    """Process transcripts for each film in the DataFrame."""
    failed_logs = []

    for index, row in df.iterrows():
        film_name = row['name']
        film_year = row['year']
        film_genre = row['genre']
        film_url = row['url']
        film_id = row['video_id']
        
        logging.info(f"Processing film: {film_name} (ID: {film_id})")

        transcript_dir = os.path.join(dir_path, "transcripts", film_genre)
        file_out = f"{film_name.replace(' ', '_')}_{film_year}.json"
        transcript_path = os.path.join(transcript_dir, file_out)

        if os.path.exists(transcript_path):
            logging.info(f"Transcript already exists for {film_name} (ID: {film_id}), skipping.")
            continue

        try:
            transcript = YouTubeTranscriptApi.get_transcript(film_id)
            save_transcript_to_file(transcript, transcript_dir, file_out)
        
        except Exception as e:
            logging.error(f"Failed to fetch transcript for {film_name} (ID: {film_id}): {e}")
            failed_logs.append({
                "name": film_name,
                "year": film_year,
                "genre": film_genre,
                "url": film_url,
                "video_id": film_id,
                "error_message": str(e)
            })

    return failed_logs

def process_target_transcripts(target_dict):
    """Process transcripts for films specified in the target dictionary."""
    failed_logs = []

    batch_dir = os.path.join(dir_path, "transcripts", "batch")
    
    for film_name, film_id in target_dict.items():
        logging.info(f"Processing film: {film_name} (ID: {film_id})")
        
        file_out = f"{film_name.lower()}_srt.json"
        transcript_path = os.path.join(batch_dir, file_out)

        if os.path.exists(transcript_path):
            logging.info(f"Transcript already exists for {film_name} (ID: {film_id}), skipping.")
            continue

        try:
            transcript = YouTubeTranscriptApi.get_transcript(film_id)
            save_transcript_to_file(transcript, batch_dir, file_out)
        
        except Exception as e:
            logging.error(f"Failed to fetch transcript for {film_name} (ID: {film_id}): {e}")
            failed_logs.append({
                "name": film_name,
                "year": "Unknown_Year",
                "genre": "Unknown_Genre",
                "url": None,
                "video_id": film_id,
                "error_message": str(e)
            })

    return failed_logs

if __name__ == "__main__":
    if len(TARGET_MOVIE_DICT) == 0:
        df = pd.read_csv(input_file_path)

        if 'video_id' not in df.columns:
            df['video_id'] = df['url'].apply(extract_video_id)
            if df['video_id'].isnull().any():
                logging.warning("Some URLs did not contain a valid video_id")

        failed_logs = process_transcripts(df)
    else:
        failed_logs = process_target_transcripts(TARGET_MOVIE_DICT)

    if failed_logs:
        failed_logs_df = pd.DataFrame(failed_logs)
        failed_logs_df.to_csv(output_failed_log, index=False)
        logging.info(f"Failed transcript requests logged to {output_failed_log}")
