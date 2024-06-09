import os
import re
import pandas as pd
import json
from youtube_transcript_api import YouTubeTranscriptApi
import logging
from datetime import datetime

# Generate a log file name with a datetime stamp
current_time = datetime.now().strftime("%Y%m%d_%H%M")
log_file = os.path.join("..", "data", f"transcript_errors_{current_time}.log")

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
output_failed_log = os.path.join(dir_path, "get_ssrt_yttapi_logs.txt")

def save_transcript_to_file(transcript, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=4)
    logging.info(f"Transcript saved to {file_path}")

# Initialize a list to log failed requests
failed_logs = []

# Main logic
if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv(input_file_path)

    # Ensure the 'video_id' column exists
    if 'video_id' not in df.columns:
        df['video_id'] = df['url'].apply(lambda x: re.search(r"v=([a-zA-Z0-9_-]+)", x).group(1) if re.search(r"v=([a-zA-Z0-9_-]+)", x) else None)

    # Iterate over each film and process the transcript
    for index, row in df.iterrows():
        film_name = row['name']
        film_year = row['year']
        film_genre = row['genre']
        film_url = row['url']
        film_id = row['video_id']
        
        logging.info(f"Processing film: {film_name} (ID: {film_id})")

        # Define the directory and file name to save the transcript
        transcript_dir = os.path.join(dir_path, "transcripts", film_genre)
        file_out = f"{film_name.replace(' ', '_')}_{film_year}.json"
        transcript_path = os.path.join(transcript_dir, file_out)

        # Check if the transcript already exists
        if os.path.exists(transcript_path):
            logging.info(f"Transcript already exists for {film_name} (ID: {film_id}), skipping.")
            continue

        try:
            # Fetch the transcript
            transcript = YouTubeTranscriptApi.get_transcript(film_id)
            # Save the transcript to file
            save_transcript_to_file(transcript, transcript_dir, file_out)
        
        except Exception as e:
            logging.error(f"Failed to fetch transcript for {film_name} (ID: {film_id}): {e}")
            # Add the failed request to the list
            failed_logs.append({
                "name": film_name,
                "year": film_year,
                "genre": film_genre,
                "url": film_url,
                "video_id": film_id,
                "error_message": str(e)
            })
            continue

    # Convert the list of failed logs to a DataFrame and write it to a file
    if failed_logs:
        failed_logs_df = pd.DataFrame(failed_logs)
        failed_logs_df.to_csv(output_failed_log, index=False)
        logging.info(f"Failed transcript requests logged to {output_failed_log}")
