import os
import re
import pandas as pd
import json
from youtube_transcript_api import YouTubeTranscriptApi
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths
dir_path = os.path.join("..", "data")
input_file = "dataset_film_small_details.csv"
input_file_path = os.path.join(dir_path, input_file)

def save_transcript_to_file(transcript, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=4)
    logging.info(f"Transcript saved to {file_path}")

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
            continue
