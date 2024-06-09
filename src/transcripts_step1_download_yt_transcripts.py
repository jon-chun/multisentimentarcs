import os
import re
import pandas as pd
import json
from youtube_transcript_api import YouTubeTranscriptApi
import logging
from datetime import datetime

def configure_logging(log_directory):
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = os.path.join(log_directory, f"transcript_errors_{current_time}.log")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])

def save_transcript(transcript, directory, filename):
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(transcript, file, ensure_ascii=False, indent=4)
    logging.info(f"Transcript saved to {file_path}")

def extract_video_id(url):
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

def process_transcripts(dataframe, data_directory):
    failed_logs = []
    for _, row in dataframe.iterrows():
        film_name = row['name']
        film_year = row['year']
        film_genre = row['genre']
        film_url = row['url']
        film_id = row['video_id']

        logging.info(f"Processing film: {film_name} (ID: {film_id})")

        transcript_directory = os.path.join(data_directory, "transcripts", film_genre)
        file_output = f"{film_name.replace(' ', '_')}_{film_year}.json"
        transcript_path = os.path.join(transcript_directory, file_output)

        if os.path.exists(transcript_path):
            logging.info(f"Transcript already exists for {film_name} (ID: {film_id}), skipping.")
            continue

        try:
            transcript = YouTubeTranscriptApi.get_transcript(film_id)
            save_transcript(transcript, transcript_directory, file_output)
        
        except Exception as error:
            logging.error(f"Failed to fetch transcript for {film_name} (ID: {film_id}): {error}")
            failed_logs.append({
                "name": film_name,
                "year": film_year,
                "genre": film_genre,
                "url": film_url,
                "video_id": film_id,
                "error_message": str(error)
            })

    return failed_logs

def main():
    data_directory = os.path.join("..", "data")
    input_file_path = os.path.join(data_directory, "dataset_yt_videos_details.csv")
    output_failed_log = os.path.join(data_directory, "transcripts_step1_get_transcript_logs.txt")

    configure_logging(data_directory)

    df = pd.read_csv(input_file_path)

    if 'video_id' not in df.columns:
        df['video_id'] = df['url'].apply(extract_video_id)

    failed_logs = process_transcripts(df, data_directory)

    if failed_logs:
        failed_logs_df = pd.DataFrame(failed_logs)
        failed_logs_df.to_csv(output_failed_log, index=False)
        logging.info(f"Failed transcript requests logged to {output_failed_log}")

if __name__ == "__main__":
    main()
