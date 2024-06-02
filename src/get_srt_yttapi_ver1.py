import os
import re
from youtube_transcript_api import YouTubeTranscriptApi

# Define film details
films = [
    {"name": "Sabrina", "year": 1954, "genre": "romance", "url": "www.youtube.com/watch?v=xYgoNiSo-kY"},
    {"name": "The Last Time I Saw Paris", "year": 1954, "genre": "romance", "url": "www.youtube.com/watch?v=UdPV5tSmO1M"},
    {"name": "Rebecca", "year": 1940, "genre": "psychological-thriller", "url": "www.youtube.com/watch?v=m1uvgx3NUR0"},
    {"name": "The Lost Moment", "year": 1947, "genre": "psychological-thriller", "url": "www.youtube.com/watch?v=YP2gId2xO-Q"}
]

# Function to extract YouTube video ID
def extract_video_id(url):
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

# Function to save transcripts
def save_transcript(transcript, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    transcript_path = os.path.join(directory, "transcript.txt")
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(transcript)

# Main logic
for film in films:
    film_id = extract_video_id(film["url"])
    if film_id:
        print(f"Fetching transcript for film: {film['name']} (ID: {film_id})")

        # Define the directory to save the transcript
        transcript_dir = os.path.join("..", "data", "transcripts", film["genre"], film["name"].replace(" ", "_"))
        print(f"Saving transcript to: {transcript_dir}")

        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(film_id)
            text_list = [item['text'] for item in transcript_list]
            text = ' '.join(text_list)
            save_transcript(text, transcript_dir)
            print(f"Transcript saved for film: {film['name']}")

        except Exception as e:
            print(f"Failed to fetch transcript for {film['name']} (ID: {film_id}): {e}")

