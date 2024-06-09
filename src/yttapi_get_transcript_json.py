# your_custom_script.py
import os
import re

# Define directory and file paths
dir_path = os.path.join("..", "data")

input_file = "dataset_yt_videos_ids.txt"
output_file = "dataset_yt_vid_small.txt"

input_file_path = os.path.join(dir_path, input_file)
output_file_path = os.path.join(dir_path, output_file)

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter

video_id = "xYgoNiSo-kY"

# Must be a single transcript.
transcript = YouTubeTranscriptApi.get_transcript(video_id)

formatter = JSONFormatter()

# .format_transcript(transcript) turns the transcript into a JSON string.
json_formatted = formatter.format_transcript(transcript)


# Now we can write it out to a file.
with open('your_filename.json', 'w', encoding='utf-8') as json_file:
    json_file.write(json_formatted)