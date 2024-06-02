# From May 30, 2023: 
# https://stackoverflow.com/questions/76357844/how-to-extract-subtitles-from-youtube-videos-in-varied-languages
# Run May 31, 2024

from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi

# Define the video URL or ID of the YouTube video you want to extract text from
video_url = 'https://www.youtube.com/watch?v=xYgoNiSo-kY'

# Download the video using pytube
youtube = YouTube(video_url)
video = youtube.streams.get_highest_resolution()
video.download()

# Get the downloaded video file path
video_path = video.default_filename

# Get the video ID from the URL
video_id = video_url.split('v=')[-1]

# Get the transcript for the specified video ID
transcript = YouTubeTranscriptApi.get_transcript(video_id)

# Extract the text from the transcript
captions_text = ''
for segment in transcript:
    caption = segment['text']
    captions_text += caption + ' '

# Print the extracted text
print(captions_text)