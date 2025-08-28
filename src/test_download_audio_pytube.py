import os
from pytube import YouTube

# Ask user for the YouTube video URL
url = "https://www.youtube.com/watch?v=CFhr-5f3Ufk&t=1679s" # input("Enter the YouTube video URL: ")
url_title = "music_classical_for_programming.mp3"

# Create a YouTube object from the URL
yt = YouTube(url)

# Get the audio stream
audio_stream = yt.streams.filter(only_audio=True).first()

# Download the audio stream
output_path = os.path.join("..","data")

audio_stream.download(output_path=output_path, filename=url_title)

print(f"Audio downloaded to {output_path}/{url_title}")

