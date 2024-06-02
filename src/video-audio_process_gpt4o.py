# https://www.youtube.com/watch?v=Jqx5TcF8_PU

from openai import OpenAI
import os
import cv2
from moviepy.editor import VideoFileClip
import time
import base64

MODEL_NAME = "gpt-4o"
os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=os.environ.get(("OPENAI_API_KEY")))

VIDEO_DIR = os.path.join("..","data","videos","test.mp4")

def process_video(video_path, sec_per_frame=2):
    base64frames = []
    base_video_path, _, = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * sec_per_frame)
    curr_frame = 0

    # Loop through the video and extract frames at specificed frame rate
    while curr_frame < total_frames -1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    # Extract audio from video
    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.audio.close()
    clip.close()

    print(f"Extracted {len(base64frames)} frames")
    print(f"Extracted audio to {audio_path}")
    return base64frames, audio_path

# Extract 1 frame per second. You can adjust the 'seconds_per_frame' to change sampling rate
base64frames, audio_path = process_video(VIDEO_DIR, seconds_per_frame=2)

transcription = client.audio.transcriptions.create(
    model="whisiper-1",
    file=open(audio_path, "rb")
)

# Generate a summary with visual and audio
response = client.chat.completions.create(
    model=MODEL_NAME,
    message=[
        {"role": "system", "content":"""You are generating a video summary. Create a summary of the provided video and it's transcription"""}
        {"role": "user", "content": [
            "These are the frames from the video.",
            *map(lambda x: {"type": "image_url",
                            "image_url": {"url": f'data:image/jpg;base64,{x}',"detail","low"}, base64frames),
            {"type": "text", "text": f"The audio transcription is: {transcription.text}"}
            ],
        }
    ],
    temperature=0,
)
print(response.choices[0].message.content)

