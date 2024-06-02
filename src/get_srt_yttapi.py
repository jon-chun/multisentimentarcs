import os
import re
from youtube_transcript_api import YouTubeTranscriptApi

#srt = YouTubeTranscriptApi.get_transcript(video_id)

film_name="Sabrina"
film_year=1954
film_genre="romance"
film_url="www.youtube.com/watch?v=xYgoNiSo-kY"

film_name="The Last Time I Saw Paris"
film_year=1954
film_genre="romance"
film_url="www.youtube.com/watch?v=UdPV5tSmO1M"

film_name="Rebecca"
film_year=1940
film_genre="psychological-thriller"
film_url="www.youtube.com/watch?v=m1uvgx3NUR0"

film_name="The Lost Moment"
film_year=1947
film_genre="psychological-thriller"
film_url="www.youtube.com/watch?v=YP2gId2xO-Q"

https://www.youtube.com/watch?v=UdPV5tSmO1M


film_id = re.search(r"v=([a-zA-Z0-9_-]+)", "www.youtube.com/watch?v=xYgoNiSo-kY").group(1)




print(f"Fetching film id={film_id}: {film_name}")

# videoListName = "youtubeVideoIDlist.txt"
transcript_dir, _ = os.path.join("..","data","transcripts","film_genre","film_name")
print(f"  saving to directory: {transcript_dir}")

# with open(videoListName) as f:
#     video_ids = f.read().splitlines()

film_id_list = [film_id]

transcript_list, unretrievable_videos = YouTubeTranscriptApi.get_transcripts(video_ids, continue_after_error=True)

for video_id in video_ids:

    if video_id in transcript_list.keys():

        print("\nvideo_id = ", video_id)
        #print(transcript)

        srt = transcript_list.get(video_id)

        text_list = []
        for i in srt:
            text_list.append(i['text'])

        text = ' '.join(text_list)
        print(text)