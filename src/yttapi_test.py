from youtube_transcript_api import YouTubeTranscriptApi

# Sabrina 1954
# https://www.youtube.com/watch?v=bnU2dWMGpMA

    transcript_str = YouTubeTranscriptApi.get_transcript("bnU2dWMGpMA")

    print(f"type(transcript_str): {type(transcript_str)}")

    print(f"\n\n{transcript_str}\n\nLength: {len(transcript_str)}")