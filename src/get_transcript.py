import os
from youtube_transcript_api import YouTubeTranscriptApi

def get_yt_transcript(video_id, file_out, dir_out):
    try:
        # Fetch the transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Convert the transcript list of dictionaries to a string
        transcript_str = '\n'.join([entry['text'] for entry in transcript])
        
        # Create the output directory if it does not exist
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        
        # Define the output file path
        file_path = os.path.join(dir_out, file_out)
        
        # Write the transcript to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(transcript_str)
        
        # Calculate the number of lines and the total character length
        lines = len(transcript_str.split('\n'))
        char_len = len(transcript_str)
        
        # Return the success flag, lines, and char_len
        return (True, lines, char_len)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return (False, 0, 0)

def main():
    video_id = "bnU2dWMGpMA"  # Sabrina (1954) video ID
    file_out = "sabrina_1954.txt"  # Example output file name
    dir_out = "../data/transcripts"  # Example output directory
    
    success_flag, lines, char_len = get_yt_transcript(video_id, file_out, dir_out)
    
    print(f"Success: {success_flag}")
    print(f"Number of lines: {lines}")
    print(f"Total characters: {char_len}")

if __name__ == "__main__":
    main()
