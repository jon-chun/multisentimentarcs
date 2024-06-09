import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

def process_transcripts(input_root_directory):
    # Create a SentimentIntensityAnalyzer object
    vader_analyzer = SentimentIntensityAnalyzer()

    # Iterate over the genres and files in the input root directory
    for genre in os.listdir(input_root_directory):
        genre_dir = os.path.join(input_root_directory, genre)
        if os.path.isdir(genre_dir):
            for file in os.listdir(genre_dir):
                if file.endswith("_clean_transcript.csv"):
                    file_name, file_year = file.rsplit("_", 2)[:2]
                    file_path = os.path.join(genre_dir, file)
                    
                    # Step 2.a: Read the content of the file into a DataFrame
                    df = pd.read_csv(file_path, names=["text", "start", "duration", "time_midpoint"])
                    
                    # Step 2.b: Create 'vader' and 'textblob' columns for sentiment analysis
                    df['vader'] = df['text'].apply(lambda text: vader_analyzer.polarity_scores(text)['compound'])
                    df['textblob'] = df['text'].apply(lambda text: TextBlob(text).sentiment.polarity)
                    
                    # Step 2.c: Create the output subdirectory if it doesn't exist
                    output_subdir = f"../data/transcripts_sentiments/{genre}/{file_name}"
                    os.makedirs(output_subdir, exist_ok=True)
                    
                    # Step 2.d: Write the DataFrame to the output file
                    output_file = f"{output_subdir}/{file_name}_{file_year}_sentiment_transcript.csv"
                    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    # Set the input root directory
    INPUT_ROOT_DIRECTORY = "../data/transcripts_combined"
    
    # Call the process_transcripts function
    process_transcripts(INPUT_ROOT_DIRECTORY)