import os
import json
import pandas as pd
import logging
from textblob import TextBlob  # Ensure you have textblob installed

# Define input and output directories
input_dir = "../data/transcripts/batch"
output_dir = "../data/transcripts_sentiments/batch"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_text_sentiment_polarity(text: str) -> float:
    """
    Perform text sentiment analysis returning a polarity anywhere between -1.0 to +1.0 for most negative to most positive respectively with 0.0 for absolutely neutral
    """
    return TextBlob(text).sentiment.polarity

def process_file(file_path: str, output_dir: str):
    # Read the JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df['start'] = df['start'].astype(float)
    df['duration'] = df['duration'].astype(float)
    
    # Add new columns
    df['line_no'] = df.index + 1
    df['midpoint'] = df.apply(lambda row: row['start'] + (row['duration'] / 2), axis=1)
    df['sentiment'] = df['text'].apply(get_text_sentiment_polarity)

    print(f"\n\nSUMMARY STATISTICS:\n{df.describe()}\n")
    
    # Define the output filename
    base_name = os.path.basename(file_path)
    output_file_name = os.path.splitext(base_name)[0] + "_srt_sentiments.csv"
    output_file_path = os.path.join(output_dir, output_file_name)
    
    # Save the DataFrame to CSV
    df.to_csv(output_file_path, index=False)
    logging.info(f"Processed and saved: {output_file_path}")

def main():
    # Iterate over all JSON files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_dir, file_name)
            process_file(file_path, output_dir)

if __name__ == "__main__":
    main()
