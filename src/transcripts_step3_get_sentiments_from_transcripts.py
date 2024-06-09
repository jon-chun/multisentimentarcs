import os
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from tqdm import tqdm
# Assume that 'ollama' is the client module for the specialized LLM model
import ollama

# Constants
MAX_CALL_OLLAMA = 3
MAX_WORKERS = 10  # Adjust based on the optimal number of concurrent requests supported by the API

# Initialize sentiment analyzers
vader_analyzer = SentimentIntensityAnalyzer()

# Input and output root directories
INPUT_ROOT_DIRECTORY = "../data/transcripts_combined"
OUTPUT_ROOT_DIRECTORY = "../data/transcripts_sentiments"

# Setup logging
logging.basicConfig(level=logging.INFO)

def analyze_sentiments(text):
    # VADER sentiment analysis
    vader_score = vader_analyzer.polarity_scores(text)['compound']
    
    # TextBlob sentiment analysis
    textblob_score = TextBlob(text).sentiment.polarity
    
    return vader_score, textblob_score

def get_sentiment_ollama(text: str) -> float:
    """
    Perform sentiment analysis on a text string using a specialized LLM model.

    Parameters:
        text (str): The text string for sentiment analysis.

    Returns:
        float: The sentiment value as a float between -1.0 and 1.0, or 0.0 if unsuccessful.
    """
    for attempt in range(1, MAX_CALL_OLLAMA + 1):
        try:
            logging.info(f"Attempt {attempt}: Sending text to Ollama for sentiment analysis")

            res = ollama.chat(
                model="llama3sentiment",
                messages=[
                    {
                        'role': 'user',
                        'content': f'Give the sentiment polarity float value for: {text}'
                    }
                ],
                options={
                    'temperature': 0.1, 
                    'top_p': 0.5
                }
            )

            if 'message' in res and 'content' in res['message']:
                text_sentiment_float_str = res['message']['content'].strip()
                logging.info(f'Text sentiment float string: {text_sentiment_float_str}')
                try:
                    text_sentiment_float = float(text_sentiment_float_str)
                    logging.info("Received sentiment analysis response and successfully converted to float")
                    return text_sentiment_float
                except ValueError:
                    logging.warning(f"Attempt {attempt}: Could not convert response to float: {text_sentiment_float_str}")
            else:
                logging.error(f"Attempt {attempt}: Unexpected API response format: {res}")

        except Exception as e:
            logging.error(f"Attempt {attempt}: Error during sentiment analysis for text: {e}")

    logging.error(f"All {MAX_CALL_OLLAMA} attempts failed for text: {text}. Returning 0.0")
    return 0.0

def process_file(file_path, genre, file_name, file_year):
    # Output file path
    output_subdir = os.path.join(OUTPUT_ROOT_DIRECTORY, genre, f"{file_name}_{file_year}")
    os.makedirs(output_subdir, exist_ok=True)
    output_file_path = os.path.join(output_subdir, f"{file_name}_{file_year}_sentiment_transcript.csv")
    
    # Check if output file already exists
    if os.path.exists(output_file_path):
        logging.info(f"Output file {output_file_path} already exists. Skipping...")
        return
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Add 'vader' and 'textblob' columns
    df['vader'], df['textblob'] = zip(*df['text'].apply(lambda text: analyze_sentiments(text)))

    # Create a thread pool for concurrent requests
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Use tqdm to monitor progress
        future_to_text = {executor.submit(get_sentiment_ollama, text): text for text in df['text']}
        df['llama3'] = [future.result() for future in tqdm(as_completed(future_to_text), total=len(future_to_text), desc="Ollama Sentiment Analysis")]
    
    # Write the DataFrame to the output path
    df.to_csv(output_file_path, index=False)

def crawl_and_process(input_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith("_clean_transcript.csv"):
                print(f"\n\nPROCESSING: {file}")
                # Extract the relative path from the input directory to the current file
                relative_path = os.path.relpath(root, INPUT_ROOT_DIRECTORY)
                # The genre is the first part of this relative path
                genre = relative_path.split(os.sep)[0]
                
                # Extract file_name and file_year from the file name
                file_name, file_year = file.rsplit('_', 2)[0], file.rsplit('_', 2)[1]
                
                # Full file path
                file_path = os.path.join(root, file)
                
                # Process the file
                process_file(file_path, genre, file_name, file_year)

if __name__ == "__main__":
    crawl_and_process(INPUT_ROOT_DIRECTORY)
