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
INPUT_ROOT_DIRECTORY = "../data/keyframes_descriptions"
OUTPUT_ROOT_DIRECTORY = "../data/keyframes_sentiments"

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
                        'content': f'Give the sentiment polarity float value anywhere between -1.0 to 1.0 for: {text}'
                    }
                ],
                options={
                    'temperature': 0.3, 
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

def process_film(genre, film_name, film_year, file_paths):
    # Output file path
    output_subdir = os.path.join(OUTPUT_ROOT_DIRECTORY, genre, f"{film_name}_{film_year}")
    os.makedirs(output_subdir, exist_ok=True)
    output_file_path = os.path.join(output_subdir, f"{film_name}_{film_year}_sentiment_transcript.csv")
    
    # Check if output file already exists
    if os.path.exists(output_file_path):
        logging.info(f"Output file {output_file_path} already exists. Skipping...")
        return

    # Combine texts from all files into one DataFrame
    all_texts = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Extract scene number from the filename
        scene_number = int(os.path.basename(file_path).split('_')[0].replace('scene', ''))
        # Concatenate all lines into one single string
        concatenated_text = " ".join([line.strip() for line in lines if line.strip()])
        all_texts.append((scene_number, concatenated_text))

    # Sort by scene number
    all_texts.sort(key=lambda x: x[0])
    data = {'text': [text for _, text in all_texts]}
    df = pd.DataFrame(data)
    
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
    film_files = {}
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.startswith('scene') and file.endswith('_description.txt'):
                # Extract the relative path from the input directory to the current file
                relative_path = os.path.relpath(root, INPUT_ROOT_DIRECTORY)
                # The genre is the first part of this relative path
                genre = relative_path.split(os.sep)[0]
                
                # Extract file_name and file_year from the file name
                parts = file.split('_')
                file_name = '_'.join(parts[1:-1])
                file_year = parts[-1].split('.')[0]
                
                # Aggregate files for each film
                film_key = (genre, file_name, file_year)
                if film_key not in film_files:
                    film_files[film_key] = []
                film_files[film_key].append(os.path.join(root, file))
        
    # Process each film
    for (genre, file_name, file_year), file_paths in film_files.items():
        process_film(genre, file_name, file_year, file_paths)

if __name__ == "__main__":
    crawl_and_process(INPUT_ROOT_DIRECTORY)
