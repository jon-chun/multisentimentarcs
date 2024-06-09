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
PLOT_TYPE = "transcripts" # "videos" or "transcripts"

MAX_CALL_OLLAMA = 3
MAX_WORKERS = 10  # Adjust based on the optimal number of concurrent requests supported by the API

# Initialize sentiment analyzers
vader_analyzer = SentimentIntensityAnalyzer()

# Input and output root directories
if PLOT_TYPE == "videos":
    INPUT_ROOT_DIRECTORY = "../data/keyframes_sentiments"
    OUTPUT_ROOT_DIRECTORY = "../data/plots"
elif PLOT_TYPE == "transcripts":
    INPUT_ROOT_DIRECTORY = "../data/transcripts_sentiments"
    OUTPUT_ROOT_DIRECTORY = "../data/transcripts_plots"
else:
    print(f"Invalid PLOT_TYPE: {PLOT_TYPE}")
    exit()

# Setup logging
logging.basicConfig(level=logging.INFO)


def process_file(file_path):
    logging.info(f"Processing file: {file_path}")
    
    return

def crawl_and_process(input_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith("_clean_sentiment_transcript.csv"):
                print(f"\n\nPROCESSING: {file}")
                
                # Full file path
                file_path = os.path.join(root, file)
                
                # Process the file
                process_file(file_path)

if __name__ == "__main__":
    crawl_and_process(INPUT_ROOT_DIRECTORY)
