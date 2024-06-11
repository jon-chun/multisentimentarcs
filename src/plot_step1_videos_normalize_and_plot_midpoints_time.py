import os
import re
import logging
import pandas as pd
from scipy.stats import zscore
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from tqdm import tqdm

# Constants
PLOT_TYPE = "videos"  # "videos" or "transcripts"

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

def split_string_on_4_digits(input_string):
    """Split a string based on the first occurrence of a 4-digit number."""
    match = re.search(r'\d{4}', input_string)
    if match:
        four_digit_number = match.group(0)
        parts = input_string.split(four_digit_number, 1)
        before = parts[0]
        after = parts[1] if len(parts) > 1 else ''
        return before, four_digit_number, after
    else:
        return input_string, '', ''

def handle_outliers(series):
    """Handle outliers in the series by capping them at a threshold."""
    threshold = 3  # Number of standard deviations to cap outliers
    mean = series.mean()
    std = series.std()
    series = series.clip(lower=mean - threshold * std, upper=mean + threshold * std)
    return series

def timeseries_norm(df, df_col_list=['vader', 'textblob', 'llama3'], sma_per=10):
    try:
        if 'time_midpoint' not in df.columns:
            raise ValueError("Column 'time_midpoint' not found in DataFrame")
        if sma_per < 3 or sma_per > 20:
            raise ValueError("sma_per must be between 3 and 20 percent")
        window_size = int(len(df) * (sma_per / 100))
        if window_size < 1:
            window_size = 1
        for col in df_col_list:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in DataFrame")
            # Handle outliers
            df[col] = handle_outliers(df[col])
            # Normalize the entire column first
            normalized_col = zscore(df[col])
            # Compute SMA on normalized data with min_periods adjusted
            smoothed_col = pd.Series(normalized_col).rolling(window=window_size, min_periods=window_size//2).mean()
            # Check for NaN values and fill them
            smoothed_col = smoothed_col.bfill().ffill()
            # Apply smoothing using spline
            spline = UnivariateSpline(df['time_midpoint'], smoothed_col, s=1)
            fitted_values = spline(df['time_midpoint'])
            new_col_name = f"{col}_norm"
            df[new_col_name] = fitted_values
        return True
    except Exception as e:
        logging.error(f"Error in timeseries normalization: {e}")
        return False

def plot_sentiment_sma(df, df_cols_list=['vader_norm', 'textblob_norm', 'llama3_norm'], win_per=10, film_name='', film_year='', output_dir=''):
    try:
        if df is None or len(df) == 0:
            raise ValueError("DataFrame is empty or None")
        plt.figure(figsize=(12, 6))
        for col in df_cols_list:
            if col in df.columns:
                sns.lineplot(data=df, x='time_midpoint', y=col, label=col)
                plt.scatter(df['time_midpoint'], df[col], alpha=0.5, label=f'{col} Values')
        plt.title(f'Sentiment Analysis for {film_name} ({film_year})')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Sentiment')
        plt.legend()
        plt.grid(True)
        output_plot_path = os.path.join(output_dir, f'{film_name}_{film_year}_sma10_plot.png')
        os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
        plt.savefig(output_plot_path)
        plt.close()
        logging.info(f"Sentiment plot saved: {output_plot_path}")
    except Exception as e:
        logging.error(f"Error plotting sentiment SMA: {e}")

def plot_kde(df=None, df_col_list=['vader_norm', 'textblob_norm', 'llama3_norm'], film_name='', film_year='', output_dir="plots", plot_title="KDE Distributions", file_suffix="_kde_plot.png"):
    try:
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.figure(figsize=(12, 8))
        for col in df_col_list:
            if col in df.columns:
                sns.kdeplot(data=df, x=col, label=col, fill=True)
        plt.title(f'{plot_title} for {film_name} ({film_year})', fontsize=16)
        plt.xlabel('Value', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend(title="Columns", fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        output_plot_path = os.path.join(output_dir, f'{film_name}_{film_year}{file_suffix}')
        os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
        plt.savefig(output_plot_path, dpi=300)
        plt.close()
        logging.info(f"KDE plot saved: {output_plot_path}")
    except Exception as e:
        logging.error(f"Error plotting KDE: {e}")

def process_file(file_path):
    """Process the given file and generate required output files."""
    logging.info(f"Processing file: {file_path}")
    
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return

    # Extract the genre, film_name, and film_year from the input file path
    parts = file_path.split(os.sep)
    genre = parts[-3]
    film_info = parts[-2]

    film_name, film_year, _ = split_string_on_4_digits(film_info)
    film_name = film_name.strip("_")
    film_year = film_year.strip("_")
    
    logging.info(f"Extracted info - film_name: '{film_name}', film_year: '{film_year}'")

    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return

    # Check if 'time_midpoint' column exists, if not, create it
    if 'time_midpoint' not in df.columns:
        logging.error("Column 'time_midpoint' not found in DataFrame")
        return

    # Normalize the time series data
    success = timeseries_norm(df, df_col_list=['vader', 'textblob', 'llama3'], sma_per=10)
    if success:
        # Construct the output directory path and ensure it exists
        output_dir = os.path.join(OUTPUT_ROOT_DIRECTORY, genre)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the normalized DataFrame to a CSV file
        normalized_csv_path = os.path.join(output_dir, f"{film_name}_{film_year}_normalized_sentiments.csv")
        os.makedirs(os.path.dirname(normalized_csv_path), exist_ok=True)
        df.to_csv(normalized_csv_path, index=False)
        logging.info(f"Normalized sentiments saved: {normalized_csv_path}")

        # Generate and save the SMA plot
        plot_sentiment_sma(df, df_cols_list=['vader_norm', 'textblob_norm', 'llama3_norm'], win_per=10, film_name=film_name, film_year=film_year, output_dir=output_dir)

        # Generate and save the KDE plot
        plot_kde(df, df_col_list=['vader_norm', 'textblob_norm', 'llama3_norm'], film_name=film_name, film_year=film_year, output_dir=output_dir)

def crawl_and_process(input_dir):
    """Crawl through the directory and process each file."""
    file_list = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith("_description_sentiment_transcript.csv"):
                file_list.append(os.path.join(root, file))
    
    # Sort the files list
    file_list.sort()

    for file_path in file_list:
        logging.info(f"\n\nPROCESSING FILE: {file_path}")
        process_file(file_path)

if __name__ == "__main__":
    crawl_and_process(INPUT_ROOT_DIRECTORY)
