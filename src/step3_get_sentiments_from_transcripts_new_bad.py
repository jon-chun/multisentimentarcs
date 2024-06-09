import os
import pandas as pd
import logging
import ollama
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import zscore
from scipy.interpolate import UnivariateSpline

# Initialize tqdm for pandas
tqdm.pandas()

# Initialize VADER sentiment analyzer
VADERanalyzer = SentimentIntensityAnalyzer()

# Configuration constants
PLOT_ONLY = False
OVERWRITE_SRT_FLAG = True
MAX_CALL_OLLAMA = 3
SAVE_INTERVAL = 100  # Save progress every 100 lines

# Define input and output directories
INPUT_ROOT_DIR = "../data/transcripts_combined/"
OUTPUT_ROOT_DIR = "../data/transcripts_sentiments/"
plot_dir = OUTPUT_ROOT_DIR  # Ensure plots are saved in the same batch output directory

# Ensure the output directories exist
if not os.path.exists(OUTPUT_ROOT_DIR):
    os.makedirs(OUTPUT_ROOT_DIR)

# Set up logging
log_file_path = os.path.join(OUTPUT_ROOT_DIR, 'process_log.log')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler()
                    ])

def get_sentiment_textblob(text: str) -> float:
    """Calculate sentiment polarity using TextBlob."""
    return TextBlob(text).sentiment.polarity

def get_sentiment_vader(text: str) -> float:
    """Calculate sentiment polarity using VADER."""
    return VADERanalyzer.polarity_scores(text)['compound']

def get_sentiment_ollama(text: str) -> float:
    """Calculate sentiment polarity using Ollama API."""
    for attempt in range(1, MAX_CALL_OLLAMA + 1):
        try:
            logging.info(f"Attempt {attempt}: Sending text to Ollama for sentiment analysis")
            res = ollama.chat(model="phi3sentiment",
                              messages=[{'role': 'user', 'content': f'Give the sentiment polarity float value for: {text}'}],
                              stream=False, temperature=0.3, top_p=0.2)
            if 'message' in res and 'content' in res['message']:
                text_sentiment_float_str = res['message']['content'].strip()
                try:
                    text_sentiment_float = float(text_sentiment_float_str)
                    logging.info("Successfully converted sentiment to float")
                    return text_sentiment_float
                except ValueError:
                    logging.warning(f"Attempt {attempt}: Could not convert to float: {text_sentiment_float_str}")
            else:
                logging.error(f"Attempt {attempt}: Unexpected API response: {res}")
        except Exception as e:
            logging.error(f"Attempt {attempt}: Error during sentiment analysis: {e}")
    logging.error(f"All {MAX_CALL_OLLAMA} attempts failed for text: {text}. Returning 0.0")
    return 0.0

def process_file(file_path: str, output_subdir: str, save_interval=SAVE_INTERVAL):
    try:
        logging.info(f"Processing file: {file_path}")
        df = pd.read_csv(file_path)
        df['start'] = df['start'].astype(float)
        df['duration'] = df['duration'].astype(float)
        df['line_no'] = df.index + 1
        df['midpoint'] = df['start'] + df['duration'] / 2
        df['sentiment-vader'] = df['text'].apply(lambda x: get_sentiment_vader(x) if x else 0.0)
        df['sentiment-textblob'] = df['text'].apply(lambda x: get_sentiment_textblob(x) if x else 0.0)
        df['sentiment-ollama'] = df['text'].progress_apply(lambda x: get_sentiment_ollama(x) if x else 0.0)
        if not timeseries_norm(df, ['sentiment-vader', 'sentiment-textblob', 'sentiment-ollama'], 10):
            logging.error("Transformation failed.")
            return None
        output_file_path = os.path.join(output_subdir, f"{os.path.splitext(os.path.basename(file_path))[0]}_partial.csv")
        df.to_csv(output_file_path, index=False)
        return df
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None

def timeseries_norm(df, df_col_list=['sentiment-vader', 'sentiment-textblob', 'sentiment-ollama'], sma_per=10):
    try:
        if sma_per < 3 or sma_per > 20:
            raise ValueError("sma_per must be between 3 and 20 percent")
        window_size = int(len(df) * (sma_per / 100))
        if window_size < 1:
            window_size = 1
        for col in df_col_list:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in DataFrame")
            smoothed_col = df[col].rolling(window=window_size, min_periods=1).mean()
            normalized_col = zscore(smoothed_col)
            spline = UnivariateSpline(df['midpoint'], normalized_col, s=1)
            fitted_values = spline(df['midpoint'])
            df[f"{col}_norm"] = fitted_values
        return True
    except Exception as e:
        logging.error(f"An error occurred during normalization: {e}")
        return False

def plot_sentiment_sma(df, df_cols_list, win_per, film_name, film_year, output_subdir):
    try:
        window_size = max(1, int(len(df) * win_per / 100))
        plt.figure(figsize=(12, 6))
        for col in df_cols_list:
            if col in df.columns:
                df[f'{col}_sma'] = df[col].rolling(window=window_size, min_periods=1).mean()
                sns.lineplot(data=df, x='midpoint', y=f'{col}_sma', label=f'{col} SMA')
                plt.scatter(df['midpoint'], df[col], alpha=0.5, label=f'{col} Values')
        plt.title(f'Sentiment Analysis for {film_name} ({film_year})')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Sentiment')
        plt.legend()
        plt.grid(True)
        output_plot_path = os.path.join(output_subdir, f'{film_name}_{film_year}_sma10_plot.png')
        plt.savefig(output_plot_path)
        plt.close()
        logging.info(f"Sentiment plot saved: {output_plot_path}")
    except Exception as e:
        logging.error(f"Error plotting sentiment SMA for {film_name} ({film_year}): {e}")

def plot_kde(df, df_col_list, film_name, film_year, output_subdir, plot_title, file_suffix):
    try:
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
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
        output_plot_path = os.path.join(output_subdir, f'{film_name}_{film_year}{file_suffix}')
        plt.savefig(output_plot_path, dpi=300)
        plt.close()
        logging.info(f"KDE plot saved: {output_plot_path}")
    except Exception as e:
        logging.error(f"Error plotting KDE for {film_name} ({film_year}): {e}")

def main():
    logging.info("Starting main processing function")
    
    if not os.path.exists(INPUT_ROOT_DIR):
        logging.error(f"Input directory does not exist: {INPUT_ROOT_DIR}")
        return
    
    for root, _, files in os.walk(INPUT_ROOT_DIR):
        logging.info(f"Current directory: {root}")
        if root == INPUT_ROOT_DIR:
            continue  # Skip the root directory itself
        genre = os.path.basename(root)
        for file_name in files:
            if file_name.endswith('_clean_transcripts.csv'):
                file_path = os.path.join(root, file_name)
                logging.info(f"Processing file: {file_path}")
                try:
                    film_name, film_year, _ = os.path.splitext(file_name)[0].rsplit('_', 2)
                except ValueError:
                    logging.error(f"Unexpected file name format: {file_name}")
                    continue
                
                output_subdir = os.path.join(OUTPUT_ROOT_DIR, genre, film_name)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                logging.info(f"Processing {file_path} into {output_subdir}")
                
                output_file_name = f"{film_name}_{film_year}_srt_sentiments.csv"
                output_file_path = os.path.join(output_subdir, output_file_name)
                if os.path.exists(output_file_path) and not OVERWRITE_SRT_FLAG:
                    logging.info(f"Output file already exists and OVERWRITE_SRT_FLAG is not set. Skipping file: {file_path}")
                    continue

                df = process_file(file_path, output_subdir) if not PLOT_ONLY else pd.read_csv(output_file_path)
                if df is None:
                    continue
                if not PLOT_ONLY:
                    df.to_csv(output_file_path, index=False)
                    logging.info(f"Processed and saved: {output_file_path}")

                logging.info("Plotting sentiment SMA...")
                plot_sentiment_sma(df, ['sentiment-vader_norm', 'sentiment-textblob_norm', 'sentiment-ollama_norm'], 10, film_name, film_year, output_subdir)
                logging.info("Sentiment SMA plotted and saved.")
                
                logging.info("Plotting KDE distributions...")
                plot_kde(df, ['sentiment-vader_norm', 'sentiment-textblob_norm', 'sentiment-ollama_norm'], film_name, film_year, output_subdir, "KDE Distributions", "_kde_sentiments.png")
                logging.info("KDE distributions plotted and saved.")

if __name__ == "__main__":
    main()
