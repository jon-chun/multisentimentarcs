import os
import pandas as pd
import logging
from tqdm import tqdm
tqdm.pandas()
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ollama
from scipy.stats import zscore
from scipy.interpolate import UnivariateSpline

VADERanalyzer = SentimentIntensityAnalyzer()

file_suffix = "pic"  # [pic|srt]
input_extension = ".txt"
output_extension = "_pic_sentiments.csv"
PLOT_ONLY = False
OVERWRITE_SRT_FLAG = True

MAX_CALL_OLLAMA = 3
SAVE_INTERVAL = 100  # Save progress every 100 lines

# Define input and output directories
input_dir = "../data/stills_sentiments"
output_dir = "../data/stills_sentiments_timeseries"
plot_dir = output_dir  # Ensure plots are saved in the same batch output directory

# Ensure the output directories exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_sentiment_textblob(text: str) -> float:
    return TextBlob(text).sentiment.polarity

def get_sentiment_vader(text: str) -> float:
    return VADERanalyzer.polarity_scores(text)['compound']

def get_sentiment_ollama(text: str) -> float:
    for attempt in range(1, MAX_CALL_OLLAMA + 1):
        try:
            logging.info(f"Attempt {attempt}: Sending text to Ollama for sentiment analysis")
            res = ollama.chat(
                model="phi3sentiment",
                messages=[{'role': 'user', 'content': f'Give the sentiment polarity float value for: {text}'}],
                stream=False,
                options={"temperature": 0.3, "top_p": 0.2}
            )
            if 'message' in res and 'content' in res['message']:
                text_sentiment_float_str = res['message']['content'].strip()
                try:
                    text_sentiment_float = float(text_sentiment_float_str)
                    logging.info(f"Received sentiment analysis response and successfully converted to float")
                    return text_sentiment_float
                except ValueError:
                    logging.warning(f"Attempt {attempt}: Could not convert response to float: {text_sentiment_float_str}")
            else:
                logging.error(f"Attempt {attempt}: Unexpected API response format: {res}")
        except Exception as e:
            logging.error(f"Attempt {attempt}: Error during sentiment analysis for text: {e}")
    logging.error(f"All {MAX_CALL_OLLAMA} attempts failed for text: {text}. Returning 0.0")
    return 0.0

def read_text_files(directory):
    text_list = []
    for filename in os.listdir(directory):
        if filename.endswith(input_extension):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                text = file.read()
                text_str = " ".join(text.split(" "))
                text_list.append(text_str)
    return text_list

def process_file(file_path: str, output_dir: str, save_interval=SAVE_INTERVAL):
    try:
        text_list = read_text_files(os.path.dirname(file_path))
        df = pd.DataFrame(list(text_list))
        df.columns = ['text']
        df['line_no'] = df.index + 1
        df['sentiment-vader'] = df['text'].apply(lambda x: get_sentiment_vader(x) if x else 0.0)
        df['sentiment-textblob'] = df['text'].apply(lambda x: get_sentiment_textblob(x) if x else 0.0)
        df['sentiment-ollama'] = df['text'].progress_apply(lambda x: get_sentiment_ollama(x) if x else 0.0)
        df['midpoint'] = range(len(df))
        result = timeseries_norm(df, ['sentiment-vader', 'sentiment-textblob', 'sentiment-ollama'], 10)
        if result:
            logging.info("Transformation successful.")
        else:
            logging.error("Transformation failed.")
            return None

        base_name = os.path.basename(file_path)
        output_file_name = os.path.splitext(base_name)[0] + "_partial.csv"
        output_file_path = os.path.join(output_dir, output_file_name)
        for i in range(0, len(df), save_interval):
            df.iloc[:i + save_interval].to_csv(output_file_path, index=False)
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
            new_col_name = f"{col}_norm"
            df[new_col_name] = fitted_values
        return True
    except Exception as e:
        logging.error(f"Error in timeseries normalization: {e}")
        return False

def plot_sentiment_sma(df, df_cols_list=['sentiment-vader_norm', 'sentiment-textblob_norm', 'sentiment-ollama_norm'], win_per=10, film_name='', film_year=''):
    try:
        if df is None or len(df) == 0:
            raise ValueError("DataFrame is empty or None")
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
        output_plot_path = os.path.join(plot_dir, f'plot_{film_name}_{film_year}_sentiment.png')
        plt.savefig(output_plot_path)
        plt.close()
        logging.info(f"Sentiment plot saved: {output_plot_path}")
    except Exception as e:
        logging.error(f"Error plotting sentiment SMA: {e}")

def plot_kde(df=None, df_col_list=['sentiment-vader_norm', 'sentiment-textblob_norm', 'sentiment-ollama_norm'], film_name='', film_year='', output_dir="plots", plot_title="KDE Distributions", file_suffix="_kde_sentiments.png"):
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
        output_plot_path = os.path.join(output_dir, f'plot_{film_name}_{film_year}{file_suffix}')
        plt.savefig(output_plot_path, dpi=300)
        plt.close()
        logging.info(f"KDE plot saved: {output_plot_path}")
    except Exception as e:
        logging.error(f"Error plotting KDE: {e}")

def main():
    for subdirs_names in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdirs_names)
        if os.path.isdir(subdir_path):
            stills_list = sorted(os.listdir(subdir_path))
            for stills_name_now in stills_list:
                if stills_name_now.endswith(input_extension):
                    file_path = os.path.join(subdir_path, stills_name_now)
                    output_file_name = f"{subdirs_names}_{stills_name_now.split('.')[0]}{output_extension}"
                    output_file_path = os.path.join(output_dir, output_file_name)
                    logging.info(f"Testing if output_file_path exists: {output_file_path}")
                    if os.path.exists(output_file_path):
                        if OVERWRITE_SRT_FLAG:
                            logging.info(f"Output file exists. Overwriting file: {stills_name_now}")
                            df = process_file(file_path, output_dir)
                            if df is not None:
                                df.to_csv(output_file_path, index=False)
                        else:
                            logging.info(f"Output file already exists. Skipping file: {stills_name_now}")
                            continue
                    else:
                        df = process_file(file_path, output_dir)
                        if df is not None:
                            df.to_csv(output_file_path, index=False)
                    if df is not None:
                        base_name = os.path.basename(file_path)
                        film_name = "_".join(base_name.split('_')[:-1])
                        film_year = base_name.split('_')[-1].split('.')[0]
                        logging.info("Plotting sentiment SMA...")
                        plot_sentiment_sma(df, df_cols_list=['sentiment-vader_norm', 'sentiment-textblob_norm', 'sentiment-ollama_norm'], win_per=10, film_name=film_name, film_year=film_year)
                        logging.info("Sentiment SMA plotted and saved.")
                        logging.info("Plotting KDE distributions...")
                        plot_kde(df, ['sentiment-vader_norm', 'sentiment-textblob_norm', 'sentiment-ollama_norm'], film_name=film_name, film_year=film_year, output_dir=output_dir, plot_title="KDE Distributions", file_suffix=f"_{file_suffix}_sentiments.png")
                        logging.info("KDE distributions plotted and saved.")

if __name__ == "__main__":
    main()
