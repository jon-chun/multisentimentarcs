import os
import json
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

PLOT_ONLY = False
OVERWRITE_SRT_FLAG = True

MAX_CALL_OLLAMA = 3
SAVE_INTERVAL = 100  # Save progress every 100 lines

# Define input and output directories
input_dir = "../data/transcripts/batch"
output_dir = "../data/transcripts_sentiments/batch"
plot_dir = output_dir  # Ensure plots are saved in the same batch output directory

# Ensure the output directories exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_sentiment_textblob(text: str) -> float:
    """
    Perform text sentiment analysis returning a polarity anywhere between -1.0 to +1.0 for most negative to most positive respectively with 0.0 for absolutely neutral
    """
    return TextBlob(text).sentiment.polarity

def get_sentiment_vader(text: str) -> float:
    """
    Perform text sentiment analysis returning a polarity anywhere between -1.0 to +1.0 for most negative to most positive respectively with 0.0 for absolutely neutral
    """
    return VADERanalyzer.polarity_scores(text)['compound']

def get_sentiment_ollama(text_str: str) -> float:
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
                model="phi3sentiment",
                messages=[
                    {
                        'role': 'user',
                        'content': 'Give a polarity value between (-1.0 and 1.0) for this text based on the sentiment or emotions evoked: {text_str}'
                    }
                ]
            )

            if 'message' in res and 'content' in res['message']:
                text_sentiment_float_str = res['message']['content']
                print(f'  ollama res = [{text_sentiment_float_str}]')
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


def process_file(file_path: str, output_dir: str, save_interval=SAVE_INTERVAL):
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
    df['sentiment-vader'] = df['text'].apply(lambda x: get_sentiment_vader(x) if x else 0.0)
    df['sentiment-textblob'] = df['text'].apply(lambda x: get_sentiment_textblob(x) if x else 0.0)
    df['sentiment-ollama'] = df['text'].progress_apply(lambda x: get_sentiment_ollama(x) if x else 0.0)
    #                        =[get_sentiment_ollama(x) if x else 0.0 for x in tqdm(df['text'], desc='Processing')]
    #                        =df['text'].progress_apply(lambda x: get_sentiment_ollama(x) if x else 0.0)
    # df['sentiment-ollama'] = df['text'].apply(lambda x: get_sentiment_ollama(x) if x else 0.0)

    # Apply the function to the DataFrame
    result = timeseries_norm(df, ['sentiment-vader', 'sentiment-textblob', 'sentiment-ollama'], 10)
    if result:
        print("Transformation successful.")
    else:
        print("Transformation failed.")
        return None  # Return None instead of -1

    # Periodically save progress
    base_name = os.path.basename(file_path)
    output_file_name = os.path.splitext(base_name)[0] + "_partial.csv"
    output_file_path = os.path.join(output_dir, output_file_name)
    
    for i in range(0, len(df), save_interval):
        df.iloc[:i + save_interval].to_csv(output_file_path, index=False)

    return df


def timeseries_norm(df, df_col_list, sma_per):
    try:
        # Validate sma_per is within the allowed range
        if sma_per < 3 or sma_per > 20:
            raise ValueError("sma_per must be between 3 and 20 percent")
        
        # Calculate the window size for SMA
        window_size = int(len(df) * (sma_per / 100))
        if window_size < 1:
            window_size = 1

        # Process each column specified in df_col_list
        for col in df_col_list:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in DataFrame")
            
            # SMA Smoothing
            smoothed_col = df[col].rolling(window=window_size, min_periods=1).mean()

            # Z-score Normalization
            normalized_col = zscore(smoothed_col)

            # Non-Parametric Curve Fitting
            spline = UnivariateSpline(df['midpoint'], normalized_col, s=1)  # Use 'midpoint' for fitting
            fitted_values = spline(df['midpoint'])

            # Create new column names
            new_col_name = f"{col}_norm"

            # Assign the fitted values to the new column
            df[new_col_name] = fitted_values

        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False    


def plot_sentiment_sma(df, df_cols_list=['sentiment-vader-zscore'], win_per=10, film_name='', film_year=''):    
    """
    Plot sentiment values with Simple Moving Average (SMA).

    Parameters:
    - df: pandas DataFrame, the data frame containing the data.
    - df_cols_list: list of str, the columns to plot.
    - win_per: int, percentage of the data length to use as SMA window size.
    - film_name: str, name of the film.
    - film_year: str, year of the film.

    Outputs a plot saved as "plot_{film_name}_{film_year}_sentiment.png" in the plot directory.
    """
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


def plot_kdes(df, df_col_list, film_name='', film_year='', output_dir="plots", plot_title="KDE Distributions", file_suffix="_kde_sentiments.png"):
    """
    Create KDE distribution plots for the specified columns in the DataFrame.

    Parameters:
    - df: pandas DataFrame, the data frame containing the data.
    - df_col_list: list of str, the columns to plot.
    - film_name: str, name of the film.
    - film_year: str, year of the film.
    - output_dir: str, directory where the plot will be saved.
    - plot_title: str, title of the plot.
    - file_suffix: str, suffix of the output plot file.

    Outputs a high-resolution KDE plot saved in the specified directory.
    """
    # Ensure the output directory exists
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
    plt.savefig(output_plot_path, dpi=300)  # Save with high resolution
    plt.close()
    logging.info(f"KDE plot saved: {output_plot_path}")

"""
def main():
    # Iterate over all JSON files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_dir, file_name)
            
            # Check if output file already exists
            base_name = os.path.basename(file_path)
            output_file_name = os.path.splitext(base_name)[0] + "_srt_sentiments.csv"
            output_file_path = os.path.join(output_dir, output_file_name)
            
            if os.path.exists(output_file_path):
                logging.info(f"Output file already exists. Skipping file: {file_path}")
                continue


            if PLOT_ONLY == True:
                output_file_name = os.path.splitext(base_name)[0] + "_srt_sentiments.csv"
                output_file_path = os.path.join(output_dir, output_file_name)
                if os.path.exists(output_file_path):
                    df = pd.read_csv(output_file_path)
                else:
                    logging.error(f"File not found: {output_file_path}")
                    continue
            else:
                df = process_file(file_path, output_dir)


            # Check if the DataFrame is valid
            if df is not None:
                # Save the DataFrame to CSV
                df.to_csv(output_file_path, index=False)
                logging.info(f"Processed and saved: {output_file_path}")

                # Extract film name and year from the base name
                film_name = "_".join(base_name.split('_')[:-1])
                film_year = base_name.split('_')[-1].split('.')[0]
                
                # Plot and save sentiment analysis
                plot_sentiment_sma(df, df_cols_list=['sentiment-vader_norm','sentiment-textblob_norm','sentiment-ollama_norm'], win_per=10, film_name=film_name, film_year=film_year)

                # Apply the KDE plot function
                plot_kdes(df, ['sentiment-vader_norm','sentiment-textblob_norm','sentiment-ollama_norm'], film_name=film_name, film_year=film_year, output_dir=output_dir, plot_title="KDE Distributions", file_suffix="_kde_sentiments.png")
            else:
                logging.error(f"Failed to process file: {file_path}")

if __name__ == "__main__":
    main()
""";

def main():
    # Iterate over all JSON files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_dir, file_name)
            
            # Check if output file already exists
            base_name = os.path.basename(file_path)
            output_file_name = os.path.splitext(base_name)[0] + "_srt_sentiments.csv"
            output_file_path = os.path.join(output_dir, output_file_name)
            
            print(f" Testing if output_file_path exists: {output_file_path}")
            if os.path.exists(output_file_path):         
                logging.info(f"Output file already exists. Skipping file: {file_path}")
                if OVERWRITE_SRT_FLAG == True:
                    continue
            
            if PLOT_ONLY:
                if os.path.exists(output_file_path):
                    df = pd.read_csv(output_file_path)
                else:
                    logging.error(f"File not found: {output_file_path}")
                    continue
            else:
                df = process_file(file_path, output_dir)

            # Check if the DataFrame is valid
            if df is not None:
                # Apply normalization
                result = timeseries_norm(df, ['sentiment-vader', 'sentiment-textblob', 'sentiment-ollama'], 10)
                if result:
                    logging.info("Normalization successful.")
                else:
                    logging.error("Normalization failed.")
                    continue
                
                # Save the DataFrame to CSV if not in PLOT_ONLY mode
                if not PLOT_ONLY:
                    df.to_csv(output_file_path, index=False)
                    logging.info(f"Processed and saved: {output_file_path}")

                # Extract film name and year from the base name
                film_name = "_".join(base_name.split('_')[:-1])
                film_year = base_name.split('_')[-1].split('.')[0]
                
                # Plot and save sentiment analysis
                plot_sentiment_sma(df, df_cols_list=['sentiment-vader_norm','sentiment-textblob_norm','sentiment-ollama_norm'], win_per=10, film_name=film_name, film_year=film_year)

                # Apply the KDE plot function
                plot_kdes(df, ['sentiment-vader_norm','sentiment-textblob_norm','sentiment-ollama_norm'], film_name=film_name, film_year=film_year, output_dir=output_dir, plot_title="KDE Distributions", file_suffix="_kde_sentiments.png")
            else:
                logging.error(f"Failed to process file: {file_path}")

if __name__ == "__main__":
    main()