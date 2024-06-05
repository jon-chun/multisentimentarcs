import os
import json
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from scipy.stats import zscore
from scipy.interpolate import UnivariateSpline  # Import UnivariateSpline

VADERanalyzer = SentimentIntensityAnalyzer()

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
    df['sentiment-vader'] = df['text'].apply(get_sentiment_vader)
    df['sentiment-textblob'] = df['text'].apply(get_sentiment_textblob)

    # Apply the function to the DataFrame
    result = timeseries_norm(df, ['sentiment-vader', 'sentiment-textblob'], 10)
    if result:
        print("Transformation successful.")
        return df
    else:
        print("Transformation failed.")
        return None  # Return None instead of -1


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

def main():
    # Iterate over all JSON files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_dir, file_name)
            df = process_file(file_path, output_dir)

            # Check if the DataFrame is valid
            if df is not None:
                # Define the output filename
                base_name = os.path.basename(file_path)
                output_file_name = os.path.splitext(base_name)[0] + "_srt_sentiments.csv"
                output_file_path = os.path.join(output_dir, output_file_name)
                
                # Save the DataFrame to CSV
                df.to_csv(output_file_path, index=False)
                logging.info(f"Processed and saved: {output_file_path}")

                # Extract film name and year from the base name
                film_name = "_".join(base_name.split('_')[:-1])
                film_year = base_name.split('_')[-1].split('.')[0]
                
                # Plot and save sentiment analysis
                plot_sentiment_sma(df, df_cols_list=['sentiment-vader_norm','sentiment-textblob_norm'], win_per=10, film_name=film_name, film_year=film_year)
            else:
                logging.error(f"Failed to process file: {file_path}")

if __name__ == "__main__":
    main()