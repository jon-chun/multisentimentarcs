import os
import json
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob

# Define input and output directories
input_dir = "../data/transcripts/batch"
output_dir = "../data/transcripts_sentiments/batch"
plot_dir = "../data/transcripts_sentiments"

# Ensure the output directories exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

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
    plot_sentiment_sma(df, df_cols_list=['sentiment'], win_per=10, film_name=film_name, film_year=film_year)

def plot_sentiment_sma(df, df_cols_list=['sentiment'], win_per=10, film_name='', film_year=''):
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
            sns.lineplot(data=df, x='line_no', y=f'{col}_sma', label=f'{col} SMA')
            plt.scatter(df['line_no'], df['sentiment'], color='red', s=10, label='Sentiment Points')
    
    plt.title(f'Sentiment Analysis for {film_name} ({film_year})')
    plt.xlabel('Line Number')
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
            process_file(file_path, output_dir)

if __name__ == "__main__":
    main()
