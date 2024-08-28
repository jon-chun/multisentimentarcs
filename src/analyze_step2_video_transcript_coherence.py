import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, stats
from dtaidistance import dtw
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

def winsorize_series(series, limits=(0.05, 0.05)):
    winsorized = stats.mstats.winsorize(series, limits=limits)
    return np.ma.filled(winsorized, np.nan)  # Convert MaskedArray to regular array, replacing masked values with NaN

def min_max_scale(series):
    scaler = MinMaxScaler()
    # Remove NaN values before scaling
    non_nan = series[~np.isnan(series)]
    scaled = scaler.fit_transform(non_nan.values.reshape(-1, 1)).flatten()
    # Create a new series with the same index as the original, filling NaNs
    result = pd.Series(index=series.index, dtype=float)
    result.loc[~np.isnan(series)] = scaled
    return result

def z_score_normalize(series):
    return (series - series.mean()) / series.std()

def remove_outliers(series, z_threshold=3):
    z_scores = np.abs(stats.zscore(series))
    return series[z_scores < z_threshold]

def normalize_to_range(series, target_min=-1, target_max=1):
    series_min, series_max = series.min(), series.max()
    return (series - series_min) / (series_max - series_min) * (target_max - target_min) + target_min

def normalized_euclidean_distance(series1, series2):
    """
    Calculate the normalized Euclidean distance between two series.
    The result is between 0 (identical) and 1 (maximally different).
    """
    diff = series1 - series2
    return np.sqrt(np.sum(diff**2) / len(diff)) / (np.std(series1) + np.std(series2))

def normalized_dtw_distance(series1, series2):
    """
    Calculate the normalized DTW distance between two series.
    The result is between 0 (identical) and 1 (maximally different).
    """
    # Ensure the series are numpy arrays
    s1 = np.array(series1)
    s2 = np.array(series2)
    
    # Calculate the raw DTW distance
    raw_distance = dtw.distance(s1, s2)
    
    # Calculate the maximum possible DTW distance
    max_distance = dtw.distance(np.ones_like(s1), np.zeros_like(s2))
    
    # Normalize the distance
    return raw_distance / max_distance

try:
    title_year_str = "royal_wedding_1951"
    title_year_caps_str = "Royal_Wedding_1951"
    genre = 'musical'

    input_video_sentiment_path = os.path.join("..", "data", "plots", genre, f"{title_year_str}_normalized_sentiments.csv")
    input_transcript_sentiment_path = os.path.join("..", "data", "transcripts_sentiments", genre, f"{title_year_caps_str}_clean", f"{title_year_caps_str}_clean_sentiment_transcript.csv")

    video_df = pd.read_csv(input_video_sentiment_path)
    transcript_df = pd.read_csv(input_transcript_sentiment_path)

    print("Video DataFrame:")
    print(video_df.head())
    print(video_df.info())
    print(video_df.describe())

    print("\nTranscript DataFrame:")
    print(transcript_df.head())
    print(transcript_df.info())
    print(transcript_df.describe())

    print(f"\nLength of video_df: {len(video_df)}")
    print(f"Length of transcript_df: {len(transcript_df)}")

    # Ensure both dataframes have the required columns
    required_video_columns = ['vader_norm', 'textblob_norm', 'llama3_norm']
    required_transcript_columns = ['vader', 'textblob', 'llama3']

    if not all(col in video_df.columns for col in required_video_columns):
        raise ValueError(f"Video DataFrame is missing one or more required columns: {required_video_columns}")

    if not all(col in transcript_df.columns for col in required_transcript_columns):
        raise ValueError(f"Transcript DataFrame is missing one or more required columns: {required_transcript_columns}")

    # Determine which dataframe is shorter
    if len(video_df) < len(transcript_df):
        shorter_df, longer_df = video_df, transcript_df
        shorter_prefix, longer_prefix = 'video', 'transcript'
    else:
        shorter_df, longer_df = transcript_df, video_df
        shorter_prefix, longer_prefix = 'transcript', 'video'

    # Create time arrays for interpolation
    shorter_time = np.linspace(0, 1, len(shorter_df))
    longer_time = np.linspace(0, 1, len(longer_df))

    # Create common_df with the longer dataframe's length
    common_df = pd.DataFrame(index=range(len(longer_df)))

    # Interpolate and add sentiments for both video and transcript
    for prefix in ['video', 'transcript']:
        df = video_df if prefix == 'video' else transcript_df
        time = shorter_time if len(df) == len(shorter_df) else longer_time

        for column in ['vader_norm', 'textblob_norm', 'llama3_norm']:
            if prefix == 'video':
                values = df[column]
            else:
                values = df[column.replace('_norm', '')]

            values = pd.Series(winsorize_series(values))
            values = min_max_scale(values)

            f = interpolate.interp1d(time, values, kind='linear', bounds_error=False, fill_value="extrapolate")
            common_df[f'{prefix}_{column}'] = f(longer_time)

    # Calculate mean for video and transcript
    common_df['video_mean'] = common_df[['video_vader_norm', 'video_textblob_norm', 'video_llama3_norm']].mean(axis=1)
    common_df['transcript_mean'] = common_df[['transcript_vader_norm', 'transcript_textblob_norm', 'transcript_llama3_norm']].mean(axis=1)

    # Remove outliers and normalize
    video_mean_no_outliers = remove_outliers(common_df['video_mean'])
    transcript_mean_no_outliers = remove_outliers(common_df['transcript_mean'])

    # Find the overall min and max
    overall_min = min(video_mean_no_outliers.min(), transcript_mean_no_outliers.min())
    overall_max = max(video_mean_no_outliers.max(), transcript_mean_no_outliers.max())

    # Normalize both means to the same range based on the overall min and max
    common_df['video_mean_norm'] = normalize_to_range(common_df['video_mean'], overall_min, overall_max)
    common_df['transcript_mean_norm'] = normalize_to_range(common_df['transcript_mean'], overall_min, overall_max)

    # Apply LOWESS smoothing to both normalized means
    common_df['video_mean_smooth'] = lowess(common_df['video_mean_norm'], common_df.index, frac=0.15, it=0)[:, 1]
    common_df['transcript_mean_smooth'] = lowess(common_df['transcript_mean_norm'], common_df.index, frac=0.15, it=0)[:, 1]

    # Apply second z-score normalization to both smoothed series
    common_df['video_mean_final'] = z_score_normalize(common_df['video_mean_smooth'])
    common_df['transcript_mean_final'] = z_score_normalize(common_df['transcript_mean_smooth'])

    common_df['normalized_time'] = np.linspace(0, 1, len(common_df))

    output_csv = os.path.join("..", "data", "plots", genre, f"{title_year_str}_video_transcript_coherence.csv")
    common_df.to_csv(output_csv, index=False)
    print(f"Normalized data saved to: {output_csv}")

    # Compute metrics
    euclidean_similarity = 1 - normalized_euclidean_distance(common_df['video_mean_final'], common_df['transcript_mean_final'])
    dtw_similarity = 1 - normalized_dtw_distance(common_df['video_mean_final'], common_df['transcript_mean_final'])
    correlation, _ = pearsonr(common_df['video_mean_final'], common_df['transcript_mean_final'])

    print(f"Euclidean Similarity: {euclidean_similarity:.4f}")
    print(f"DTW Similarity: {dtw_similarity:.4f}")
    print(f"Pearson Correlation: {correlation:.4f}")

    # Plotting
    plt.figure(figsize=(16, 12))
    plt.plot(common_df['normalized_time'], common_df['video_mean_final'], label='Video Mean', linewidth=4, color='blue')
    plt.plot(common_df['normalized_time'], common_df['transcript_mean_final'], label='Transcript Mean', linewidth=4, color='red', linestyle='--')

    plt.legend(fontsize=24)
    plt.title('Royal Wedding (1951) Video and Transcript Sentiment Arcs\nSmoothed and Z-Score Normalized Means', fontsize=32)
    plt.xlabel('Normalized Time', fontsize=24)
    plt.ylabel('Z-Score Normalized Sentiment', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Set y-axis limits to ensure both series are fully visible
    y_min = min(common_df['video_mean_final'].min(), common_df['transcript_mean_final'].min())
    y_max = max(common_df['video_mean_final'].max(), common_df['transcript_mean_final'].max())
    plt.ylim(y_min - 0.1, y_max + 0.1)  # Add a small padding

    textstr = f'Euclidean Similarity: {euclidean_similarity:.4f}\nDTW Similarity: {dtw_similarity:.4f}\nCorrelation: {correlation:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=36,
             verticalalignment='top', bbox=props)

    plt.tight_layout()

    output_dir = os.path.join("..", "data", "plots", genre)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{title_year_str}_video_transcript_coherence.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    plt.close()

    print(common_df.describe())
    print(f"\nShape of common_df: {common_df.shape}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    print(traceback.format_exc())