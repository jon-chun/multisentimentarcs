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
    model_name = 'claude-3-opus-20240229'

    # Update input file paths
    input_video_claude3_path = os.path.join('..', 'data', 'plots_claude3', f"{title_year_str}_{model_name}.csv")
    input_video_open2step_path = os.path.join('..', 'data', 'plots', genre, f'{title_year_str}_normalized_sentiments.csv')

    # Read the CSV files
    claude3_df = pd.read_csv(input_video_claude3_path)
    open2step_df = pd.read_csv(input_video_open2step_path)

    print("Claude3 DataFrame:")
    print(claude3_df.head())
    print(claude3_df.info())
    print(claude3_df.describe())

    print("\nOpen2Step DataFrame:")
    print(open2step_df.head())
    print(open2step_df.info())
    print(open2step_df.describe())

    print(f"\nLength of claude3_df: {len(claude3_df)}")
    print(f"Length of open2step_df: {len(open2step_df)}")

    # Ensure both dataframes have the required columns
    required_claude3_columns = ['sentiment']  # Adjust this based on the actual column name in the Claude3 CSV
    required_open2step_columns = ['vader_norm', 'textblob_norm', 'llama3_norm']

    if not all(col in claude3_df.columns for col in required_claude3_columns):
        raise ValueError(f"Claude3 DataFrame is missing one or more required columns: {required_claude3_columns}")

    if not all(col in open2step_df.columns for col in required_open2step_columns):
        raise ValueError(f"Open2Step DataFrame is missing one or more required columns: {required_open2step_columns}")

    # Determine which dataframe is shorter
    if len(claude3_df) < len(open2step_df):
        shorter_df, longer_df = claude3_df, open2step_df
        shorter_prefix, longer_prefix = 'claude3', 'open2step'
    else:
        shorter_df, longer_df = open2step_df, claude3_df
        shorter_prefix, longer_prefix = 'open2step', 'claude3'

    # Create time arrays for interpolation
    shorter_time = np.linspace(0, 1, len(shorter_df))
    longer_time = np.linspace(0, 1, len(longer_df))

    # Create common_df with the longer dataframe's length
    common_df = pd.DataFrame(index=range(len(longer_df)))

    # Interpolate and add sentiments for both Claude3 and Open2Step
    for prefix in ['claude3', 'open2step']:
        df = claude3_df if prefix == 'claude3' else open2step_df
        time = shorter_time if len(df) == len(shorter_df) else longer_time

        if prefix == 'claude3':
            values = df['sentiment']  # Adjust this if the column name is different
            values = pd.Series(winsorize_series(values))
            values = min_max_scale(values)
            f = interpolate.interp1d(time, values, kind='linear', bounds_error=False, fill_value="extrapolate")
            common_df[f'{prefix}_sentiment'] = f(longer_time)
        else:
            for column in ['vader_norm', 'textblob_norm', 'llama3_norm']:
                values = df[column]
                values = pd.Series(winsorize_series(values))
                values = min_max_scale(values)
                f = interpolate.interp1d(time, values, kind='linear', bounds_error=False, fill_value="extrapolate")
                common_df[f'{prefix}_{column}'] = f(longer_time)

    # Calculate mean for Open2Step
    common_df['open2step_mean'] = common_df[['open2step_vader_norm', 'open2step_textblob_norm', 'open2step_llama3_norm']].mean(axis=1)

    # Remove outliers and normalize
    claude3_no_outliers = remove_outliers(common_df['claude3_sentiment'])
    open2step_mean_no_outliers = remove_outliers(common_df['open2step_mean'])

    # Find the overall min and max
    overall_min = min(claude3_no_outliers.min(), open2step_mean_no_outliers.min())
    overall_max = max(claude3_no_outliers.max(), open2step_mean_no_outliers.max())

    # Normalize both means to the same range based on the overall min and max
    common_df['claude3_norm'] = normalize_to_range(common_df['claude3_sentiment'], overall_min, overall_max)
    common_df['open2step_norm'] = normalize_to_range(common_df['open2step_mean'], overall_min, overall_max)

    # Apply LOWESS smoothing to both normalized means
    common_df['claude3_smooth'] = lowess(common_df['claude3_norm'], common_df.index, frac=0.15, it=0)[:, 1]
    common_df['open2step_smooth'] = lowess(common_df['open2step_norm'], common_df.index, frac=0.15, it=0)[:, 1]

    # Apply second z-score normalization to both smoothed series
    common_df['claude3_final'] = z_score_normalize(common_df['claude3_smooth'])
    common_df['open2step_final'] = z_score_normalize(common_df['open2step_smooth'])

    common_df['normalized_time'] = np.linspace(0, 1, len(common_df))

    output_csv = os.path.join("..", "data", "plots", genre, f"{title_year_str}_coherence_claude3_open2step.csv")
    common_df.to_csv(output_csv, index=False)
    print(f"Normalized data saved to: {output_csv}")

    # Compute metrics
    euclidean_similarity = 1 - normalized_euclidean_distance(common_df['claude3_final'], common_df['open2step_final'])
    dtw_similarity = 1 - normalized_dtw_distance(common_df['claude3_final'], common_df['open2step_final'])
    correlation, _ = pearsonr(common_df['claude3_final'], common_df['open2step_final'])

    print(f"Euclidean Similarity: {euclidean_similarity:.4f}")
    print(f"DTW Similarity: {dtw_similarity:.4f}")
    print(f"Pearson Correlation: {correlation:.4f}")

    # Plotting
    plt.figure(figsize=(16, 12))
    plt.plot(common_df['normalized_time'], common_df['claude3_final'], label='Claude3', linewidth=4, color='blue')
    plt.plot(common_df['normalized_time'], common_df['open2step_final'], label='Open2Step', linewidth=4, color='red', linestyle='--')

    plt.legend(fontsize=24)
    plt.title(f'{title_year_caps_str} Claude3 and Open2Step Sentiment Arcs\nSmoothed and Z-Score Normalized', fontsize=32)
    plt.xlabel('Normalized Time', fontsize=24)
    plt.ylabel('Z-Score Normalized Sentiment', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Set y-axis limits to ensure both series are fully visible
    y_min = min(common_df['claude3_final'].min(), common_df['open2step_final'].min())
    y_max = max(common_df['claude3_final'].max(), common_df['open2step_final'].max())
    plt.ylim(y_min - 0.1, y_max + 0.1)  # Add a small padding

    textstr = f'Euclidean Similarity: {euclidean_similarity:.4f}\nDTW Similarity: {dtw_similarity:.4f}\nPearson Correlation: {correlation:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=36,
             verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Add this code after plt.tight_layout() and before plt.savefig()

    plt.figtext(0.5, 0.01, "Figure 10: Open2Step vs. Claude3 Coherence", ha='center', fontsize=32, fontweight='bold')

    # Adjust the bottom margin to make room for the new label
    plt.subplots_adjust(bottom=0.15)

    # Update output plot path
    output_plot = os.path.join('..', 'data', 'plots', genre, f'{title_year_str}_coherence_claude3_open2step.png')
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_plot}")

    plt.close()

    print(common_df.describe())
    print(f"\nShape of common_df: {common_df.shape}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    print(traceback.format_exc())