import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from dtaidistance import dtw
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import MinMaxScaler

def winsorize_series(series, limits=(0.05, 0.05)):
    winsorized = stats.mstats.winsorize(series, limits=limits)
    return np.ma.filled(winsorized, np.nan)

def min_max_scale(series):
    scaler = MinMaxScaler()
    non_nan = series[~np.isnan(series)]
    scaled = scaler.fit_transform(non_nan.values.reshape(-1, 1)).flatten()
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
    diff = series1 - series2
    return np.sqrt(np.sum(diff**2) / len(diff)) / (np.std(series1) + np.std(series2))

def normalized_dtw_distance(series1, series2):
    s1, s2 = np.array(series1), np.array(series2)
    raw_distance = dtw.distance(s1, s2)
    max_distance = dtw.distance(np.ones_like(s1), np.zeros_like(s2))
    return raw_distance / max_distance

title_year_str = "royal_wedding_1951"
genre = 'musical'

input_ossmodels_path = os.path.join("..", "data", "plots", genre, f"{title_year_str}_normalized_sentiments.csv")

sentiment_df = pd.read_csv(input_ossmodels_path)
models_columns = ['vader_norm', 'textblob_norm', 'llama3_norm']

print(sentiment_df.info())
print(sentiment_df.describe())

# Process each model's sentiment
for col in models_columns:
    sentiment_df[f'{col}_winsorized'] = winsorize_series(sentiment_df[col])
    sentiment_df[f'{col}_scaled'] = min_max_scale(sentiment_df[f'{col}_winsorized'])

# Calculate mean of scaled sentiments
sentiment_df['mean_scaled'] = sentiment_df[[f'{col}_scaled' for col in models_columns]].mean(axis=1)

# Remove outliers and normalize to common range
for col in models_columns + ['mean']:
    sentiment_df[f'{col}_no_outliers'] = remove_outliers(sentiment_df[f'{col}_scaled'])

overall_min = sentiment_df[[f'{col}_no_outliers' for col in models_columns + ['mean']]].min().min()
overall_max = sentiment_df[[f'{col}_no_outliers' for col in models_columns + ['mean']]].max().max()

for col in models_columns + ['mean']:
    sentiment_df[f'{col}_norm'] = normalize_to_range(sentiment_df[f'{col}_no_outliers'], overall_min, overall_max)
    sentiment_df[f'{col}_smooth'] = lowess(sentiment_df[f'{col}_norm'], sentiment_df.index, frac=0.15, it=0)[:, 1]
    sentiment_df[f'{col}_final'] = z_score_normalize(sentiment_df[f'{col}_smooth'])

# Compute similarity metrics
euclidean_similarities = []
dtw_similarities = []

for col in models_columns:
    euclidean_similarities.append(1 - normalized_euclidean_distance(sentiment_df[f'{col}_final'], sentiment_df['mean_final']))
    dtw_similarities.append(1 - normalized_dtw_distance(sentiment_df[f'{col}_final'], sentiment_df['mean_final']))

euclidean_similarity = np.mean(euclidean_similarities)
dtw_similarity = np.mean(dtw_similarities)

print(f"Euclidean Similarity: {euclidean_similarity:.4f}")
print(f"DTW Similarity: {dtw_similarity:.4f}")

# Plot the sentiment arcs and save to file
plt.figure(figsize=(16, 12))
for col in models_columns:
    plt.plot(sentiment_df.index, sentiment_df[f'{col}_final'], label=col, linewidth=3)
plt.plot(sentiment_df.index, sentiment_df['mean_final'], label='Mean', linewidth=4, color='black')

plt.legend(fontsize=24)
plt.title('Royal Wedding (1951) Video Sentiment Arcs\nOpen Model Coherence', fontsize=32)
plt.xlabel('Time', fontsize=24)
plt.ylabel('Z-Score Normalized Sentiment', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

textstr = f'Euclidean Similarity: {euclidean_similarity:.4f}\nDTW Similarity: {dtw_similarity:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=36,
         verticalalignment='top', bbox=props)

plt.tight_layout()

output_dir = os.path.join("..", "data", "plots", genre)
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{title_year_str}_intermodel_coherence.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_file}")

plt.close()