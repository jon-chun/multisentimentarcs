import os
import pandas as pd
import numpy as np
from io import StringIO
from dtaidistance import dtw

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

input_ossmodels_path = os.path.join("..", "data", "plots","musical","royal_wedding_1951_normalized_sentiments.csv")
input_claude3_path = os.path.join("..", "data", "plots_claude3","sentiments_keyframes_royal_wedding_1951_claude-3-opus-20240229.csv")


sentiment_df = pd.read_csv(input_ossmodels_path)
models_columns = ['vader_norm', 'textblob_norm', 'llama3_norm']

claude3_df = pd.read_csv(input_claude3_path, encoding='utf-8')
print(claude3_df.head())
print(claude3_df.info())
print(claude3_df.describe())


# Interpolate considering up to 2 adjacent rows
claude3_df['sentiment'] = claude3_df['sentiment'].interpolate(method='linear', limit=2)

print(claude3_df.head())
print(claude3_df.info())
print(claude3_df.describe())

sentiment_df['claude3'] = claude3_df['sentiment']
print(sentiment_df.info())
print(sentiment_df.describe())

# Read the data into a DataFrame
data = """
text,vader,textblob,llama3,vader_norm,textblob_norm,llama3_norm
"The image is a film poster for the movie ""Art Directors"". The poster prominently features a list of names, presumably the directors involved in the film. The text on the poster is black and white, giving it a classic and timeless feel. The layout of the poster is such that the list of names takes up most of the space, indicating their importance. The background of the poster is white, which contrasts with the black text and makes the information stand out. The overall design of the poster suggests a sense of professionalism and seriousness about the film's production.",0.5719,0.0925925925925926,0.5,0.1299434484023996,0.1903655865802074,0.10938673349545565
"The image presents a striking contrast between the black background and the gray foreground. The gray foreground is slightly blurred, adding depth to the image. On the right side of this gray foreground, there's a small white square. This square contains a black and white image of a person's face. The person in the image has a neutral expression, their eyes are closed, suggesting a moment of contemplation or rest. The overall composition is simple yet evocative, with the stark contrast between the black background and the gray foreground drawing attention to the small white square and the image within it.",0.0,-0.0453968253968253,0.7,0.11657531722250909,0.1825770377672791,0.12815899265502648
"""

df = pd.read_csv(StringIO(data))

# 1. Create a new time series mean_norm based on the average of the 3 normalized timeseries columns
df['mean_norm'] = df[['vader_norm', 'textblob_norm', 'llama3_norm']].mean(axis=1)

# 2. Compute absolute coherence metric
def euclidean_distance(row):
    return np.sqrt(((row['vader_norm'] - row['mean_norm'])**2 +
                    (row['textblob_norm'] - row['mean_norm'])**2 +
                    (row['llama3_norm'] - row['mean_norm'])**2) / 3)

df['euclidean_distance'] = df.apply(euclidean_distance, axis=1)
absolute_coherence = df['euclidean_distance'].mean()

# 3. Compute relative shape similarity metric using DTW
def dtw_distance(series1, series2):
    return dtw.distance(series1.values, series2.values)

dtw_distances = [
    dtw_distance(df['vader_norm'], df['mean_norm']),
    dtw_distance(df['textblob_norm'], df['mean_norm']),
    dtw_distance(df['llama3_norm'], df['mean_norm'])
]

relative_shape_similarity = np.mean(dtw_distances)

print(f"Absolute Coherence Metric: {absolute_coherence}")
print(f"Relative Shape Similarity Metric: {relative_shape_similarity}")

# Display the resulting DataFrame
print("\nResulting DataFrame:")
print(df)








import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

def z_score_normalize(df, columns):
    """
    Normalize specified columns using z-score normalization.
    """
    return df[columns].apply(stats.zscore)

def apply_lowess(data, frac=0.15):
    """
    Apply LOWESS smoothing to the data.
    """
    y_smooth = lowess(data, np.arange(len(data)), frac=frac)[:, 1]
    return y_smooth

def plot_sentiment_analysis(sentiment_df, columns, output_file='royal_wedding_sentiment_analysis.png'):
    # Z-score normalization
    normalized_df = z_score_normalize(sentiment_df, columns)
    
    # Set up the plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 8))
    
    # Plotting with LOWESS smoothing
    for column in columns:
        smoothed_data = apply_lowess(normalized_df[column])
        sns.lineplot(x=normalized_df.index, y=smoothed_data, label=column)
    
    plt.title('Royal Wedding (1951) Video Sentiment\nLOWESS Smoothing (frac=0.15)', fontsize=16)
    plt.xlabel('Scene', fontsize=12)
    plt.ylabel('Normalized Sentiment Score', fontsize=12)
    plt.legend(fontsize=10)
    
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Plot saved as {output_file}")

# Columns to plot
columns_to_plot = ['vader_norm', 'textblob_norm', 'llama3_norm', 'claude3']

# Call the function to create and save the plot
plot_sentiment_analysis(sentiment_df, columns_to_plot)




