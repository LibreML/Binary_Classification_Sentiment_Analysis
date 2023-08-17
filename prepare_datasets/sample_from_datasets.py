import pandas as pd
import os

# Constants
DATASETS_DIR = "./datasets/aligned/"
TARGET_SAMPLE_SIZE = 500_000  # or whatever size you desire

# Manually define the list of dataset files
dataset_files = ['imdb_reviews.csv', 'yelp_reviews.csv']  # add other filenames as needed

def load_and_sample_data(file_path, sample_size, random_state=42):
    """Load and sample data from the given file path."""
    data = pd.read_csv(file_path, usecols=['text', 'sentiment'])
    
    # Ensure 'text' column contains strings and 'sentiment' contains integers
    data = data.dropna()
    data = data[data['text'].apply(lambda x: isinstance(x, str))]
    data = data[data['sentiment'].apply(lambda x: isinstance(x, int))]
    
    # Shuffle the data
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Split data into positive and negative sentiments
    pos_data = data[data['sentiment'] == 1]
    neg_data = data[data['sentiment'] == 0]
    
    # Sample equal amounts from both positive and negative data
    num_samples_each = sample_size // 2
    pos_sample = pos_data.head(num_samples_each)
    neg_sample = neg_data.head(num_samples_each)
    
    return pd.concat([pos_sample, neg_sample], axis=0).sample(frac=1, random_state=random_state)

all_samples = []

# Calculate the samples per dataset
samples_per_dataset = TARGET_SAMPLE_SIZE // len(dataset_files)

# Load and sample each dataset
for i, dataset_file in enumerate(dataset_files):
    # If it's the last dataset, fill remaining samples
    if i == len(dataset_files) - 1:
        current_sample_size = TARGET_SAMPLE_SIZE - sum([len(sample) for sample in all_samples])
    else:
        current_sample_size = samples_per_dataset

    sampled_data = load_and_sample_data(os.path.join(DATASETS_DIR, dataset_file), current_sample_size)
    all_samples.append(sampled_data)

# Concatenate all the sampled data
combined_reviews = pd.concat(all_samples, axis=0).reset_index(drop=True)

# Save to a new CSV
output_path = os.path.join(DATASETS_DIR, f"combined_reviews_{TARGET_SAMPLE_SIZE}.csv")
combined_reviews.to_csv(output_path, index=False)
print(f"Saved combined dataset to {output_path}")
