import re
import pandas as pd
import os

# Constants
DATASET_PATH = './datasets/raw/yelp_reviews.json'
ALIGNED_OUTPUT_PATH = './datasets/aligned/yelp_reviews.csv'
SENTIMENT_THRESHOLD = 3  # Reviews with stars greater than this threshold will be considered positive

def clean_text(text):
    """Clean the text data using regex."""
    text = re.sub(r"<br />", " ", text) 
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text

# Load the entire dataset
data = pd.read_json(DATASET_PATH, lines=True)

# Clean the text
data['text'] = data['text'].apply(clean_text)

# Convert stars to binary sentiment: 1 for positive and 0 for negative
data['sentiment'] = data['stars'].apply(lambda star: 1 if star > SENTIMENT_THRESHOLD else 0)

# Drop unnecessary columns
processed_data = data[['text', 'sentiment']]

# Shuffle the processed data using the sample function
shuffled_data = processed_data.sample(frac=1).reset_index(drop=True)

# Save the shuffled data to CSV with headers explicitly set
shuffled_data.to_csv(ALIGNED_OUTPUT_PATH, index=False, header=True)
print(f"Saved shuffled dataset to {ALIGNED_OUTPUT_PATH}")
