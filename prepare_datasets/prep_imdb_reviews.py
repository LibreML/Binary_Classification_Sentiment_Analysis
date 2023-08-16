import os
import pandas as pd
import re

# Constants
DATASET_DIRECTORY = './datasets/raw/imdb_reviews'
ALIGNED_OUTPUT_PATH = './datasets/aligned/imdb_reviews.csv'

# Function to clean text
def clean_text(text):
    # Remove everything other than a-z and A-Z
    text = re.sub(r"<br />", " ", text) 
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Initialize lists to hold reviews and their corresponding sentiments
reviews = []
sentiments = []

# Load positive reviews
pos_folder = os.path.join(DATASET_DIRECTORY, 'pos')
for filename in os.listdir(pos_folder):
    with open(os.path.join(pos_folder, filename), 'r', encoding='utf-8') as file:
        review = clean_text(file.read())
        reviews.append(review)
        sentiments.append(1)  # 1 indicates positive sentiment

# Load negative reviews
neg_folder = os.path.join(DATASET_DIRECTORY, 'neg')
for filename in os.listdir(neg_folder):
    with open(os.path.join(neg_folder, filename), 'r', encoding='utf-8') as file:
        review = clean_text(file.read())
        reviews.append(review)
        sentiments.append(0)  # 0 indicates negative sentiment

# Create a DataFrame from the lists
df = pd.DataFrame({
    'text': reviews,
    'sentiment': sentiments
})

# Shuffle the dataframe to mix positive and negative reviews
df = df.sample(frac=1).reset_index(drop=True)

# Save the DataFrame to CSV
df.to_csv(ALIGNED_OUTPUT_PATH, index=False)
print(f"Saved processed dataset to {ALIGNED_OUTPUT_PATH}")
