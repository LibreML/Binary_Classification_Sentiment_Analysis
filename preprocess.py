import pandas as pd
import re
import tomllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Define constants and paths
INPUT_FILES = ["combined_reviews_500000.csv"]
DATASET_DIRECTORY = "./datasets/aligned/"
OUTPUT_FILE = './datasets/preprocessed/preproc_combined_reviews_500k.csv'
TOKENIZER_FILE = './tokenizers/MLBiLSTM_tokenizer_500k.pickle'
CONFIG_FILE = './config.toml'  # Path to your config.toml file

# Text cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"<br />", " ", text)  # Remove <br /> tags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation and numbers
    return text

# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame(columns=['text', 'sentiment'])

# Load and combine datasets
for filename in INPUT_FILES:
    path = DATASET_DIRECTORY + filename
    data_chunk = pd.read_csv(path)
    combined_data = pd.concat([combined_data, data_chunk], ignore_index=True)

# Shuffle the combined data
combined_data = combined_data.sample(frac=1).reset_index(drop=True)

# Clean the reviews
combined_data['text'] = combined_data['text'].apply(clean_text)

# Tokenize the reviews
tokenizer = Tokenizer(oov_token="<UNK>")
tokenizer.fit_on_texts(combined_data['text'])
sequences = tokenizer.texts_to_sequences(combined_data['text'])

# Compute the average sequence length
max_length = 250 #int(sum(map(len, sequences)) / len(sequences))

# Load configuration from config.toml file
with open(CONFIG_FILE, 'rb') as config_file:
    config = tomllib.load(config_file)

# Update the configuration with the computed average length
config['MAX_LENGTH'] = max_length

# Write the updated configuration back to the config.toml file
with open(CONFIG_FILE, 'w') as config_file:
    for key, value in config.items():
        config_file.write(f"{key} = {value}\n")

# Pad the sequences using the updated average length
padded_data = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Convert the preprocessed data to a DataFrame
preprocessed_data = pd.DataFrame({
    'text': [' '.join(map(str, seq)) for seq in padded_data],
    'sentiment': combined_data['sentiment']
})

# Save the preprocessed data to a CSV file
preprocessed_data.to_csv(OUTPUT_FILE, index=False)

# Save the tokenizer
with open(TOKENIZER_FILE, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)