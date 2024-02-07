import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import tomllib

# Load configuration from config.toml
config_path = 'config.toml'  # Path to the config.toml file

with open(config_path, 'rb') as config_file:
    config = tomllib.load(config_file)

# Retrieve configuration values
VOCAB_SIZE = config.get('VOCAB_SIZE', 10000)  # Default to 10000 if not specified
MAX_LENGTH = config.get('MAX_LENGTH', 100)    # Default to 100 if not specified

EMBEDDING_DIM = config.get('EMBEDDING_DIM', 50)  # Embedding dimension for the Embedding layer
TRAINING_DATA_PATH = './datasets/preprocessed/preproc_combined_reviews.csv'  # Example data path

# Load data
data = pd.read_csv(TRAINING_DATA_PATH)
texts = data['text']  # Assuming 'text' column contains text data
labels = data['sentiment']  # Assuming 'sentiment' column contains labels

# Tokenize the data
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<UNK>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

# Pad sequences to ensure uniform input size
data_padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')

# Split data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH),
    Dropout(0.2),
    Bidirectional(LSTM(64)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Checkpoint to save the best model
checkpoint_path = './models/best_model.h5'
model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=512, validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint])
