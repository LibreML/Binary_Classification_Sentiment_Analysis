import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Precision, Recall
from metrics import metrics
from config import load_config
import time

# Load GloVe embeddings
def load_glove_embeddings(filename):
    embeddings = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

glove_embeddings = load_glove_embeddings('./datasets/raw/glove.840B.300d.txt')

# Custom callback for saving metrics after each epoch
class MetricsCheckpoint(Callback):
    def __init__(self, model_name):
        super(MetricsCheckpoint, self).__init__()
        self.model_name = model_name
        self.filename = ""
        self.metrics = {}
    
    def on_epoch_end(self, epoch, logs=None):
        self.filename = f"{time.time()}_{self.model_name}"
        for key in logs.keys():
            if not self.metrics.get(key, None):
                self.metrics[key] = []
            self.metrics[key].append(logs[key])
        epoch_model_name = f"{self.filename}_epoch_{epoch+1}"
        metrics(self.metrics, epoch_model_name)

# Load configuration from TOML
VOCAB_SIZE, MAX_LENGTH = load_config()

# Load preprocessed data
data_path = './datasets/preprocessed/preproc_combined_reviews_100k.csv'
data = pd.read_csv(data_path)

# Tokenize the data
tokenizer = Tokenizer(oov_token="<UNK>")
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])

# Create GloVe Embedding Matrix
EMBEDDING_DIM = 300
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    embedding_vector = glove_embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Split data into training and test sets
data_train, data_test, label_train, label_test = train_test_split(
    data['text'].apply(lambda x: list(map(int, x.split()))), 
    data['sentiment'], 
    test_size=0.2, 
    random_state=42
)

# Convert data into a format suitable for training
data_train = pad_sequences(data_train, padding='post', maxlen=MAX_LENGTH)
data_test = pad_sequences(data_test, padding='post', maxlen=MAX_LENGTH)

# Create the model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1,
              output_dim=EMBEDDING_DIM,
              weights=[embedding_matrix],
              input_length=MAX_LENGTH,
              trainable=False),
    Dropout(0.5),
    Bidirectional(LSTM(32, return_sequences=True, recurrent_dropout=0.5, kernel_regularizer=regularizers.l2(0.01))),
    Dropout(0.5),
    Bidirectional(LSTM(32, recurrent_dropout=0.5, kernel_regularizer=regularizers.l2(0.01))),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Metrics
precision = Precision(name='precision')
recall = Recall(name='recall')

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', precision, recall])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Model Checkpoint
checkpoint_path = './models/checkpoints/Multilayer_Bidirectional_LSTM_ep{epoch:03d}_acc{accuracy:.4f}_loss{loss:.4f}_valAcc{val_accuracy:.4f}_valLoss{val_loss:.4f}_dataset_100k.h5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_format='h5')

# Add MetricsCheckpoint callback
metrics_checkpoint = MetricsCheckpoint('MLBiLSTM')

# Train the model with the added ModelCheckpoint and MetricsCheckpoint callbacks
history = model.fit(data_train, label_train, epochs=100, batch_size=512, validation_split=0.2, callbacks=[early_stopping, checkpoint, metrics_checkpoint])

# Get the newest checkpointed model file
list_of_files = glob.glob('./models/checkpoints/*')
newest_file = max(list_of_files, key=os.path.getctime)

# Load the best model from the newest checkpoint
best_model = tf.keras.models.load_model(newest_file)

# Evaluate the best model on the test set
test_loss, test_accuracy, test_precision, test_recall = best_model.evaluate(data_test, label_test)

# Save evaluated data into a CSV file
evaluation_data = {
    'loss': [test_loss],
    'accuracy': [test_accuracy],
    'precision': [test_precision],
    'recall': [test_recall]
}
df = pd.DataFrame(evaluation_data)
df.to_csv('./metrics/evaluate_model.csv', index=False)
