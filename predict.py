import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pretrained model
model_path = './models/sentiment_model_BiLSTM_5e_89acc_0.27loss_16_embedding.keras'
model = load_model(model_path)

# Load the tokenizer
with open('./tokenizers/sentiment_analysis/lstm_tokenizer_100k.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def encode_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequences, maxlen=250, padding='post', truncating='post')

text = "The movie was great! But i did not like the plot twist.".lower()

# Convert the text into tokenized and padded sequence
encoded_text = encode_text(text)

# Make predictions
prediction = model.predict(encoded_text)

sentiment = "positive" if prediction[0][0] >= 0.5 else "negative"

print(f"Sentiment: {sentiment}, Probability: {prediction[0][0]:.2f}")
