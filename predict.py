import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import load_config

# Load configuration from TOML
VOCAB_SIZE, MAX_LENGTH = load_config()

def load_resources(model_path, tokenizer_path):
    # Load the pretrained model
    model = load_model(model_path)

    # Load the tokenizer
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    return model, tokenizer

def encode_text(text, tokenizer):
    sequences = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

def predict(text, model_path, tokenizer_path):
    # Load model and tokenizer
    model, tokenizer = load_resources(model_path, tokenizer_path)
    
    # Convert the text into tokenized and padded sequence
    encoded_text = encode_text(text, tokenizer)

    # Make predictions
    prediction = model.predict(encoded_text)

    sentiment = 1 if prediction[0][0] >= 0.5 else 0

    return sentiment, int(float(f"{prediction[0][0]:.2f}")*100)

# Sample usage
if __name__ == "__main__":
    result = predict("The movie was great! But I did not like the plot twist.", './models/sentiment_model_BiLSTM_5e_89acc_0.27loss_16_embedding.keras', './tokenizers/sentiment_analysis/lstm_tokenizer_100k.pickle')
    print(result)
