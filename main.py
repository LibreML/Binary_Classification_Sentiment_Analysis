import tkinter as tk
from tkinter import filedialog
from predict import predict

class SentimentApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Sentiment Analysis (Binary Classification)")
        
        # Initial page for file selection
        self.init_page()

    def init_page(self):
        self.model_label = tk.Label(self.master, text="Model File:")
        self.model_label.pack(pady=10)

        self.model_btn = tk.Button(self.master, text="Choose Model File", command=self.load_model)
        self.model_btn.pack(pady=10)

        self.tokenizer_label = tk.Label(self.master, text="Tokenizer File:")
        self.tokenizer_label.pack(pady=10)

        self.tokenizer_btn = tk.Button(self.master, text="Choose Tokenizer File", command=self.load_tokenizer)
        self.tokenizer_btn.pack(pady=10)

    def load_model(self):
        self.model_path = filedialog.askopenfilename()
        self.model_label.config(text=f"Model File: {self.model_path}")
        self.check_files_selected()

    def load_tokenizer(self):
        self.tokenizer_path = filedialog.askopenfilename()
        self.tokenizer_label.config(text=f"Tokenizer File: {self.tokenizer_path}")
        self.check_files_selected()

    def check_files_selected(self):
        if hasattr(self, 'model_path') and hasattr(self, 'tokenizer_path'):
            for widget in self.master.winfo_children():
                widget.destroy()
            self.setup_predict_page()

    def setup_predict_page(self):
        # Emoji display label
        self.emoji_label = tk.Label(self.master, font=("Arial", 40))
        self.emoji_label.pack(pady=20)

        # Text sentiment display label
        self.sentiment_label = tk.Label(self.master, font=("Arial", 20))
        self.sentiment_label.pack(pady=10)

        # Multiline textbox with 250 character limit
        self.textbox = tk.Text(self.master, height=10, width=50, wrap=tk.WORD)
        self.textbox.pack(pady=20)
        self.textbox.bind("<Key>", self.limit_characters)

        # Counter label
        self.counter_label = tk.Label(self.master, text="0/250 characters", font=("Arial", 12))
        self.counter_label.pack()

        # Predict button
        self.predict_btn = tk.Button(self.master, text="Predict", command=self.update_sentiment)
        self.predict_btn.pack(pady=20)

    def update_sentiment(self):
        text_input = self.textbox.get(1.0, tk.END).strip()
        sentiment, probability = predict(text_input, self.model_path, self.tokenizer_path)
        if sentiment == 1:
            self.emoji_label.config(text="😊")
            self.sentiment_label.config(text=f"This text is POSITIVE")
        else:
            self.emoji_label.config(text="☹")
            self.sentiment_label.config(text=f"This text is NEGATIVE")

    def limit_characters(self, event):
        # Limit the characters to 250 and allow only backspace and delete after that
        if len(self.textbox.get(1.0, tk.END)) - 1 > 250:
            if event.keysym != "BackSpace" and event.keysym != "Delete":
                return "break"

        # Update character counter
        text_length = len(self.textbox.get(1.0, tk.END)) - 1  # Subtract 1 for the newline character
        self.counter_label.config(text=f"{text_length}/250 characters")

if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentApp(root)
    root.mainloop()
