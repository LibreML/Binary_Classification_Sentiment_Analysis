import tkinter as tk
from random import choice

class SentimentApp:
    def __init__(self, master):
        # Set up main window properties
        self.master = master
        self.master.title("Sentiment Analysis (Binary Classification)")

        # Emoji display label
        self.emoji_label = tk.Label(master, font=("Arial", 40))
        self.emoji_label.pack(pady=20)

        # Text sentiment display label
        self.sentiment_label = tk.Label(master, font=("Arial", 20))
        self.sentiment_label.pack(pady=10)

        # Multiline textbox with 250 character limit
        self.textbox = tk.Text(master, height=10, width=50, wrap=tk.WORD)
        self.textbox.pack(pady=20)
        self.textbox.bind("<Key>", self.limit_characters)

        # Counter label
        self.counter_label = tk.Label(master, text="0/250 characters", font=("Arial", 12))
        self.counter_label.pack()

        # Predict button
        self.predict_btn = tk.Button(master, text="Predict", command=self.update_sentiment)
        self.predict_btn.pack(pady=20)

    def update_sentiment(self):
        sentiment = choice(["POSITIVE", "NEGATIVE"])
        if sentiment == "POSITIVE":
            self.emoji_label.config(text="😊")
            self.sentiment_label.config(text="This text is POSITIVE")
        else:
            self.emoji_label.config(text="☹")
            self.sentiment_label.config(text="This text is NEGATIVE")

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
