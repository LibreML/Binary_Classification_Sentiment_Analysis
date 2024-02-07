# Training Notes
In this section i will discuss obervations on training.

# L2 Regularization
I increased the L2 Regularization rate from 0.01 to 0.05, and it underfitted with a validation accuracy of 50%, which is a lot worse than the 89% we got before. this is with my Bidirectional MultiLayer LSTM architecture. 

I removed the L2 Regularization, and overfitting started. so i readded it.

# Changing Max Length

## Model with MAX_LENGTH set to the longest review
Low Accuracy: Your model seems to be struggling with distinguishing between positive and negative sentiments, hovering around 50% accuracy. This is basically equivalent to random guessing.

Overfitting: The training accuracy and validation accuracy are very close, which could be a sign that the model is not learning meaningful patterns from the data.

## Model with MAX_LENGTH set to 300
Improved Accuracy: The model's accuracy has improved significantly compared to the one with MAX_LENGTH set to the longest review.

F1 Score: The F1 score also shows improvement, indicating a better balance between precision and recall.

## Model with MAX_LENGTH set to 250

Good Performance: The model seems to perform well with decent accuracy and F1 scores.
Early Stopping: The model seems to stabilize in terms of accuracy and loss after a certain number of epochs, indicating early stopping could be useful.

### Impact of Sequence Length on Model Performance
Padding: Using the longest sequence as MAX_LENGTH means most other sequences will have to be padded, which might make the model focus on the padding tokens rather than the actual words.

Computational Cost: The longer the sequences, the more computational resources required.

Memory: If the longest review is significantly longer than the average, it may require a lot of memory.