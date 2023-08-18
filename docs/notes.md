# Training Notes
In this section i will discuss obervations on training.

# L2 Regularization
I increased the L2 Regularization rate from 0.01 to 0.05, and it underfitted with a validation accuracy of 50%, which is a lot worse than the 89% we got before. this is with my Bidirectional MultiLayer LSTM architecture. 

I removed the L2 Regularization, and overfitting started. so i readded it.