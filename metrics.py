import csv
import matplotlib.pyplot as plt

def plot_metrics(history, filename):
    """Plot and save given metrics from the training history."""
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    # Plot accuracy
    plt.plot(epochs, history.history['accuracy'], 'bo', label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'b', label='Validation Accuracy')
    
    # Plot loss
    plt.plot(epochs, history.history['loss'], 'ro', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r', label='Validation Loss')
    
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epochs')
    plt.legend()

    # Save the plot to a PNG file
    plt.savefig(filename)
    plt.close()

def metrics_to_csv(history, filename='metrics.csv'):
    """Save the training history metrics to a CSV file."""
    
    # Extract metrics from the history object
    metrics_data = history.history

    # Open a file in write mode
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header (column names)
        header = ['epoch'] + list(metrics_data.keys())
        writer.writerow(header)
        
        # Write the metric values for each epoch
        for epoch in range(1, len(metrics_data['accuracy']) + 1):
            row = [epoch] + [metrics_data[key][epoch - 1] for key in metrics_data.keys()]
            writer.writerow(row)

def metrics(history, model_name):
    """Visualize and save training and validation metrics from the training history."""
    # Plot and save the metrics
    plot_metrics(history, f'./metrics/training_metrics_{model_name}.png')
    
    # Save metrics to a CSV file
    metrics_to_csv(history, f'./metrics/training_metrics_{model_name}.csv')
