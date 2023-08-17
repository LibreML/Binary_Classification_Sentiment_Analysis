import csv
import matplotlib.pyplot as plt

def plot_metrics(metrics, filename):
    """Plot and save given metrics from the training history."""
    epochs = range(1, len(metrics['accuracy']) + 1)
    
    # Plot accuracy
    plt.plot(epochs, metrics['accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, metrics['val_accuracy'], 'b-', label='Validation Accuracy')
    
    # Plot loss
    plt.plot(epochs, metrics['loss'], 'ro-', label='Training Loss')
    plt.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
    
    # Plot precision
    plt.plot(epochs, metrics['precision'], 'go-', label='Training Precision')
    plt.plot(epochs, metrics['val_precision'], 'g-', label='Validation Precision')
    
    # Plot recall
    plt.plot(epochs, metrics['recall'], 'mo-', label='Training Recall')
    plt.plot(epochs, metrics['val_recall'], 'm-', label='Validation Recall')
    
    # Calculate F1-Score from Precision and Recall for both training and validation
    f1_train = [2 * (p * r) / (p + r) if p + r != 0 else 0 for p, r in zip(metrics['precision'], metrics['recall'])]
    f1_val = [2 * (p * r) / (p + r) if p + r != 0 else 0 for p, r in zip(metrics['val_precision'], metrics['val_recall'])]

    
    # Plot F1-Score
    plt.plot(epochs, f1_train, 'co-', label='Training F1-Score')
    plt.plot(epochs, f1_val, 'c-', label='Validation F1-Score')
    
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epochs')
    plt.legend()
    plt.ylim(0, 1.0)  # Set the y-axis limits, any values above 1.0 will be off the graph

    # Save the plot to a PNG file
    plt.savefig(filename)
    plt.close()


def metrics_to_csv(metrics, filename='metrics.csv'):
    """Save the training metrics to a CSV file."""
    
    # Calculate F1-Score from Precision and Recall for both training and validation
    f1_train = [2 * (p * r) / (p + r) if p + r != 0 else 0 for p, r in zip(metrics['precision'], metrics['recall'])]
    f1_val = [2 * (p * r) / (p + r) if p + r != 0 else 0 for p, r in zip(metrics['val_precision'], metrics['val_recall'])]


    metrics['f1_train'] = f1_train
    metrics['f1_val'] = f1_val
    
    # Open a file in write mode
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header (column names)
        header = list(metrics.keys())
        writer.writerow(['epoch'] + header)
        
        # Write the metric values for each epoch
        for epoch in range(1, len(metrics['accuracy']) + 1):
            row = [epoch] + [metrics[key][epoch - 1] for key in header]
            writer.writerow(row)


def metrics(logs, model_name):
    """Visualize and save training and validation metrics from the logs."""
    # Plot and save the metrics
    plot_metrics(logs, f'./metrics/training_metrics_{model_name}.png')
    
    # Save metrics to a CSV file
    metrics_to_csv(logs, f'./metrics/training_metrics_{model_name}.csv')
