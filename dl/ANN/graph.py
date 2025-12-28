import matplotlib.pyplot as plt
import numpy as np

def visualize_learning_curves(train_accuracies, test_accuracies, train_losses, test_losses, epochs):
    # Calculate epochs for x-axis to match the frequency of metric storage
    epoch_points = [i for i in range(0, epochs, 50)]
    if (epochs - 1) % 50 != 0: # Add the last epoch if it wasn't a multiple of 50
        epoch_points.append(epochs - 1)

    plt.figure(figsize=(12, 5))

    # Plot Training and Testing Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epoch_points, train_accuracies, label='Training Accuracy')
    plt.plot(epoch_points, test_accuracies, label='Testing Accuracy')
    plt.title('ANN Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Plot Training and Testing Loss
    plt.subplot(1, 2, 2)
    plt.plot(epoch_points, train_losses, label='Training Loss')
    plt.plot(epoch_points, test_losses, label='Testing Loss')
    plt.title('ANN Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('ann_learning_curves.png')

    print("Learning curves visualized.")
