import matplotlib.pyplot as plt

def visualize_learning_curves(
    train_accuracies,
    test_accuracies,
    train_losses,
    test_losses,
    epochs
):
    epoch_points = list(range(epochs))

    plt.figure(figsize=(12, 5))

    # ---- Accuracy ----
    plt.subplot(1, 2, 1)
    plt.plot(epoch_points, train_accuracies, label='Training Accuracy')
    plt.plot(epoch_points, test_accuracies, label='Testing Accuracy')
    plt.title('CNN Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # ---- Loss ----
    plt.subplot(1, 2, 2)
    plt.plot(epoch_points, train_losses, label='Training Loss')
    plt.plot(epoch_points, test_losses, label='Testing Loss')
    plt.title('CNN Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("cnn_learning_curves.png")
    print("Learning curves Saved.")
