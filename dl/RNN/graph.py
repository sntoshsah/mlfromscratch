import matplotlib.pyplot as plt

def visualizing_learning_curves(epoch_losses, epoch_accuracies):


    # Plotting the learning curves
    plt.figure(figsize=(12, 5))

    # plt.subplot(1, 2, 1)
    plt.plot(epoch_losses)
    plt.title('RNN Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rnn_learning_curves.png")
    print("RNN learning curves visualized.")