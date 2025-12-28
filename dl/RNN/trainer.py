from rnn import TinyRNN, cross_entropy_loss
from datasetup import char_to_ix, ix_to_char, inputs, target_indices, vocab_size
import numpy as np
from graph import visualizing_learning_curves

rnn = TinyRNN(vocab_size=4, hidden_size=10)
#  Lists to store metrics for plotting
epoch_losses = []
epoch_accuracies = []

print("Starting RNN training...")

# Simple Training Loop
for epoch in range(1000):
    xs, hs, ps = rnn.forward(inputs)

    current_loss = 0
    correct_predictions = 0
    for t in range(len(xs)):
        # Calculate loss
        y_true_one_hot = np.zeros((vocab_size, 1))
        y_true_one_hot[target_indices[t]] = 1
        current_loss += cross_entropy_loss(ps[t], y_true_one_hot)

        # Calculate accuracy
        if np.argmax(ps[t]) == target_indices[t]:
            correct_predictions += 1

    # Average loss over time steps
    avg_loss = current_loss / len(xs)
    accuracy = (correct_predictions / len(xs)) * 100

    epoch_losses.append(avg_loss)
    epoch_accuracies.append(accuracy)

    rnn.backward(xs, hs, ps, target_indices)

    if epoch % 100 == 0 or epoch == 999:
        prediction = "".join([ix_to_char[np.argmax(ps[t])] for t in range(len(ps))])
        print(f"Epoch {epoch}: Predicted '{prediction}', Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

print("Training complete.")

visualizing_learning_curves(epoch_accuracies=epoch_accuracies, epoch_losses=epoch_losses)
