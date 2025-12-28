import numpy as np 


class TinyRNN:
    def __init__(self, vocab_size, hidden_size):
        # Weights
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01 # Input to Hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01 # Hidden to Hidden
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01 # Hidden to Output
        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))
        self.h = np.zeros((hidden_size, 1)) # Memory

    def forward(self, inputs):
        """
        inputs: list of one-hot encoded vectors
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(self.h)

        for t, x in enumerate(inputs):
            xs[t] = x.reshape(-1, 1)
            # Update hidden state: tanh(Wxh*x + Whh*h_prev + bh)
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)
            # Compute output: Why*h + by
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            # Softmax for probabilities
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

        return xs, hs, ps

    def backward(self, xs, hs, ps, targets, lr=0.1):
        # Initialize gradients
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])

        # Backpropagate through time (BPTT) - reverse order
        for t in reversed(range(len(xs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1 # Softmax + Cross Entropy gradient

            dWhy += np.dot(dy, hs[t].T)
            dby += dy

            dh = np.dot(self.Why.T, dy) + dhnext # Gradient flowing into h
            dhraw = (1 - hs[t] * hs[t]) * dh    # Backprop through tanh

            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)

        # Update weights
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) # Clip to prevent exploding gradients

        self.Wxh -= lr * dWxh
        self.Whh -= lr * dWhh
        self.Why -= lr * dWhy
        self.bh -= lr * dbh
        self.by -= lr * dby

# Helper functions
def softmax(x):
    exps = np.exp(x - np.max(x)) # Subtract max for numerical stability
    return exps / np.sum(exps)

def cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(y_pred))
