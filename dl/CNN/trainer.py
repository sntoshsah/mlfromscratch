from cnn import ConvLayer, FlattenLayer, softmax, cross_entropy_loss
import numpy as np
from datasetup import X_data, y_data, one_hot, X_test, y_test
from graph import visualize_learning_curves

# -----------------------------
# Model Initialization
# -----------------------------
conv = ConvLayer(kernel_size=3)
flatten = FlattenLayer()

# Input: 8x8 → Conv(3x3) → 6x6 → 36 features
dense_weights = np.random.randn(36, 10) * 0.1

# -----------------------------
# Forward Pass (Reusable)
# -----------------------------
def forward(image, dense_weights):
    out_conv = conv.forward(image)
    out_relu = np.maximum(0, out_conv)
    out_flat = flatten.forward(out_relu)
    logits = np.dot(out_flat, dense_weights)
    probs = softmax(logits)
    return out_conv, out_flat, probs

# -----------------------------
# Single Training Step
# -----------------------------
def train_step(image, label_one_hot, dense_weights, lr=0.01):
    # ---- Forward ----
    out_conv, out_flat, probs = forward(image, dense_weights)

    # ---- Loss ----
    loss = cross_entropy_loss(probs, label_one_hot)

    # ---- Backward ----
    error = probs - label_one_hot                     # dL/dlogits
    grad_dense = np.dot(out_flat.T, error)

    grad_flat = np.dot(error, dense_weights.T)
    grad_relu = flatten.backward(grad_flat)
    grad_conv = grad_relu * (out_conv > 0)

    conv.backward(grad_conv, lr)

    # ---- Update Dense ----
    dense_weights -= lr * grad_dense

    return loss, dense_weights



# -----------------------------
# Training Loop
# -----------------------------
epochs = 10
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(epochs):
    total_train_loss = 0
    train_correct = 0

    # -------- Training --------
    for img, label in zip(X_data, y_data):
        label_oh = one_hot(label, depth=10)

        loss, dense_weights = train_step(
            img, label_oh, dense_weights, lr=0.01
        )
        total_train_loss += loss

        _, _, probs = forward(img, dense_weights)
        if np.argmax(probs) == label:
            train_correct += 1

    avg_train_loss = total_train_loss / len(X_data)
    train_accuracy = train_correct / len(X_data)

    # -------- Testing --------
    total_test_loss = 0
    test_correct = 0

    for img, label in zip(X_test, y_test):
        label_oh = one_hot(label, depth=10)

        _, out_flat, probs = forward(img, dense_weights)
        loss = cross_entropy_loss(probs, label_oh)
        total_test_loss += loss

        if np.argmax(probs) == label:
            test_correct += 1

    avg_test_loss = total_test_loss / len(X_test)
    test_accuracy = test_correct / len(X_test)

    # -------- Store Metrics --------
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Test Loss: {avg_test_loss:.4f} | "
        f"Test Acc: {test_accuracy:.4f}"
    )

visualize_learning_curves(
    train_accuracies,
    test_accuracies,
    train_losses,
    test_losses,
    epochs
)
