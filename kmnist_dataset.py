import numpy as np
import matplotlib.pyplot as plt

# Activation functions and derivatives
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Loss function (cross-entropy)
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / m

# Neural Network Class
class NeuralNetwork:
    def __init__(self, layer_sizes, activations, optimizer="sgd", learning_rate=0.01, momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize weights and biases
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i]) for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros((1, layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]

        # Initialize optimizer-specific variables
        self.velocity = [np.zeros_like(w) for w in self.weights]  # For momentum and Nesterov
        self.cache = [np.zeros_like(w) for w in self.weights]  # For RMSprop
        self.m_t = [np.zeros_like(w) for w in self.weights]  # For Adam
        self.v_t = [np.zeros_like(w) for w in self.weights]  # For Adam

    def forward(self, X):
        self.activations_list = [X]
        self.pre_activations = []
        for i in range(len(self.weights)):
            Z = np.dot(self.activations_list[-1], self.weights[i]) + self.biases[i]
            self.pre_activations.append(Z)
            A = relu(Z) if self.activations[i] == "relu" else softmax(Z)
            self.activations_list.append(A)
        return self.activations_list[-1]

    def backward(self, X, y_true):
        m = X.shape[0]
        gradients_w = []
        gradients_b = []

        # Output layer gradient
        dZ = self.activations_list[-1] - y_true
        dW = np.dot(self.activations_list[-2].T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        gradients_w.insert(0, dW)
        gradients_b.insert(0, db)

        # Hidden layers gradients
        for i in reversed(range(len(self.weights) - 1)):
            dA = np.dot(dZ, self.weights[i + 1].T)
            dZ = dA * relu_derivative(self.pre_activations[i])
            dW = np.dot(self.activations_list[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            gradients_w.insert(0, dW)
            gradients_b.insert(0, db)

        return gradients_w, gradients_b

    def update_weights(self, gradients_w, gradients_b, t):
        for i in range(len(self.weights)):
            if self.optimizer == "sgd":
                self.weights[i] -= self.learning_rate * gradients_w[i]
                self.biases[i] -= self.learning_rate * gradients_b[i]

            elif self.optimizer == "momentum":
                self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * gradients_w[i]
                self.weights[i] += self.velocity[i]
                self.biases[i] -= self.learning_rate * gradients_b[i]

            elif self.optimizer == "nesterov":
                prev_velocity = self.velocity[i]
                self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * gradients_w[i]
                self.weights[i] += -self.momentum * prev_velocity + (1 + self.momentum) * self.velocity[i]
                self.biases[i] -= self.learning_rate * gradients_b[i]

            elif self.optimizer == "rmsprop":
                self.cache[i] = 0.9 * self.cache[i] + 0.1 * (gradients_w[i] ** 2)
                self.weights[i] -= self.learning_rate * gradients_w[i] / (np.sqrt(self.cache[i]) + self.epsilon)
                self.biases[i] -= self.learning_rate * gradients_b[i]

            elif self.optimizer == "adam":
                self.m_t[i] = self.beta1 * self.m_t[i] + (1 - self.beta1) * gradients_w[i]
                self.v_t[i] = self.beta2 * self.v_t[i] + (1 - self.beta2) * (gradients_w[i] ** 2)
                m_t_hat = self.m_t[i] / (1 - self.beta1 ** t)
                v_t_hat = self.v_t[i] / (1 - self.beta2 ** t)
                self.weights[i] -= self.learning_rate * m_t_hat / (np.sqrt(v_t_hat) + self.epsilon)
                self.biases[i] -= self.learning_rate * gradients_b[i]

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        self.loss_history = []
        self.accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []

        for epoch in range(epochs):
            t = 1  # Adam time step
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Forward pass
                self.forward(X_batch)

                # Backward pass
                gradients_w, gradients_b = self.backward(X_batch, y_batch)

                # Update weights
                self.update_weights(gradients_w, gradients_b, t)
                t += 1

            # Compute training loss and accuracy
            y_pred_train = self.forward(X_train)
            train_loss = cross_entropy_loss(y_train, y_pred_train)
            train_predictions = np.argmax(y_pred_train, axis=1)
            train_true_labels = np.argmax(y_train, axis=1)
            train_accuracy = np.mean(train_predictions == train_true_labels)

            # Compute validation loss and accuracy
            y_pred_val = self.forward(X_val)
            val_loss = cross_entropy_loss(y_val, y_pred_val)
            val_predictions = np.argmax(y_pred_val, axis=1)
            val_true_labels = np.argmax(y_val, axis=1)
            val_accuracy = np.mean(val_predictions == val_true_labels)

            # Store metrics for plotting
            self.loss_history.append(train_loss)
            self.accuracy_history.append(train_accuracy)
            self.val_loss_history.append(val_loss)
            self.val_accuracy_history.append(val_accuracy)

            # Print metrics for every epoch
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            print("-" * 50)

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

# Load KMNIST dataset
def load_kmnist_images(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28 * 28).astype('float32')

def load_kmnist_labels(filename):
    with open(filename, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)

# Load and normalize data
x_train = load_kmnist_images("train-images-idx3-ubyte")
y_train = load_kmnist_labels("train-labels-idx1-ubyte")
x_test = load_kmnist_images("t10k-images-idx3-ubyte")
y_test = load_kmnist_labels("t10k-labels-idx1-ubyte")

# Normalize data
x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)

# One-hot encode labels
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# Split training data into training and validation sets
val_size = int(0.2 * x_train.shape[0])  # 20% validation set
X_val, y_val = x_train[:val_size], y_train[:val_size]
X_train, y_train = x_train[val_size:], y_train[val_size:]

# Define network architecture and optimizer
layer_sizes = [784, 512, 256, 10]
activations = ["relu", "relu", "softmax"]
optimizer = "adam"  # Choose from "sgd", "momentum", "nesterov", "rmsprop", "adam"

# Create and train the network
nn = NeuralNetwork(layer_sizes, activations, optimizer=optimizer, learning_rate=0.001)
nn.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=128)

# Evaluate the network on the test set
predictions = nn.predict(x_test)
accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
print(f"Test Accuracy: {accuracy:.4f}")

# Plot training and validation loss and accuracy
def plot_training_history(loss_history, accuracy_history, val_loss_history, val_accuracy_history):
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history, label='Training Accuracy')
    plt.plot(val_accuracy_history, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot training history
plot_training_history(nn.loss_history, nn.accuracy_history, nn.val_loss_history, nn.val_accuracy_history)

# Plot sample images with true and predicted labels
def plot_sample_images(x_test, y_test, predictions, num_samples=10):
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        true_label = np.argmax(y_test[i])
        predicted_label = predictions[i]
        plt.title(f"True: {true_label}\nPred: {predicted_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Plot sample images
plot_sample_images(x_test, y_test, predictions)
