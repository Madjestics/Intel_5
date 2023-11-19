import numpy as np
import random

def relu(t):
    return np.maximum(t, 0)

def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)

def relu_deriv(t):
    return (t >= 0).astype(float)

class NeuralNetwork:
    def __init__(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

    def feedforward(self, inputs, weights, bias):
        # Умножаем входы на веса, прибавляем порог, затем используем функцию активации
        return inputs @ weights + bias

    def predict(self, x):
        t1 = self.feedforward(x, self.W1, self.b1)
        h1 = relu(t1)
        t2 = self.feedforward(h1, self.W2, self.b2)
        z = softmax(t2)
        return z

    def to_full_output(self, y, num_classes):
        y_full = np.zeros((1, num_classes))
        y_full[0, y] = 1
        return y_full

    def calc_accuracy(self, dataset):
        correct = 0
        for x, y in dataset:
            z = self.predict(x)
            y_pred = np.argmax(z)
            if y_pred == y:
                correct += 1
        acc = correct / len(dataset)
        return acc

    def train(self, dataset):
        NUM_EPOCHS = 200
        ALPHA = 0.05
        for ep in range(NUM_EPOCHS):
            random.shuffle(dataset)
            for i in range(len(dataset)):
                x, y = dataset[i]

                # Forward
                t1 = self.feedforward(x, self.W1, self.b1)
                h1 = relu(t1)
                t2 = self.feedforward(h1, self.W2, self.b2)
                z = softmax(t2)

                # Backward
                y_full = self.to_full_output(y, len(self.b2[0]))
                dE_dt2 = z - y_full
                dE_dW2 = h1.T @ dE_dt2
                dE_db2 = dE_dt2
                dE_dh1 = dE_dt2 @ self.W2.T
                dE_dt1 = dE_dh1 * relu_deriv(t1)
                dE_dW1 = x.T @ dE_dt1
                dE_db1 = dE_dt1

                # Update
                self.W1 = self.W1 - ALPHA * dE_dW1
                self.b1 = self.b1 - ALPHA * dE_db1
                self.W2 = self.W2 - ALPHA * dE_dW2
                self.b2 = self.b2 - ALPHA * dE_db2

            print(ep)
            accuracy = self.calc_accuracy(dataset)
            print("Accuracy:", accuracy)
            if 1 - accuracy < 0.01:
                break