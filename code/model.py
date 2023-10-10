import numpy as np
from scipy.special import expit


class nn_one_layer:
    def __init__(self, input_size, hidden_size, output_size):

        self.W1 = 0.1 * np.random.randn(input_size, hidden_size)
        self.W2 = 0.1 * np.random.randn(hidden_size, output_size)

    def forward(self, u):
        z = np.matmul(u, self.W1)
        h = self.sigmoid(z)
        v = np.matmul(h, self.W2)
        return v, h, z

    def sigmoid(self, a):
        return expit(a)

    def sigmoid_prime(self, a):
        dsigmoid_da = self.sigmoid(a) * (1 - self.sigmoid(a))
        return dsigmoid_da

    def loss_mse(self, preds, targets):
        loss = np.sum((preds - targets) ** 2)
        return 0.5 * loss

    def loss_deriv(self, preds, targets):
        dL_dPred = preds - targets
        return dL_dPred

    def backprop(self, W1, W2, dL_dPred, U, H, Z, activate=True, prob_not_backprop=0):
        dL_dW2 = np.matmul(H.T, dL_dPred)

        if np.random.uniform() > prob_not_backprop:
            dL_dH = np.matmul(dL_dPred, W2.T)
            if activate:
                dL_dZ = np.multiply(self.sigmoid_prime(Z), dL_dH)
            else:
                dL_dZ = dL_dH
            dL_dW1 = np.matmul(U.T, dL_dZ)
        else:
            dL_dH = 1
            dL_dZ = dL_dH
            dL_dW1 = U.T

        return dL_dW1, dL_dW2


class nn_autoencoder:
    def __init__(self, input_size, hidden_size):

        self.W1 = 0.01 * np.random.randn(input_size, hidden_size)
        self.W2 = 0.01 * np.random.randn(hidden_size, input_size)

    def forward(self, u):
        z = np.matmul(u, self.W1)
        h = self.sigmoid(z)
        v = np.matmul(h, self.W2)
        return v, h, z

    def sigmoid(self, a):
        siga = 1 / (1 + np.exp(-a))
        return siga

    def sigmoid_prime(self, a):
        dsigmoid_da = self.sigmoid(a) * (1 - self.sigmoid(a))
        return dsigmoid_da

    def loss_mse(self, preds, targets):
        loss = np.sum((preds - targets) ** 2)
        return 0.5 * loss

    def loss_deriv(self, preds, targets):
        dL_dPred = preds - targets
        return dL_dPred

    def backprop(self, W1, W2, dL_dPred, U, H, Z, activate=True, prob_not_backprop=0):
        dL_dW2 = np.matmul(H.T, dL_dPred)

        dL_dH = np.matmul(dL_dPred, W2.T)
        dL_dZ = np.multiply(self.sigmoid_prime(Z), dL_dH)
        dL_dW1 = np.matmul(U.T, dL_dZ)

        return dL_dW1, dL_dW2
