import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class ARFlow:
    def __init__(self, n_vars, lr=10e-3):
        """
        :param n_vars: number of 1D variables to model
        """
        self.optimiser = tf.optimizers.Adam(learning_rate=lr)
        self.n_vars = n_vars
        self.model = self.setup_model()

    def train(self, x):
        with tf.GradientTape() as tape:
            loss = self.loss(x)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimiser.apply_gradients(zip(grads, self.model.trainable_variables))

    def setup_model(self):
        return None

    def loss(self, x):
        return None


class ARFlowModel(tf.keras.Model):
    def cdf(self, x):
        return None

    def pdf(self, x):
        return None


class ARFlowComponent(tf.keras.layers.Layer):
    def cdf(self, x):
        return None

    def pdf(self, x):
        return None


def sample_data():
    count = 100000
    rand = np.random.RandomState(0)
    a = [[-1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    b = [[1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    c = np.c_[2 * np.cos(np.linspace(0, np.pi, count // 3)),
    -np.sin(np.linspace(0, np.pi, count // 3))]
    c += rand.randn(*c.shape) * 0.2
    data_x = np.concatenate([a, b, c], axis=0)
    data_y = np.array([0] * len(a) + [1] * len(b) + [2] * len(c))
    perm = rand.permutation(len(data_x))
    return data_x[perm], data_y[perm]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x, y = sample_data()
    # plt.plot(x[:, 0], x[:, 1], "x")
    # plt.show()
