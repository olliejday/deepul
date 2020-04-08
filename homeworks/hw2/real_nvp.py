import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from autoregressive_flow import sample_data, DenseNN


class RealNVP:
    def __init__(self):
        self.model = self.setup_model()

    def setup_model(self):
        pass

    def train(self, x):
        pass


class RealNVPModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.affine_transf_flow = AffineTransformationFlow()

    def f_x(self, inputs):
        """
        z = f(x)
        :param inputs: (bs, 2) Xs inputs
        :return: (bs, 2) Zs output
        """
        inputs_split = tf.split(inputs, 2, -1)
        return tf.stack(self.affine_transf_flow(inputs_split), -1)

    def log_p_x(self, inputs):
        """
        computes log p(x) =   log p(f(x)) + log(det( df(x) / dx))
        :param inputs: (bs, 2) Xs inputs
        :return: (bs,) probs output
        """
        pass


class AffineTransformationFlow(tf.keras.layers.Layer):
    def __init__(self, n_units=128, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.n_units = n_units

    def build(self, input_shape):
        # build params
        # TODO: are g_thetas NNs??
        self.g_theta_scale = DenseNN(self.n_units, 1, activation="tanh")
        self.g_theta_shift = DenseNN(self.n_units, 1, activation="tanh")
        self.scale = self.add_weight("scale", shape=(1,))
        self.scale_shift = self.add_weight("scale_shift", shape=(1,))

    def call(self, inputs, **kwargs):
        """
        :param inputs: ((bs,), (bs,)) tuple of x1 and x2 inputs
        :return: ((bs,), (bs,)) tuple of z1 and z2 outputs
        """
        x1, x2 = inputs
        z1 = x1
        log_scale = self.scale * tf.tanh(self.g_theta_scale(x1)) + self.scale_shift
        z2 = tf.exp(log_scale) * x2 + self.g_theta_shift(x1)
        return z1, z2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x, y = sample_data()
    # plt.plot(x[:, 0], x[:, 1], "x")
    # plt.show()

    model = RealNVP()
    bs = 128
    for batch in np.array_split(x, int(len(x) / bs)):
        print(model.train(x))
