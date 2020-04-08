import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from autoregressive_flow import sample_data, DenseNN


class RealNVP:
    """
    2 variable, MLP version of RealNVP
    """
    def __init__(self):
        self.model = self.setup_model()

    def setup_model(self):
        return RealNVPModel()

    def train(self, x):
        pass


class RealNVPModel(tf.keras.Model):
    """
    Model for RealNVP flow
    For 2 variables using MLPs (FF-NN)
    """
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
        # for z prior (2 variables)
        self.z_prior = tfp.distributions.Normal([0, 0], [1, 1])

    def f_x(self, inputs):
        """
        computes z = f(x)
        :param inputs: (bs, 2) Xs inputs
        :return: (bs, 2) Zs output
        """
        x1, x2 = tf.split(inputs, 2, -1)
        z1 = x1
        log_scale = self.scale * tf.tanh(self.g_theta_scale(x1)) + self.scale_shift
        z2 = tf.exp(log_scale) * x2 + self.g_theta_shift(x1)
        return tf.stack([z1, z2], -1)

    def log_p_x(self, inputs):
        """
        computes log p(x) =   log p(f(x)) + log(det( df(x) / dx))
        :param inputs: (bs, 2) Xs inputs
        :return: (bs,) probs output
        """
        # z prior prob
        log_p_z = self.z_prior.log_prob(self.f_x(inputs))
        x1, _ = inputs
        log_scale = self.scale * tf.tanh(self.g_theta_scale(x1)) + self.scale_shift
        # since jac is triangular, det of jac is prod of diag
        # but in this 2 variable case the first diagonal is 1
        df_dx = tf.exp(log_scale)
        log_det_jac = tf.math.log(df_dx)
        return log_p_z + log_det_jac


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x, y = sample_data()
    # plt.plot(x[:, 0], x[:, 1], "x")
    # plt.show()

    model = RealNVP()
    bs = 128
    for batch in np.array_split(x, int(len(x) / bs)):
        print(model.train(x))
