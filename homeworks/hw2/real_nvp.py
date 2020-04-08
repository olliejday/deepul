import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from autoregressive_flow import sample_data, DenseNN


class RealNVP:
    """
    2 variable, MLP version of RealNVP
    """
    def __init__(self, lr=0.01):
        self.model = self.setup_model()
        self.optimiser = tf.optimizers.Adam(learning_rate=lr)
        self.n_vars = 2

    def setup_model(self):
        return RealNVPModel()

    def train(self, x):
        """
        Run training step
        Returns loss for batch (1,)
        """
        with tf.GradientTape() as tape:
            loss = self.loss(x)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimiser.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def loss(self, x):
        """
        Returns loss for batch (1,) in nats / dim
        """
        log_p_x = self.log_p_x(x)
        return - tf.reduce_mean(log_p_x) / self.n_vars

    def log_p_x(self, x):
        """
        Returns log (joint) prob of given xs (bs,)
        """
        x = tf.cast(x, tf.float32)
        return self.model.log_p_x(x)

    def f_x(self, x):
        """
        Returns z values for given xs (bs, n_vars)
        """
        x = tf.cast(x, tf.float32)
        return self.model.f_x(x)


class RealNVPModel(tf.keras.Model):
    """
    Model for RealNVP flow
    For 2 variables using MLPs (FF-NN)
    """
    def __init__(self, n_units=128, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.n_units = n_units
        self.n_vars = 2  # 2 variable case
        self.setup()

    def setup(self):
        """
        Build with no input shape
        """
        # build params
        # TODO: should these be NNs or ??
        self.g_theta_scale = DenseNN(self.n_units, 1, activation="tanh")  # TODO: relu?
        self.g_theta_shift = DenseNN(self.n_units, 1, activation="tanh")
        self.scale = self.add_weight("scale", shape=(1,))
        self.scale_shift = self.add_weight("scale_shift", shape=(1,))
        # for z prior standard normal
        self.z_prior = tfp.distributions.Normal(tf.zeros(self.n_vars), tf.ones(self.n_vars))

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
        return tf.concat([z1, z2], -1)

    def log_p_x(self, inputs):
        """
        computes log p(x) =   log p(f(x)) + log(det( df(x) / dx))
        :param inputs: (bs, n_vars) Xs inputs
        :return: (bs,) joint probs output
        """
        # z prior prob
        log_p_z = self.z_prior.log_prob(self.f_x(inputs))

        x1, x2 = tf.split(inputs, 2, -1)
        log_scale = self.scale * tf.tanh(self.g_theta_scale(x1)) + self.scale_shift
        # since jac is triangular, det of jac is prod of diag
        # but in this 2 variable case the first diagonal is 1
        df_dx = tf.exp(log_scale)
        log_det_jac = tf.math.log(df_dx)
        return tf.reduce_sum(log_p_z + log_det_jac, -1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x, y = sample_data()
    # plt.plot(x[:, 0], x[:, 1], "x")
    # plt.show()

    model = RealNVP()
    bs = 128

    for batch in np.array_split(x, int(len(x) / bs)):
        print(model.train(batch))
