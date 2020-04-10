import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from autoregressive_flow import sample_data
from common import MLP


class RealNVP(tf.keras.Model):
    """
    RealNVP flow
    For 2 variables using MLPs
    """
    def __init__(self, n_affine_transfs=6, lr=0.001, *args, **kwargs):
        """
        n_affine_transfs = number of affine transformations in model, alternates order of conditioning
        lr = learning rate
        """
        super().__init__(*args, **kwargs)
        self.n_vars = 2  # 2 variable case
        self.n_affine_transfs = n_affine_transfs
        self.setup_model()
        self.optimiser = tf.optimizers.Adam(learning_rate=lr)

    def setup_model(self):
        # setup affine transformations, alternate orders (sequential or reversed) of conditioning
        self.affine_transfs = [AffineTransformation(i % 2 == 0) for i in range(self.n_affine_transfs)]
        # for z prior standard normal
        self.z_prior = tfp.distributions.Normal(tf.zeros(self.n_vars), tf.ones(self.n_vars))

    def forward(self, x):
        """
        computes z = f(x) and log(det(df(x)/dx)) (log det jac) of model (over all affine transformations)
        :return: (Zs, log_det_jacs), (bs, 2) Zs output,  (bs,) log der jacs output
        """
        z = tf.cast(x, tf.float32)
        log_det_jac = tf.zeros((len(x),), dtype=tf.float32)
        # compose flows and sum log_det_jac
        for aff_transf in self.affine_transfs:
            z, log_scale = aff_transf(z)
            log_det_jac += log_scale
        return z, log_det_jac

    def f_x(self, x):
        """
        computes z = f(x) of model (over all affine transformations)
        :return: (bs, 2) Zs output
        """
        z, _ = self.forward(x)
        return z

    def log_p_x(self, x):
        """
        computes log(det(df(x)/dx)) (log det jac) of model (over all affine transformations)
        :return: (bs,) log der jacs output
        """
        z, log_det_jac = self.forward(x)
        # z prior prob
        log_p_z = self.z_prior.log_prob(z)
        # sum over vars
        return tf.reduce_sum(log_p_z, -1) + log_det_jac

    def train(self, x):
        """
        Run training step
        Returns loss for batch (1,)
        """
        with tf.GradientTape() as tape:
            loss = self.loss(x)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimiser.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def loss(self, x):
        """
        Returns loss for batch (1,) in nats / dim
        """
        log_p_x = self.log_p_x(x)
        return - tf.reduce_mean(log_p_x) / self.n_vars


class AffineTransformation(tf.keras.layers.Layer):
    """
    Affine transformation for RealNVP
    For 2 variables using MLPs (FF-NN)
    """
    def __init__(self, left_cond, n_units=64, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        """
        :param n_units: number dense units in MLP
        :param left_cond: if True then left half variables are conditioned on ie. x2 | x1
        else reversed to right conditioned on x1 | x2 for conditioning
        """
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.n_units = n_units
        self.left_cond = left_cond
        self.setup()

    def setup(self):
        """
        Build with no input shape
        """
        # build params
        self.g_theta = MLP(self.n_units, 2, activation="relu")
        self.scale = self.add_weight("scale", shape=(1,))
        self.scale_shift = self.add_weight("scale_shift", shape=(1,))
        # input mask
        if self.left_cond:
            self.mask = np.array([1.0, 0.0])
        else:
            self.mask = np.array([0.0, 1.0])

    def call(self, inputs, **kwargs):
        """
        computes z = f(x) and log(det(df(x)/dx)) (log det jac) of affine transformation

        # since jac is triangular, det of jac is prod of diag
        # but in this 2 variable case the first diagonal (dz1 / dx1) is 1
        # 2nd diag (dz2 / dx2) is tf.exp(log_scale)
        # so log this gives log_scale as log det jac, we sum over vars (as log of prod of diag is sum)

        :param inputs: (bs, 2) Xs inputs
        :return: (Zs, log_det_jacs), (bs, 2) Zs output,  (bs,) log der jacs output
        """
        log_scale, g_theta_shift = self.log_scale(inputs)
        z = tf.exp(log_scale) * inputs + g_theta_shift
        return z, tf.reduce_sum(log_scale, -1)

    def log_scale(self, x):
        """
        Retunrs log scale term and g_theta, shift term output from MLP
        :param x1: (bs, n_vars-1) conditoned vars input
        :return: (bs,), (bs,)
        """
        # MLP outputs
        g_theta_scale, g_theta_shift = tf.split(self.g_theta(self.mask * x), 2, -1)
        log_scale = self.scale * tf.tanh(g_theta_scale) + self.scale_shift
        # mask outputs opposite to inputs
        log_scale = (1.0 - self.mask) * log_scale
        mask_g_theta_shift = (1.0 - self.mask) * g_theta_shift
        return log_scale, mask_g_theta_shift


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x, y = sample_data()
    # plt.plot(x[:, 0], x[:, 1], "x")
    # plt.show()

    model = RealNVP()
    bs = 128

    for batch in np.array_split(x, int(len(x) / bs)):
        print(model.train(batch))
