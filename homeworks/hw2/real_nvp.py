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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_vars = 2  # 2 variable case
        self.setup()

    def setup(self):
        self.affine_transf1 = AffineTransformation(True)
        self.affine_transf2 = AffineTransformation(False)
        # for z prior standard normal
        self.z_prior = tfp.distributions.Normal(tf.zeros(self.n_vars), tf.ones(self.n_vars))

    def f_x(self, x):
        # compose flows
        return self.affine_transf2.f_x(self.affine_transf1.f_x(x))

    def log_p_x(self, x):
        # z prior prob
        log_p_z = self.z_prior.log_prob(self.f_x(x))
        # sum log det jacs
        log_det_jac = self.affine_transf1.log_det_jac(x) + self.affine_transf2.log_det_jac(x)
        # sum over vars
        return tf.reduce_sum(log_p_z + log_det_jac, -1)


class AffineTransformation(tf.keras.layers.Layer):
    """
    Affine transformation for RealNVP
    For 2 variables using MLPs (FF-NN)
    """
    def __init__(self, sequential_cond, n_units=128, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        """
        :param n_units: number dense units in MLP
        :param reverse: if True then x2 | x1 else  x1 | x2 for conditioning
        """
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.n_units = n_units
        self.sequential_cond = sequential_cond
        self.setup()

    def setup(self):
        """
        Build with no input shape
        """
        # build params
        self.g_theta_scale = DenseNN(self.n_units, 1, activation="tanh")  # TODO: relu?
        self.g_theta_shift = DenseNN(self.n_units, 1, activation="tanh")
        self.scale = self.add_weight("scale", shape=(1,))
        self.scale_shift = self.add_weight("scale_shift", shape=(1,))

    def f_x(self, inputs):
        """
        computes z = f(x)
        :param inputs: (bs, 2) Xs inputs
        :return: (bs, 2) Zs output
        """
        x_cond, x = self.split_vars(inputs)
        z1 = x_cond
        log_scale = self.scale * tf.tanh(self.g_theta_scale(x_cond)) + self.scale_shift
        z2 = tf.exp(log_scale) * x + self.g_theta_shift(x_cond)
        return tf.concat([z1, z2], -1)

    def split_vars(self, inputs):
        # order variables sequentially or not? This functions uses x | x_cond
        if self.sequential_cond:
            x_cond, x = tf.split(inputs, 2, -1)
        else:
            x, x_cond = tf.split(inputs, 2, -1)
        return x_cond, x

    def log_det_jac(self, inputs):
        """
        computes log(det( df(x) / dx))
        :param inputs: (bs, n_vars) Xs inputs
        :return: (bs,)
        """
        x_cond, x = self.split_vars(inputs)
        # since jac is triangular, det of jac is prod of diag
        # but in this 2 variable case the first diagonal (dz1 / dx1) is 1
        # 2nd diag (dz2 / dx2) is tf.exp(log_scale) (see above in f_x)
        # so log this gives log_scale as log det jac
        return self.scale * tf.tanh(self.g_theta_scale(x_cond)) + self.scale_shift

# TODO: try another affine trasnf ontop with other variable ordering
#   compose ie. f2(f1(x)) then add for log det jac sum over k det(dfk/fk-1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x, y = sample_data()
    # plt.plot(x[:, 0], x[:, 1], "x")
    # plt.show()

    model = RealNVP()
    bs = 128

    for batch in np.array_split(x, int(len(x) / bs)):
        print(model.train(batch))
