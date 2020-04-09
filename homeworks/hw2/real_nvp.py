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
    n_affine_transfs = number of affine transformations in model, alternates order of conditioning
    """
    def __init__(self, n_affine_transfs=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_vars = 2  # 2 variable case
        self.n_affine_transfs = n_affine_transfs
        self.setup()

    def setup(self):
        # setup affine transformations, alternate orders (sequential or reversed) of conditioning
        self.affine_transfs = [AffineTransformation(i % 2 == 0) for i in range(self.n_affine_transfs)]
        # for z prior standard normal
        self.z_prior = tfp.distributions.Normal(tf.zeros(self.n_vars), tf.ones(self.n_vars))

    def f_x(self, x):
        # compose flows
        for aff_transf in self.affine_transfs:
            x = aff_transf.f_x(x)
        return x

    def log_p_x(self, x):
        # z prior prob
        log_p_z = self.z_prior.log_prob(self.f_x(x))
        # sum log det jacs
        transfs_list = []
        for aff_transf in self.affine_transfs:
            x = aff_transf.log_det_jac(x)
            transfs_list.append(x)
        log_det_jac = tf.reduce_sum(transfs_list, 0)
        # sum over vars
        return tf.reduce_sum(log_p_z, -1) + tf.reduce_sum(log_det_jac, -1)


class AffineTransformation(tf.keras.layers.Layer):
    """
    Affine transformation for RealNVP
    For 2 variables using MLPs (FF-NN)
    """
    def __init__(self, left_cond, n_units=128, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
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
        self.g_theta = DenseNN(self.n_units, 2, activation="relu")  # TODO: tanh?
        self.scale = self.add_weight("scale", shape=(1,))
        self.scale_shift = self.add_weight("scale_shift", shape=(1,))

    def f_x(self, inputs):
        """
        computes z = f(x)
        :param inputs: (bs, 2) Xs inputs
        :return: (bs, 2) Zs output
        """
        x1, x2 = self.split_vars(inputs)
        z1 = x1
        # MLP outputs on conditioned vars
        log_scale, g_theta_shift = self.log_scale(x1)
        z2 = tf.exp(log_scale) * x2 + g_theta_shift
        # TODO: should this be different if order of vars is different?
        if self.left_cond:
            return tf.concat([z1, z2], -1)
        else:
            return tf.concat([z2, z1], -1)

    def log_scale(self, x1):
        """
        Retunrs log scale term and g_theta, shift term output from MLP
        :param x1: (bs, n_vars-1) conditoned vars input
        :return: (bs,), (bs,)
        """
        g_theta_scale, g_theta_shift = tf.split(self.g_theta(x1), 2, -1)
        log_scale = self.scale * tf.tanh(g_theta_scale) + self.scale_shift
        return log_scale, g_theta_shift

    def split_vars(self, inputs):
        """
        Splits inputs into current and conditioned variables based on order of conditioning
        Note x1 and x2 are just the order of conditioning: x2 | x1, not necessarily variable order or labels in data
        :param inputs: (bs, n_vars)
        :return: (bs, n), (bs, m) where n+m=n_vars
        """
        # order variables sequentially or not?
        # TODO: input ordering?
        if self.left_cond:
            x1, x2 = tf.split(inputs, 2, -1)
        else:
            x2, x1 = tf.split(inputs, 2, -1)
        return x1, x2

    def log_det_jac(self, inputs):
        """
        computes log(det( df(x) / dx))
        :param inputs: (bs, n_vars) Xs inputs
        :return: (bs,)
        """
        x1, _ = self.split_vars(inputs)
        # since jac is triangular, det of jac is prod of diag
        # but in this 2 variable case the first diagonal (dz1 / dx1) is 1
        # 2nd diag (dz2 / dx2) is tf.exp(log_scale) (see above in f_x)
        # so log this gives log_scale as log det jac
        log_scale, _ = self.log_scale(x1)
        # TODO: should this be different if order of vars is different?
        if self.left_cond:
            return tf.concat([tf.zeros((len(inputs), 1)), log_scale], -1)
        else:
            return tf.concat([log_scale, tf.zeros((len(inputs), 1))], -1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x, y = sample_data()
    # plt.plot(x[:, 0], x[:, 1], "x")
    # plt.show()

    model = RealNVP()
    bs = 128

    for batch in np.array_split(x, int(len(x) / bs)):
        print(model.train(batch))
