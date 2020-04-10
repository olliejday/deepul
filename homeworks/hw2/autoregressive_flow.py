import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from common import DenseNN

class ARFlow:
    """
    Wraps the model with function handles and training code
    """
    def __init__(self, n_vars, lr=10e-3):
        """
        :param n_vars: number of 1D variables to model
        """
        self.optimiser = tf.optimizers.Adam(learning_rate=lr)
        self.n_vars = n_vars
        self.model = self.setup_model()

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

    def setup_model(self):
        return ARFlowModel(self.n_vars)

    def loss(self, x):
        """
        Returns loss for batch (1,) in nats / dim
        """
        log_p_x = self.log_p_x(x)
        return - tf.reduce_mean(log_p_x) / self.n_vars

    def log_p_x(self, x):
        """
        Returns log prob of given xs (bs, n_vars)
        """
        x = tf.cast(x, tf.float32)
        # zi are uniform -> p(zi) /propto 1 -> log p (zi) = 0
        # since f(x) are cdf, derivative of f wrt x is pdf
        log_det_jac = self.model.log_pdf(x)
        # sum the log probs to get joint log prob
        # log p(x1, x2) = log p(x1) + log p(x2|x1)
        return tf.reduce_sum(log_det_jac, axis=-1)

    def f_x(self, x):
        """
        Returns z values for given xs (bs, n_vars)
        """
        x = tf.cast(x, tf.float32)
        # f is cdf
        return self.model.cdf(x)

class ARFlowModel(tf.keras.Model):
    """
    Gathers the components for each variable into the flow.
    """
    def __init__(self, n_vars, k=5, *args, **kwargs):
        """
        :param n_vars: number of variables to model
        :param k: number of gaussians in each mixture
        """
        super().__init__(*args, **kwargs)
        self.n_vars = n_vars
        self.k = k
        self.components = [ARFlowComponent(self.k, i) for i in range(self.n_vars)]

    def cdf(self, x):
        """
        :param x: (bs, n_vars) in conditioning order ie. 1st unconditioned, last conditioned on all others
        :return: cdf (bs, n_vars)
        """
        return tf.stack([comp.cdf(x[:, i], x[:, :i]) for i, comp in enumerate(self.components)], 1)

    def log_pdf(self, x):
        """
        :param x: (bs, n_vars) in conditioning order ie. 1st unconditioned, last conditioned on all others
        :return: pdf (bs, n_vars)
        """
        return tf.stack([comp.log_pdf(x[:, i], x[:, :i]) for i, comp in enumerate(self.components)], 1)


class ARFlowComponent(tf.keras.layers.Layer):
    """
    One variable component of a flow.
    A mixture of (univariate) Gaussians.
    """
    def __init__(self, k, n_cond, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        """
        :param k: # gaussians in mixture
        :param n_cond: number of vars conditioned on, 0 for first unconditioned var, if > 0 then conditioned
        vars are input to NN to compute params
        """
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.k = k
        self.n_cond = n_cond
        self.model = self.setup_model()  # model outputs MoG distrbn params

    def setup_model(self):
        """
        Model takes self.n_cond inputs and outputs params for mixture
        """
        if self.n_cond == 0:
            return ARFlowUnconditionedParamsModel(self.k)
        return ARFlowConditionedParamsModel(self.k)

    def get_distributions(self, x):
        """
        :param x: (bs, n_cond)
        :return: weights, dist
        weights is a tensor (bs, k)
        dist is a batched tfp distribution (batch shape (bs, k))
        """
        # outputs of model
        weights_logits, means, log_stddevs = self.model(x)
        stddevs = tf.exp(log_stddevs)
        weights = tf.nn.softmax(weights_logits, -1)
        return weights, tfp.distributions.Normal(means, stddevs)

    def cdf(self, x, cond_x):
        """
        :param x: (bs, 1) current var's value
        :param cond_x: (bs, n_cond) conditioned vars' values
        :return: (bs,)
        """
        x = tf.reshape(x, (len(x), 1))  # expand dims
        weight, dist = self.get_distributions(cond_x)
        # sum over mixture components
        cdf = tf.reduce_sum(weight * dist.cdf(x), axis=1)
        return cdf

    def log_pdf(self, x, cond_x):
        """
        :param x: (bs, 1) current var's value
        :param cond_x: (bs, n_cond) conditioned vars' values
        :return: (bs,)
        """
        x = tf.reshape(x, (len(x), 1))  # expand dims
        weight, dist = self.get_distributions(cond_x)
        # TODO: clip prob if nan? ensure != 0 for log
        # sum over mixture components
        pdf = tf.reduce_sum(weight * dist.prob(x), axis=1)
        log_pdf = tf.math.log(pdf)
        return log_pdf


class ARFlowConditionedParamsModel(tf.keras.layers.Layer):
    """
    Dense NN that computes the params for the mixture of gaussians given the conditioned variables as input
    """
    def __init__(self, k, n_units=128, **kwargs):
        """
        :param k: # gaussians in mixture
        """
        super().__init__(**kwargs)
        self.k = k
        self.n_units = n_units

    def build(self, input_shape):
        # weight, mean, stddev for each k component
        self.dense_nn = DenseNN(self.n_units, self.k * 3, activation="tanh")

    def call(self, inputs, **kwargs):
        """
        Returns a tuple of the params (weights logits, means, log stddevs) for each input
        ((bs, k), (bs, k), (bs, k))
        """
        x = self.dense_nn(inputs)
        return tf.split(x, 3, axis=1)


class ARFlowUnconditionedParamsModel(tf.keras.layers.Layer):
    """
    Outputs the params for the mixture of gaussians unconditioned (no inputs)
    """
    def __init__(self, k, **kwargs):
        """
        :param k: # gaussians in mixture
        """
        super().__init__(**kwargs)
        self.k = k

    def build(self, input_shape):
        self.mixture_weights_logits = self.add_weight(shape=(1, self.k))
        self.means = self.add_weight(shape=(1, self.k))
        self.log_stddevs = self.add_weight(shape=(1, self.k))

    def call(self, inputs, **kwargs):
        """
        Returns a tuple of the params (weights logits, means, log stddevs) for each input
        ((bs, k), (bs, k), (bs, k))
        """
        bs = len(inputs)
        return (tf.repeat(self.mixture_weights_logits, bs, 0),
                tf.repeat(self.means, bs, 0),
                tf.repeat(self.log_stddevs, bs, 0))


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

    model = ARFlow(2)
    bs = 128
    for batch in np.array_split(x, int(len(x) / bs)):
        print(model.train(batch))
