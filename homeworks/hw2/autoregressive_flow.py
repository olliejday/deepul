import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# class ARFlowTransformation(tf.keras.layers.Layer):
#     """
#     A NN that computes the flow transformation
#     z_i = f(x_i; x_{1:i-1})
#     """
#     def __init__(self, n_units, activation="relu", **kwargs):
#         super().__init__(**kwargs)
#         self.n_units = n_units
#         self.activation = activation
#
#     def build(self, input_shape):
#         self.layers = []
#         self.layers.append(Dense(self.n_units, activation=self.activation))
#         self.layers.append(Dense(self.n_units, activation=self.activation))
#         # output layer is sigmoid as z_i \in [0, 1]
#         self.layers.append(Dense(self.n_units, activation="sigmoid"))
#
#     def call(self, inputs, **kwargs):
#         """
#         From data to latent z space [0, 1] (inference)
#         :param inputs: (x_i; x_{1:i-1}) where x_i is the input and x_{1:i-1} are the conditioned variables
#         :return: z_i = f(x_i; x_{1:i-1})
#         """
#         # concat inputs
#         x = tf.concat(inputs, axis=-1)
#         for layer in self.layers:
#             x = layer(x)
#         return x
#
#
#     def inverse(self, inputs):
#         """
#         From latent z space to data (sampling)
#         :param inputs: (z_i; x_{1:i-1}) inputs to sampling
#         :return: x_i = f^{-1}(z_i; x_{1:i-1})
#         """
#         pass
#
# class ARFlowModel(tf.keras.Model):
#     # x_1 no condition just a fixed MoL/MoG, inv is then inv of this
#     # x_2 is a MoL/MoG with params as outputs from a NN that takes x_1 as input, inv is then inv of this
#     #  parameterised CDF
#     # Q says use MoL/MoG to map TO latent space ie. f : x -> z
#     # how to invert mixture function? z = a * f1(x) + b * f2(x) ,,, -> x = ?
#     pass


#

class MoGMLP(tf.keras.layers.Layer):
    """
    MLP that outputs the mean, stds of a mixture of gaussians
    with inputs as prev vars
    For first var, no prev vars to input to outputs are directly the parameters
    """
    def __init__(self, n_in, n_vals, n_units=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_in = n_in
        self.n_units = n_units
        self.n_vals = n_vals
        self.n_out = self.n_vals * 2

    def build(self, input_shape):
        self.layers_list = []
        if self.n_in > 0:
            self.layers_list.append(tf.keras.layers.Dense(self.n_in))
            self.layers_list.append(tf.keras.layers.Dense(self.n_units))
            self.layers_list.append(tf.keras.layers.Dense(self.n_out))
        else:
            # if first layer fixed params, since no inputs
            self.params = self.add_weight(name="Params", shape=(self.n_out,))

    def call(self, inputs, **kwargs):
        if self.n_in > 0:
            x = inputs
            for layer in self.layers_list:
                x = layer(x)
        else:
            # unconditional if first layer fixed params
            # since no inputs, input is batch size for number of distributions to have
            x = tf.tile(self.params, tf.reshape(inputs, (1,)))
        return tf.unstack(tf.reshape(x, (-1, self.n_vals, 2)), axis=-1)


class MoG(tf.keras.Model):
    """
    Mixture of Gaussians where mixture components, means apnd stddevs are compute by MLP with inputs of
    prev vars (no prev vars for first var)
    """
    def __init__(self, n_in, n_out, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.n_in = n_in
        self.n_out = n_out
        self.components = [MoGMLP(n_in, n_out) for _ in range(self.k)]
        self.mixture_weights = self.add_weight(name="MixureWeights", shape=(self.k,), trainable=True)

    def get_distributions(self, prev_x, batch_size):
        dstrbns = []
        for comp in self.components:
            dstbn = self.get_distribution(comp, prev_x, batch_size)
            dstrbns.append(dstbn)
        return dstrbns

    def get_distribution(self, comp, prev_x, batch_size):
        # unconditional (1st var) has no inputs, so we input batch size for return shape
        if self.n_in == 0:
            prev_x = batch_size
        mean, std = comp(prev_x)
        dstbn = tfp.distributions.Normal(tf.squeeze(mean), tf.squeeze(std))
        return dstbn

    def get_mixture_weights(self):
        return tf.nn.softmax(self.mixture_weights)

    def pdf(self, x, prev_x):
        weighted_sum, dstrbns, weights = self.setup_df(prev_x, x)
        for i in range(self.k):
            prob = dstrbns[i].prob(x)
            # prob = tf.where(tf.math.is_nan(prob), 0, prob)  # TODO: ok?
            weighted_sum += weights[i] * prob
        return weighted_sum

    def concat_history(self, prev_x):
        if prev_x is not None and len(prev_x) > 1:
            prev_x = tf.concat(prev_x, axis=-1)
        return prev_x

    def cdf(self, x, prev_x):
        weighted_sum, dstrbns, weights = self.setup_df(prev_x, x)
        for i in range(self.k):
            weighted_sum += weights[i] * dstrbns[i].cdf(x)
        return weighted_sum

    def setup_df(self, prev_x, x):
        prev_x = self.concat_history(prev_x)
        aggr = tf.zeros(tf.shape(x)[0])
        dstrbns = self.get_distributions(prev_x, len(x))
        weights = self.get_mixture_weights()
        return aggr, dstrbns, weights

    def sample(self, n, z_sample, prev_x_sample):
        # TODO: not using z_sample here??
        weights = self.get_mixture_weights()
        # sample a distribution
        indxs = tfp.distributions.Categorical(logits=weights).sample(n)
        samples = []
        # sample as many samples from each distribution
        for i, comp in enumerate(self.components):
            # if we have samples from this component distribution
            indx = tf.where(indxs == i)
            n_i = len(indx)
            if n_i > 0:
                # get distribution for this component for each input
                prev_x_i = None
                if self.n_in > 0:
                    prev_x_i = tf.gather(prev_x_sample, indx)
                dstrbns = self.get_distribution(comp, prev_x_i, n_i)
                # get a sample, the batch dimension is in the distribution by inputing a batch of inputs
                sample = tf.reshape(dstrbns.sample(1), (n_i, 1))
                samples.append(sample)
        samples = tf.concat(samples, 0)
        return samples


class ARFlow:
    # TODO: atm assumes each var only sinlge 1D output
    def __init__(self, n_vars, k=3, learning_rate=10e-4):
        self.n_vars = n_vars
        # mixture per variable
        self.mixtures = [MoG(i, 1, k) for i in range(self.n_vars)]
        self.optimiser = tf.optimizers.Adam(learning_rate=learning_rate)

    def log_p_x(self, x):
        """
        The log(p(f_{\theta}(x))) term is uniform
        The log(det(d f(x) / d x)) term is the sum of the pdfs
        """
        logpx = tf.zeros((len(x),))
        for i in range(self.n_vars - 1):
            logpx += self.mixtures[i].pdf(x[:, i], x[:, :i])
        return logpx

    def f(self, x):
        """
        Returns z = f_{\theta}(x)
        Numerically computed by mixture of CDFs of gaussians
        """
        zs = []
        for i in range(self.n_vars - 1):
            z = self.mixtures[i].cdf(x[:, i], x[:, :i])
            zs.append(z)
        return zs

    def sample(self, n):
        """
        Returns n samples
        """
        # sample z
        z_sample = tfp.distributions.Uniform().sample((n, self.n_vars))
        x_samples = []
        x_sample = None
        for mix in self.mixtures:
            if len(x_samples) > 0:
                x_sample = tf.concat(x_samples, -1)
            x_samples.append(mix.sample(n, z_sample, x_sample))
        x_sample = tf.concat(x_samples, -1)
        return x_sample

    def train(self, X):
        with tf.GradientTape() as tape:
            loss = self.loss(X)
        grads = tape.gradient(loss, self.get_trainable_weights())
        self.optimiser.apply_gradients(zip(grads, self.get_trainable_weights()))
        return loss

    def loss(self, X):
        return tf.reduce_mean(self.log_p_x(X))

    def get_trainable_weights(self):
        return [w for mix in self.mixtures for w in mix.trainable_weights]


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

    # TODO: seed
    model = ARFlow(2)
    bs = 128
    x_dataset = tf.data.Dataset.from_tensor_slices(tf.cast(x, tf.float32))
    x_iter = x_dataset.shuffle(bs * 2).batch(bs)
    for batch in x_iter:
        loss = model.train(batch)
        print(loss)
    samples = model.sample(10000)
    plt.plot(samples[:, 0], samples[:, 1], "x")
    plt.show()
