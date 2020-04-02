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

class MoGMLP(tf.keras.Model):
    """
    MLP that outputs the mixture components, mean, stds of a mixture of gaussians
    with inputs as prev vars
    For first var, no prev vars to input to outputs are directly the parameters
    """
    pass


class MoG:
    """
    Mixture of Gaussians where mixture components, means and stddevs are compute by MLP with inputs of
    prev vars (no prev vars for first var)
    """
    def __init__(self):
        self.MoGMLP = MoGMLP()
        self.trainable_weights = self.MoGMLP.trainable_weights

    def get_distributions(self, prev_x):
        dstrbns = []
        for (weight, mean, std) in self.MoGMLP(prev_x):
            dstbn = tfp.distributions.Normal(mean, std)
            dstrbns.append((weight, dstbn))
        return dstrbns

    def pdf(self, x, prev_x):
        prob_density = 0
        for (weight, dstrbn) in self.get_distributions(prev_x):
            prob_density += weight * dstrbn.prob(x)
        return prob_density

    def cdf(self, x, prev_x):
        cum_density = 0
        for (weight, dstrbn) in self.get_distributions(prev_x):
            cum_density += weight * dstrbn.cdf(x)
        return cum_density

    def sample(self, n, z_sample, prev_x_sample):
        # sample a distribution
        dists_and_weights = self.get_distributions(prev_x_sample)
        indx = tfp.distributions.Categorical(logits=[w for (_, w) in dists_and_weights]).sample()
        dstrbn, _ = dists_and_weights[indx]
        return dstrbn.sample(n)


class ARFlow:
    def __init__(self, n_components):
        self.n_components = n_components
        # TODO: input prev component into next
        self.components = [MoG() for _ in range(self.n_components)]
        self.trainable_weights = [comp.trainable_weights for comp in self.components]

    def log_p_x(self, x):
        """
        The log(p(f_{\theta}(x))) term is uniform
        The log(det(d f(x) / d x)) term is the sum of the pdfs
        :param x:
        """
        logpx = 0
        for comp in self.components:
            logpx += comp.pdf(x)
        return logpx

    def f(self, x):
        """
        Returns z = f_{\theta}(x)
        Numerically computed by mixture of CDFs of gaussians
        """
        z = []
        for comp in self.components:
            z.append(comp.cdf(x))
        return z

    def sample(self, n):
        """
        Returns n samples
        """
        # sample z
        z_sample = tfp.distributions.Uniform().sample((n, self.n_components))
        x_sample = []
        for comp in self.components:
            x_sample.append(comp.sample(n, z_sample, x_sample))
        return x_sample

    def train(self, X):
        with tf.GradientTape() as tape:
            loss = self.log_p_x(X)
        tape.gradient(loss, self.trainable_weights)



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
    # x, y = sample_data()
    # plt.plot(x[:, 0], x[:, 1], "x")
    # plt.show()