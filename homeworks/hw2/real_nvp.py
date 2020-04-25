import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class RealNVP:
    def __init__(self):
        # Adam Optimizer with a warmup over 200 steps till a learning rate of 5e-4.
        # We didn’t decay the learning rate but it is a generally recommended practice while training generative models
        self.optimiser = None


class ActNorm(tf.keras.layers.Layer):
    # TODO: will just batch norm do?
    pass


class Squeeze(tf.keras.layers.Layer):
    # TODO: reshape to channels
    pass


class Unsqueeze(tf.keras.layers.Layer):
    # TODO: reshape back
    pass


class RealNVPModel(tf.keras.Model):
    # TODO: weight normalisation for init
    def build(self, input_shape):
        # Model from paper Dinh et al, architecture from course homework handout
        self._layers = []
        for _ in range(4):
            self._layers.append(AffineCouplingWithCheckerboard())  # Figure 3 in Dinh et al - (left)
            self._layers.append(ActNorm())  # described in Glow (Kingma & Dhariwal) Section 3.1
        self._layers.append(Squeeze()),  # [b, h, w, c] --> [b, h//2, w//2, c*4]

        for _ in range(3):
            self._layers.append(AffineCouplingWithChannel())
            self._layers.append(ActNorm())
        self._layers.append(Unsqueeze())  # [b, h//2, w//2, c*4] --> [b, h, w, c]

        for _ in range(3):
            self._layers.append(AffineCouplingWithCheckerboard())
            self._layers.append(ActNorm())

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters=128, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters

    def build(self, input_shape):
        self._layers = []
        # TODO: pseudocode had in and out as padding=0 and middle as padding=1
        self._layers.append(tf.keras.layers.Conv2D(self.n_filters, (1, 1), stride=1, padding="VALID"))
        self._layers.append(tf.keras.layers.Conv2D(self.n_filters, (3, 3), stride=1, padding="SAME"))
        self._layers.append(tf.keras.layers.Conv2D(self.n_filters, (1, 1), stride=1, padding="VALID"))

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x + inputs


class SimpleResnet(tf.keras.layers.Layer):
    def __init__(self, n_out, n_filters=128, n_res=8, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.n_layers = n_res
        self.n_out = n_out

    def build(self, **kwargs):
        self._layers = []
        # TODO: pseudocode has padding=1 what does this mean in tf - think it could be same as this means output same size
        self._layers.append(tf.keras.layers.Conv2D(self.n_filters, (3, 3), strides=1, padding="SAME",
                                                   activation="relu"))
        for _ in range(self.n_layers):
            self._layers.append(ResnetBlock(self.n_filters))
        self._layers.append(tf.keras.layers.Conv2D(self.n_out, (3, 3), stride=1, padding="SAME"))

    def call(self, x, **kwargs):
        for layer in self._layers:
            x = layer(x)
        return x


class AffineCoupling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = None
        self.n_out = None  # TODO

    def get_mask(self, input_shape):
        """
        Overwrite to setup mask
        :return: mask to mask input to resnet
        """
        raise NotImplementedError

    def build(self, input_shape):
        self.mask = self.get_mask(input_shape)
        self.resnet = SimpleResnet(self.n_out)

    def call(self, x, **kwargs):
        x_masked = x * self.mask
        resnet = self.resnet(x_masked)
        log_scale, t = tf.split(resnet, 2, axis=-1)
        # calculate log_scale, as done in Q1(b)
        t = t * (1.0 - self.mask)
        log_scale = log_scale * (1.0 - self.mask)
        z = x * tf.exp(log_scale) + t
        log_det_jacobian = log_scale
        return z, log_det_jacobian


class AffineCouplingWithCheckerboard(AffineCoupling):
    pass


class AffineCouplingWithChannel(AffineCoupling):
    pass


def preprocess():
    # TODO
    # dequantization, logit trick from RealNVP (Dinh et al) Section 4.1
    return None
