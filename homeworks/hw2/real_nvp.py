import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class RealNVP:
    def __init__(self, H, W, C, N, clip_norm=1.):
        """
        :param H, W, C: the height, width and number channels of image input
        :param N: number of values each variable (channel of pixel) can take
        :param clip_norm: clip gradient norm before update
        """
        # TODO: setup opt
        # Adam Optimizer with a warmup over 200 steps till a learning rate of 5e-4.
        # We didn’t decay the learning rate but it is a generally recommended practice while training generative models
        self.optimiser = None
        self.clip_norm = clip_norm
        self.N = N
        self.n_vars = H * W * C
        self.model = self.setup_model()

    def train(self, x):
        """
        Run training step
        Returns loss for batch (1,)
        """
        with tf.GradientTape() as tape:
            loss = self.loss(x)
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.clip_norm)
        self.optimiser.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def setup_model(self):
        return RealNVPModel(self.N)

    def loss(self, x):
        """
        Returns negative log prob for batch (1,) in nats / dim
        We scale (in logs) based on preprocessing (scale_loss)
        And scale by number of variables for nats / dim
        """
        log_p_x = self.log_p_x(x)
        # log_p_x is (bs,) summed over vars so need to get mean over number of vars for nats / dim
        # scale (in log space) to account for preprocessing scaling
        return - tf.reduce_mean(log_p_x) / self.n_vars

    def log_p_x(self, x):
        """
        Returns log (joint) prob of given xs (bs,)
        """
        x = tf.cast(x, tf.float32)
        _, log_det_jac = self.model(x)
        return tf.reduce_sum(log_det_jac, axis=-1)

    def f_x(self, x):
        """
        Returns z values for given xs (bs, h, w, c)
        """
        x = tf.cast(x, tf.float32)
        z, _ = self.model(x)
        return z


class ActNorm(tf.keras.layers.Layer):
    """
    Described in Glow (Kingma & Dhariwal) Section 3.1
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # use first call to initialise from data
        self.is_initialised = False

    def build(self, input_shape):
        # define the parameters, reinititialised in first call (data-dependent)
        self._weight = self.add_weight("weight", shape=input_shape[1:])
        self._bias = self.add_weight("bias", shape=input_shape[1:])
        self.H, self.W = input_shape[1:3]
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        :param x: inputs
        :return: z, log_det
        Where z is the forward pass f(x)
        log_det is the log-determinant of f(x)
        """
        if not self.is_initialised:
            self.init_kernel(inputs)
        z = self.f_x(inputs)
        log_det = self.log_det()
        return z, log_det

    def log_det(self):
        """
        Compute log-det of f(x)
        """
        return self.H * self.W * tf.reduce_sum(tf.math.log(tf.abs(self._weight)))

    def f_x(self, inputs):
        """
        Compute function of act-norm z = f_x(x)
        """
        return self._weight * inputs + self._bias

    def init_kernel(self, inputs):
        """
        As per paper, data dependent init so post-act norm per channel has zero mean and unit std deviation
        :param inputs: batch of data [bs, H, W, C]
        """
        # post act norm
        outs = self.f_x(inputs)
        # get statistics per channel
        mean_t = tf.reduce_mean(outs, axis=[0, 1, 2])
        stddev_t = tf.math.reduce_std(outs, axis=[0, 1, 2])
        # init weights
        self._weight = 1. / stddev_t
        self._bias = - mean_t / stddev_t
        # initialised ok
        self.is_initialised = True


class Squeeze(tf.keras.layers.Layer):
    """
    Reshape to channels
    [b, h, w, c] --> [b, h//2, w//2, c*4]
    """
    def call(self, inputs, **kwargs):
        bs, h, w, c = tf.shape(inputs)
        return tf.reshape(inputs, (bs, h // 2, w // 2, c * 4))


class Unsqueeze(tf.keras.layers.Layer):
    """
    [b, h//2, w//2, c*4] --> [b, h, w, c]
    """
    def call(self, inputs, **kwargs):
        bs, h, w, c = tf.shape(inputs)
        return tf.reshape(inputs, (bs, h * 2, w * 2, c // 4))


class RealNVPModel(tf.keras.Model):
    def __init__(self, N, n_filters=128, *args, **kwargs):
        """
        :param N: number of values each variable (channel of pixel) can take
        :param n_filters: number filters each conv layer
        """
        self.N = N
        self.n_filters = n_filters
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        # Model from paper Dinh et al, architecture from course homework handout
        self._layer_group1 = []
        for _ in range(4):
            self._layer_group1.append(AffineCouplingWithCheckerboard(self.n_filters))
            self._layer_group1.append(ActNorm())
        self.squeeze = Squeeze()

        self._layer_group2 = []
        for _ in range(3):
            self._layer_group2.append(AffineCouplingWithChannel(self.n_filters))
            self._layer_group2.append(ActNorm())
        self.unsqueeze = Unsqueeze()

        self._layer_group3 = []
        for _ in range(3):
            self._layer_group3.append(AffineCouplingWithCheckerboard(self.n_filters))
            self._layer_group3.append(ActNorm())

    def call(self, inputs, training=None, mask=None):
        z = inputs
        # TODO (note) summing log_det in outputs from affinecouple layers - not sure this is right but shapes don't match o/w
        # diagonal jacobian so log det jac is sum of log dets
        log_det_jac = 0
        # pass through model, model is grouped in layers by masking pattern with un/squeeze ops
        for layer in self._layer_group1:
            z, log_det = layer(z)
            log_det_jac += log_det
        z = self.squeeze(z)
        for layer in self._layer_group2:
            z, log_det = layer(z)
            log_det_jac += log_det
        z = self.unsqueeze(z)
        for layer in self._layer_group3:
            z, log_det = layer(z)
            log_det_jac += log_det
        # we use logit trick so have to invert to map to data
        # TODO (note) is this how we map z and log_det_jac?
        z = inverse_logit_trick(z, self.N)
        log_det_jac = inverse_logit_trick(log_det_jac, self.N)
        return z, log_det_jac


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters=128, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters

    def build(self, input_shape):
        self._layers = []
        # TODO (Note): pseudocode had in and out as padding=0 and middle as padding=1
        self._layers.append(Conv2D(self.n_filters, (1, 1), strides=1, padding="VALID"))
        self._layers.append(Conv2D(self.n_filters, (3, 3), strides=1, padding="SAME"))
        self._layers.append(Conv2D(self.n_filters, (1, 1), strides=1, padding="VALID"))

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x + inputs


class Conv2D(tf.keras.layers.Conv2D):
    """
    Overwrite Conv2D with data dependent weight initialisation
    as per Weight Normalisation, Salimans and Kingma, 2016
    """
    def __init__(self, *args, seed=123, **kwargs):
        super().__init__(*args, **kwargs)
        # use first call to initialise from data
        self.is_initialised = False
        self.seed = seed

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        # use first call to initialise
        if not self.is_initialised:
            self.init_kernel(inputs)
        return super().call(inputs)

    def init_kernel(self, inputs):
        kernel_shape = tf.shape(self.kernel)
        bias_shape = tf.shape(self.bias)
        # sample v from Normal mean 0 and stddev 0.05
        v = tfp.distributions.Normal(np.zeros(kernel_shape), np.ones(kernel_shape) * 0.05).sample(1, seed=self.seed)[0]
        v = tf.cast(v, tf.float32)
        v_norm = tf.norm(v)
        # pre activation, not tf uses upper strings for padding
        t = tf.nn.conv2d(inputs, v / v_norm, strides=self.strides, padding=self.padding.upper())
        # get statistics over filters TODO: this right?
        mean_t = tf.reduce_mean(t, axis=[0, 1, 2])
        stddev_t = tf.math.reduce_std(t, axis=[0, 1, 2])
        # params to init weights
        g = 1. / stddev_t
        b = - mean_t / stddev_t
        # init weights
        self.kernel = v * g / v_norm
        self.bias = b
        # check ok
        tf.assert_equal(tf.shape(self.kernel), kernel_shape, "Kernel init to wrong size. Was {}, now {}".format(tf.shape(self.kernel), kernel_shape))
        tf.assert_equal(tf.shape(self.bias), bias_shape, "Bias init to wrong size. Was {}, now {}".format(tf.shape(self.bias), bias_shape))
        # initialised ok
        self.is_initialised = True


class SimpleResnet(tf.keras.layers.Layer):
    def __init__(self, n_out, n_filters=128, n_res=8, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.n_layers = n_res
        self.n_out = n_out

    def build(self, input_shape, **kwargs):
        self._layers = []
        # TODO (Note): pseudocode has padding=1 what does this mean in tf - think it could be same as this means output same size
        self._layers.append(Conv2D(self.n_filters, (3, 3), strides=1, padding="SAME",
                                                   activation="relu"))
        for _ in range(self.n_layers):
            self._layers.append(ResnetBlock(self.n_filters))
        self._layers.append(Conv2D(self.n_out, (3, 3), strides=1, padding="SAME"))

    def call(self, x, **kwargs):
        for layer in self._layers:
            x = layer(x)
        return x


class AffineCoupling(tf.keras.layers.Layer):
    def __init__(self, n_filters, **kwargs):
        """
        :param n_filters: number of filers each conv layer
        """
        super().__init__(**kwargs)
        self.mask = None
        self.n_filters = n_filters

    def get_mask(self, input_shape):
        """
        Overwrite to setup mask
        :param input_shape: (bs, H, W, C)
        :return: mask (0, 1) of shape (H, W, C)
        """
        raise NotImplementedError

    def build(self, input_shape):
        self.mask = self.get_mask(input_shape)
        # want same shape as input, double output size for t and s
        n_out = input_shape[-1]
        self.resnet = SimpleResnet(n_out * 2, self.n_filters)

    def call(self, x, **kwargs):
        """
        :param x: inputs
        :return: z, log_det_jacobian
        Where z is the forward pass f(x)
        log_det_jacobian is the log-determinant of f(x)
        """
        # element wise mask
        x_masked = x * self.mask
        resnet = self.resnet(x_masked)
        log_scale, t = tf.split(resnet, 2, axis=-1)
        # calculate log_scale, as done in Q1(b)
        t = t * (1.0 - self.mask)
        log_scale = log_scale * (1.0 - self.mask)
        z = x * tf.exp(log_scale) + t
        log_det_jacobian = log_scale
        return z, tf.reduce_sum(log_det_jacobian, [1, 2, 3])


class AffineCouplingWithCheckerboard(AffineCoupling):
    """
    Figure 3 in Dinh et al - (left)
    """
    def get_mask(self, input_shape):
        # checkerboard mask
        return np.indices(input_shape[1:]).sum(axis=0) % 2


class AffineCouplingWithChannel(AffineCoupling):
    def get_mask(self, input_shape):
        mask = np.ones(input_shape[1:])
        # mask out 1st half channel
        mask[:, :, :input_shape[-1] // 2] = 0.
        return mask


def logit_trick(x, n, a=0.05):
    """
    logit trick from RealNVP (Dinh et al) Section 4.1
    :param x: inputs (in data-space)
    :param n: number of values x can take
    :param a: alpha hyper param, default=0.05 as paper
    :return: same shape as x
    """
    # pre-logit
    p = a + (1 - a) * x / n
    # log odds function
    return np.log(p) - np.log(1 - p)


def inverse_logit_trick(y, n, a=0.05):
    """
    inverse of logit trick to map back to data
    :param y: inputs (in logit-trick-space)
    :param n: number of values x can take
    :param a: alpha hyper param, default=0.05 as paper
    :return: same shape as y
    """
    # sigmoid is inverse logit function
    # TODO: overflow in exp
    p = tf.nn.sigmoid(y)
    return n * (p - a) / (1 - a)


def preprocess(data, n, alpha=0.05):
    """
    Preprocess the data - dequantise and logit trick from RealNVP (Dinh et al) Section 4.1
    :param data: input data (N, H, W, C) N examples, H,W,C image shape
    :param n: number of values each variable can take on
    :param alpha: hyper param, default=0.05 as paper
    :return: same shape as input, preprocessed
    """
    # dequantization
    data = data + np.random.random(np.shape(data))
    # logit trick
    data = logit_trick(data, n, alpha)
    return data


if __name__ == "__main__":
    h, w, c = 6, 6, 2
    n = 3
    real_nvp = RealNVP(h, w, c, n)

    bs = 64
    x = np.stack([np.eye(h)] * bs * 2).reshape((bs, h, w, c))
    x = preprocess(x, n)
    for i in range(10):
        print(real_nvp.train(x))