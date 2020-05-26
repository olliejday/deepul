import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def linear_interpolate(a, b, n):
    """
    Linearly interpolate between a and b in n steps
    eg. linear_interpolate(0, 4, 3) = [0, 1, 2, 3, 4]
    :param a: start (SHAPE)
    :param b: stop (SHAPE)
    :param n: number of steps in interpolation (not including end points)
    :return: n+2 steps from a to b (includes a and b endpoints), (n+2, SHAPE)
    """
    # we want to capture n steps so need n+1 incremements
    n += 1
    return [a + (b - a) * (i / n) for i in range(n + 1)]


def get_input_shape(inputs):
    """
    Splits input shape into vars
    :param inputs: x (bs, h, w, c)
    :return: bs, h, w, c
    """
    input_shape = tf.shape(inputs)
    bs, h, w, c = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    return bs, h, w, c


class RealNVP:
    def __init__(self, H, W, C, N, lr=5e-4, clip_norm=0.1):
        """
        :param H, W, C: the height, width and number channels of image input
        :param N: number of values each variable (channel of pixel) can take
        :param clip_norm: clip gradient norm before update
        """
        self.optimiser = tf.optimizers.Adam(learning_rate=lr)
        self.clip_norm = clip_norm
        self.N = N
        self.H, self.W, self.C = H, W, C
        self.n_vars = H * W * C
        self.model = self.setup_model()

    def train(self, x, log_det_x):
        """
        x, log_det_x the inputs and the log_det to account for preprocessing
        Run training step
        Returns loss for batch (1,)
        """
        with tf.GradientTape() as tape:
            loss = self.loss(x, log_det_x)
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.clip_norm)
        self.optimiser.apply_gradients(zip(grads, self.model.trainable_variables))
        # update steps for learning rate schedule
        return loss

    def setup_model(self):
        return RealNVPModel(self.N)

    @tf.function
    def loss(self, x, log_det_x):
        """
        Returns negative log prob for batch (1,) in nats / dim
        We scale (in logs) based on preprocessing (scale_loss)
        And scale by number of variables for nats / dim
        """
        log_p_x = self.log_p_x(x)
        # log_p_x is (bs,) summed over vars so need to get mean over number of vars for nats / dim
        # scale (in log space) to account for preprocessing scaling
        return - tf.reduce_mean(log_p_x + log_det_x) / self.n_vars

    def log_p_x(self, x):
        """
        :return: log (joint) prob of given xs (bs,)
        log p(x) = log p(z) + log det(dz / dx)
        """
        x = tf.cast(x, tf.float32)
        # forward model and log det term
        z, log_det = self.model(x)
        # sum log det for log det jac
        log_det_jac = tf.reduce_sum(log_det, axis=[1, 2, 3])
        # prior z
        log_pz = self.model.get_prior_z().log_prob(z)
        # sum over logs is joint
        log_pz = tf.reduce_sum(log_pz, axis=[1, 2, 3])
        return log_pz + log_det_jac

    def f_x(self, x):
        """
        Returns z values for given xs (bs, h, w, c)
        """
        x = tf.cast(x, tf.float32)
        z, _ = self.model(x)
        return z

    @tf.function
    def interpolate(self, im1, im2, n):
        """
        Interpolate between two images
        :param im1: image 1 (bs, h, w, c)
        :param im2: image 2 (bs, h, w, c)
        :param n: number of images to interpolate
        :return: { start image, n generated interpolations, end image } for each pair in batch
        (bs * (n+2), h, w, c)
        """
        im1 = tf.cast(im1, tf.float32)
        im2 = tf.cast(im2, tf.float32)
        bs, h, w, c = get_input_shape(im1)
        # get z values
        z1 = self.f_x(im1)
        z2 = self.f_x(im2)
        # interpolate, drop end points, (n, bs, h, w, c)
        zs = linear_interpolate(z1, z2, n)[1:-1]
        # flatten n interpolations into batch for passing to model.inv()
        # careful to gather in batches not in interpolations
        zs = tf.reshape(tf.transpose(zs, (1, 0, 2, 3, 4)), (bs * n, h, w, c))
        # generate images
        xs = self.model.inverse(zs)
        # add endpoints
        xs_batches = tf.reshape(xs, (bs, n, h, w, c))
        xs_endpoints = tf.concat([tf.expand_dims(im1, 1), xs_batches, tf.expand_dims(im2, 1)], axis=1)
        xs_endpoints = tf.reshape(xs_endpoints, (bs * (n + 2), h, w, c))
        return xs_endpoints

    @tf.function
    def sample(self, n):
        """
        Sample n images
        :param n: number of samples
        :return: n samples
        """
        # get zs
        zs = self.model.get_prior_z().sample((n, self.H, self.W, self.C))
        # generate images
        xs = self.model.inverse(zs)
        return xs


class RealNVPModel(tf.keras.Model):
    def __init__(self, n_filters=128, *args, **kwargs):
        """
        :param n_filters: number filters each conv layer
        """
        self.n_filters = n_filters
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        # store shape, ignore batch size
        _, self.H, self.W, self.C = input_shape
        # prior on z; we have a z per channel per pixel of x
        # standard normal
        self.prior_z = tfp.distributions.Normal(0., 1.)
        # Model from paper Dinh et al, architecture from course homework handout
        self._layer_group1 = []
        for i in range(4):
            # use alternate pattern (inverse mask) on even layers
            alt_pattern = i % 2 != 0
            self._layer_group1.append(AffineCouplingWithCheckerboard(self.n_filters, alt_pattern))
            # no act norm last layer
            if i < 3:
                self._layer_group1.append(ActNorm())
        self.squeeze = Squeeze()

        self._layer_group2 = []
        for i in range(3):
            alt_pattern = i % 2 != 0
            self._layer_group2.append(AffineCouplingWithChannel(self.n_filters, alt_pattern))
            if i < 2:
                self._layer_group2.append(ActNorm())
        self.unsqueeze = Unsqueeze()

        self._layer_group3 = []
        for i in range(3):
            alt_pattern = i % 2 != 0
            self._layer_group3.append(AffineCouplingWithCheckerboard(self.n_filters, alt_pattern))
            if i < 2:
                self._layer_group3.append(ActNorm())

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs:
        :return: z, log_det
        """
        z = inputs
        # diagonal jacobian so log det jac is sum of log dets
        log_det = tf.zeros_like(inputs)
        # pass through model, model is grouped in layers by masking pattern with un/squeeze ops
        for layer in self._layer_group1:
            z, delta_log_det = layer(z)
            log_det += delta_log_det
        z, log_det = self.squeeze(z), self.squeeze(log_det)
        for layer in self._layer_group2:
            z, delta_log_det = layer(z)
            log_det += delta_log_det
        z, log_det = self.unsqueeze(z), self.unsqueeze(log_det)
        for layer in self._layer_group3:
            z, delta_log_det = layer(z)
            log_det += delta_log_det
        return z, log_det

    def inverse(self, zs):
        """
        Compute inverse flow x = f^-1(z)
        :param zs: bacth of zs (bs, H, W, C)
        :return: xs of shape (bs, H, W, C)
        """
        if not self.built:
            raise ValueError("Model not yet built. Please call() model first.")
        # go through layers of forward pass (call()) in reverse calling .inverse()
        # swap squeeze and unsqueeze
        x = zs
        for layer in reversed(self._layer_group3):
            x = layer.inverse(x)
        x = self.squeeze(x)
        for layer in reversed(self._layer_group2):
            x = layer.inverse(x)
        x = self.unsqueeze(x)
        for layer in reversed(self._layer_group1):
            x = layer.inverse(x)
        return x

    def get_prior_z(self):
        """
        :return: tfp.distribution, the prior for z (of shape (H, W, C)
        """
        if not self.built:
            raise ValueError("Model not yet built. Please call() model first.")
        return self.prior_z


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
        self.H, self.W = input_shape[1:3]
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        :param x: inputs (bs, H, W, C)
        :return: z (bs, H', W', C'), log_det_jacobian (bs,)
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
        return self._log_scale

    def f_x(self, inputs):
        """
        Compute function of act-norm z = f_x(x)
        """
        return tf.exp(self._log_scale) * inputs + self._bias

    def init_kernel(self, inputs):
        """
        As per paper, data dependent init so post-act norm per channel has zero mean and unit std deviation
        :param inputs: batch of data [bs, H, W, C]
        """
        # get statistics per channel
        mean_t = tf.reduce_mean(inputs, axis=[0, 1, 2], keepdims=True)
        stddev_t = tf.math.reduce_std(inputs, axis=[0, 1, 2], keepdims=True)
        # init weights
        self._log_scale = tf.Variable(-tf.math.log(stddev_t), name="log_scale", trainable=True)
        # bias not trainable, since doesn't appear in log det
        self._bias = tf.Variable(-mean_t, name="bias", trainable=False)
        # initialised ok
        self.is_initialised = True

    def inverse(self, zs):
        """
        Inverse flow x = f^-1(z)
        :param zs: z inputs (bs, h, w, c)
        :return: x outputs
        """
        return (zs - self._bias) * tf.exp(-self._log_scale)


class Squeeze(tf.keras.layers.Layer):
    """
    Squeeze - reshape to channels by dividing into 2 x 2 x c sub squares and then flattening each to 1 x 1 x 4c
    [b, h, w, c] --> [b, h//2, w//2, c*4]
    """
    def call(self, inputs, **kwargs):
        bs, h, w, c = get_input_shape(inputs)
        inputs_sub_sq = tf.reshape(inputs, (bs, h // 2, 2, w // 2, 2, c))
        return tf.reshape(tf.transpose(inputs_sub_sq, (0, 1, 3, 5, 2, 4)), (bs, h // 2, w // 2, 4 * c))


class Unsqueeze(tf.keras.layers.Layer):
    """
    Reverse of Squeeze()
    [b, h//2, w//2, c*4] --> [b, h, w, c]
    """

    def call(self, inputs, **kwargs):
        bs, h, w, c = get_input_shape(inputs)
        inputs_sub_sq = tf.reshape(tf.transpose(inputs, (0, 3, 1, 2)), (bs, c // 4, 2, 2, h, w))
        return tf.reshape(tf.transpose(inputs_sub_sq, (0, 4, 2, 5, 3, 1)), (bs, h * 2, w * 2, c // 4))


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters=128, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters

    def build(self, input_shape):
        self._layers = []
        self._layers.append(Conv2D(self.n_filters, (1, 1), activation="relu", strides=1, padding="VALID"))
        self._layers.append(Conv2D(self.n_filters, (3, 3), activation="relu", strides=1, padding="SAME"))
        self._layers.append(Conv2D(self.n_filters, (1, 1), strides=1, padding="VALID"))

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x + inputs


def Conv2D(*args, **kwargs):
    return tfp.layers.weight_norm.WeightNorm(tf.keras.layers.Conv2D(*args, **kwargs))


class SimpleResnet(tf.keras.layers.Layer):
    def __init__(self, n_out, n_filters=128, n_res=8, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.n_layers = n_res
        self.n_out = n_out

    def build(self, input_shape, **kwargs):
        self._layers = []
        self._layers.append(Conv2D(self.n_filters, (3, 3), strides=1, padding="SAME",
                                   activation="relu"))
        for _ in range(self.n_layers):
            self._layers.append(ResnetBlock(self.n_filters))
        self._layers.append(tf.keras.layers.Activation("relu"))
        self._layers.append(Conv2D(self.n_out, (3, 3), strides=1, padding="SAME"))

    def call(self, x, **kwargs):
        for layer in self._layers:
            x = layer(x)
        return x


class AffineCoupling(tf.keras.layers.Layer):
    def __init__(self, n_filters, alt_pattern, **kwargs):
        """
        :param n_filters: number of filers each conv layer
        :param alt_pattern: if True then masking uses inverse of mask pattern so that each layer alternates
        which parts are masked.
        """
        super().__init__(**kwargs)
        self.mask = None
        self.n_filters = n_filters
        self.alt_pattern = alt_pattern

    def get_mask(self, input_shape):
        """
        Overwrite to setup mask.
        Don't set to alt pattern, this is done in build.
        :param input_shape: (bs, H, W, C)
        :return: mask (0, 1) of shape (H, W, C)
        """
        raise NotImplementedError

    def build(self, input_shape):
        mask = self.get_mask(input_shape)
        # invert if alt pattern
        if self.alt_pattern:
            mask = 1. - mask
        self.mask = mask
        self._scale = self.add_weight(name="scale", shape=(1,), initializer=tf.zeros_initializer())
        self._scale_shift = self.add_weight(name="scale_shit", shape=(1,), initializer=tf.zeros_initializer())
        # want same shape as input, double output size for t and s
        n_out = input_shape[-1]
        self.resnet = SimpleResnet(n_out * 2, self.n_filters)

    def call(self, x, **kwargs):
        """
        :param x: inputs (bs, H, W, C)
        :return: z (bs, H', W', C'), log_det_jacobian (bs,)
        Where z is the forward pass f(x)
        log_det_jacobian is the log-determinant of f(x)
        """
        # element wise mask
        log_scale, t = self.get_scale_and_shift(x)
        z = x * tf.exp(log_scale) + t
        return z, log_scale

    def inverse(self, zs):
        """
        Inverse flow x = f^-1(z)
        :param zs: z inputs (bs, h, w, c)
        :return: x outputs
        """
        # element wise mask
        log_scale, t = self.get_scale_and_shift(zs)
        # inverse flow
        x = (zs - t) * tf.exp(-log_scale)
        return x

    def get_scale_and_shift(self, x):
        """
        Get log_scale and shift, t, by processing outputs of resnet
        :param x: inputs to resnet, can be Z OR X! (unmasked)
        :return: log_scale, t
        """
        x_mask = x * self.mask
        resnet = self.resnet(x_mask)
        log_scale, t = tf.split(resnet, 2, axis=-1)
        log_scale = self._scale * tf.tanh(log_scale) + self._scale_shift
        log_scale = log_scale * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        return log_scale, t


class AffineCouplingWithCheckerboard(AffineCoupling):
    """
    Figure 3 in Dinh et al - (left)
    """

    def get_mask(self, input_shape):
        # checkerboard mask
        # we want main mask here, top left corner = 1
        checkerboard = 1. - np.indices(input_shape[1:3]).sum(axis=0) % 2
        # stack channels
        mask = np.stack([checkerboard] * input_shape[-1], 2)
        return mask


# class AffineCouplingWithChannel(AffineCoupling):
#     def get_mask(self, input_shape):
#         mask = np.ones(input_shape[1:])
#         # mask out last half channel
#         mask[:, :, input_shape[-1] // 2:] = 0.
#         return mask

class AffineCouplingWithChannel(tf.keras.layers.Layer):
    def __init__(self, n_filters, alt_pattern, **kwargs):
        """
        :param n_filters: number of filers each conv layer
        :param alt_pattern: if True then masks later channels o/w masks first channels
        """
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.alt_pattern = alt_pattern

    def build(self, input_shape):
        self._scale = self.add_weight(name="scale", shape=(1,), initializer=tf.zeros_initializer())
        self._scale_shift = self.add_weight(name="scale_shit", shape=(1,), initializer=tf.zeros_initializer())
        # double #channels to split into t and s, but we only use 1/2 of channels input so same #channels overall
        self.n_out = input_shape[-1]
        self.resnet = SimpleResnet(self.n_out, self.n_filters)

    def call(self, x, **kwargs):
        """
        :param x: inputs (bs, H, W, C)
        :return: z (bs, H', W', C'), log_det_jacobian (bs,)
        Where z is the forward pass f(x)
        log_det_jacobian is the log-determinant of f(x)
        """
        # element wise mask
        x_on, x_off = self.apply_mask(x)
        log_scale, t = self.get_scale_and_shift(x_off)
        z = x_on * tf.exp(log_scale) + t
        return self.join_mask(z, x_off), self.join_mask(log_scale, tf.zeros_like(log_scale))

    def inverse(self, zs):
        """
        Inverse flow x = f^-1(z)
        :param zs: z inputs (bs, h, w, c)
        :return: x outputs
        """
        z_on, z_off = self.apply_mask(zs)
        log_scale, t = self.get_scale_and_shift(z_off)
        # inverse flow
        x = (z_on - t) * tf.exp(-log_scale)
        return self.join_mask(x, z_off)

    def get_scale_and_shift(self, x_on):
        """
        Get log_scale and shift, t, by processing outputs of resnet
        :param x: inputs to resnet, can be Z OR X! Masked to on channels
        :return: log_scale, t
        """
        resnet = self.resnet(x_on)
        log_scale, t = tf.split(resnet, 2, axis=-1)
        log_scale = self._scale * tf.tanh(log_scale) + self._scale_shift
        return log_scale, t

    def apply_mask(self, x):
        """
        Masks input channel wise
        Either first n/2 are 1s and last 0s or reversed in alt masking
        :param x: X OR Z, inputs to mask
        :return: x_on, x_off the masked as 1s and 0s respectively
        """
        if self.alt_pattern:
            x_off, x_on = tf.split(x, (self.n_out // 2, self.n_out - self.n_out // 2), axis=-1)
            return x_on, x_off
        else:
            return tf.split(x, (self.n_out // 2, self.n_out - self.n_out // 2), axis=-1)

    def join_mask(self, x_on, x_off):
        """
        Joins outputs with the masked off inputs
        Order depends on masking order
        Both inputs can be Xs or Zs but in this model should be different
        :param x_on: model outputs or pixels to replace x_on in masking (1)
        :param x_off: data inputs that were masked off, x_off in masking (0)
        :return: x, joined data and outputs
        """
        if self.alt_pattern:
            return tf.concat([x_off, x_on], axis=-1)
        else:
            return tf.concat([x_on, x_off], axis=-1)


def logit_trick(x, n, a=0.05, b=0.9):
    """
    logit trick from RealNVP (Dinh et al) Section 4.1
    :param x: inputs (in data-space)
    :param n: number of values x can take
    :param a: hyper param, min value
    :param b: hyper param, max value
    :return: same shape as x, numpy
    """
    # pre-logit
    p = a + b * x / n
    # log odds function
    return np.log(p) - np.log(1 - p)


def inverse_logit_trick(y, n, a=0.05, b=0.9):
    """
    inverse of logit trick to map back to data
    :param y: inputs (in logit-trick-space)
    :param n: number of values x can take
    :param a: hyper param, min value
    :param b: hyper param, max value
    :return: same shape as y, numpy
    """
    # sigmoid is inverse logit function
    p = tf.nn.sigmoid(y).numpy()
    return (p - a) / b


def preprocess(data, n, dq=True):
    """
    Preprocess the data - dequantise and logit trick from RealNVP (Dinh et al) Section 4.1
    :param data: input data (N, H, W, C) N examples, H,W,C image shape
    :param n: number of values each variable can take on
    :param dq: bool, whether to dequatise data, defaults True
    :return: same shape as input, preprocessed
    """
    data = tf.cast(data, tf.float32)
    # dequantization
    if dq:
        data = data + np.random.random(np.shape(data))
    # logit trick
    data = logit_trick(data, n)
    # account for preprocessing
    log_det = tf.cast(tf.nn.softplus(data) + tf.nn.softplus(-data), tf.float32) + tf.math.log(0.9) \
              - tf.math.log(tf.cast(n, tf.float32))
    log_det = tf.reduce_sum(log_det, [1, 2, 3])
    return data, log_det


def squeeze_test(n, bs, c):
    """
    :param n: n x n images
    :param bs: batch size
    :param c: # channels
    """
    # this matches image in paper for n=4
    im = np.arange(1, bs * c * (n ** 2) + 1).reshape((bs,c, n // 2, n // 2, 2, 2)).transpose((0,1,2,4,3,5)).reshape((bs,c,n,n)).transpose((0,2,3,1))
    #  type 1 (0,1,3, 5,2,4) so s1c1 s2c1 s3c1 s4c1 s1c2 ... ie. subsquares first - this matches ucb solutions - WHY?
    #  type 2(0, 1, 3, 2, 4, 5) so s1c1 s1c2 s1c3 s2c1 s2c2 .... ie. channels first - I think channels first
    #   because you want to flatten by pixel and so mask channels to mask pixels not infact channels
    # type 2
    squeezed = im.reshape((bs, n // 2, 2, n // 2, 2, c)).transpose((0, 1, 3, 2, 4, 5) ).reshape((bs, n // 2, n // 2, 4*c))
    # type 1
    # squeezed = im.reshape((bs, n // 2, 2, n // 2, 2, c)).transpose((0, 1, 3, 5, 2, 4)).reshape((bs, n // 2, n // 2, 4*c))
    for i in range(6):
        print(squeezed[0, :, :, i])
    # use reference of squeezed shape
    _, n_sq, n_sq, c_sq = np.shape(squeezed)
    # type 2
    unsqueezed = squeezed.reshape((bs, n_sq, n_sq, 2, 2, c_sq//4)).transpose((0, 1, 3, 2, 4, 5)).reshape(bs, n_sq*2, n_sq*2, c_sq//4)
    # type 1
    # unsqueezed = squeezed.transpose((0,3,1,2)).reshape((bs, c_sq // 4, 2, 2, n_sq, n_sq)).transpose((0,1,2,5,3,4)).reshape((bs,n_sq*2,n_sq*2, c_sq//4))
    print(unsqueezed[0, :, :, 0])
    print(np.allclose(im, unsqueezed))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(123)

    h, w, c = 6, 6, 3
    n = 4
    real_nvp = RealNVP(h, w, c, n, lr=5e-3)

    bs = 64
    x = np.stack([np.eye(h) * np.random.randint(0, n, (h,))] * bs * c).reshape((bs, h, w, c))
    x, log_det_x = preprocess(x, n)

    for i in range(150):
        loss = real_nvp.train(x, log_det_x).numpy()
        if i % 10 == 0:
            print(loss)
            interp = real_nvp.interpolate(x[:2], x[2:4], 4).numpy()
            interp_plot = np.hstack(np.hstack(interp.reshape(2, 4 + 2, h, w, c)))
            plt.imshow(interp_plot)
            plt.title("Interp")
            plt.show()

    interp = real_nvp.interpolate(x[:2], x[2:4], 4).numpy()
    interp_plot = np.hstack(np.hstack(interp.reshape(2, 4 + 2, h, w, c)))
    plt.imshow(interp_plot)
    plt.title("Interp")
    plt.show()

    sample = real_nvp.sample(8).numpy()
    plt.imshow(np.hstack(sample))
    plt.title("Samples")
    plt.show()
