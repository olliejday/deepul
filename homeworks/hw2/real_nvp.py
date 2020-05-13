import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class AdamLRSchedule:
    def __init__(self, start_lr, end_lr, n_steps):
        """
        Warm up the learning rate in the optimizer
        From 0 to steps, linearly interpolates between start_lr and end_lr, then onwards uses end_lr
        :param start_lr: start learning rate
        :param end_lr: end learning rate
        :param n_steps: number of steps to warm up in
        """
        self.n_steps = n_steps
        # n_steps-2 bc the learning rate steps include endpoints as steps whereas lin_interp() does not
        self.lrs = linear_interpolate(start_lr, end_lr, n_steps - 2)
        # init optimiser and step count
        self.optimiser = None
        self.step = 0

    def __call__(self):
        """
        :return: tf.optimizers.Adam object with current learning rate at step
        """
        # linearly interpolate up to n_steps, otherwise keep same Adam object so it can use momentum
        if self.step < self.n_steps:
            lr = self.lrs[self.step]
            self.optimiser = tf.optimizers.Adam(lr)
        self.step += 1
        return self.optimiser


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


class RealNVP:
    def __init__(self, H, W, C, N, lr=5e-4, clip_norm=1.):
        """
        :param H, W, C: the height, width and number channels of image input
        :param N: number of values each variable (channel of pixel) can take
        :param clip_norm: clip gradient norm before update
        """
        # warm up 200 steps
        self.optimiser = AdamLRSchedule(lr / 10, lr, 2)
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
        # get the optimiser for this step
        optimiser = self.optimiser()
        optimiser.apply_gradients(zip(grads, self.model.trainable_variables))
        # update steps for learning rate schedule
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
        bs, h, w, c = tf.shape(im1)
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

    def sample(self, n):
        """
        Sample n images
        :param n: number of samples
        :return: n samples
        """
        # get zs
        zs = self.model.get_prior_z().sample(n)
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
        self.prior_z = tfp.distributions.Normal(tf.zeros((self.H, self.W, self.C)), tf.ones((self.H, self.W, self.C)))
        # Model from paper Dinh et al, architecture from course homework handout
        self._layer_group1 = []
        for i in range(4):
            # use alternate pattern (inverse mask) on even layers
            alt_pattern = i % 2 == 0
            self._layer_group1.append(AffineCouplingWithCheckerboard(self.n_filters, alt_pattern))
            # self._layer_group1.append(ActNorm())
        self.squeeze = Squeeze()

        self._layer_group2 = []
        for i in range(3):
            alt_pattern = i % 2 == 0
            self._layer_group2.append(AffineCouplingWithChannel(self.n_filters, alt_pattern))
            self._layer_group2.append(ActNorm())
        self.unsqueeze = Unsqueeze()

        self._layer_group3 = []
        for i in range(3):
            alt_pattern = i % 2 == 0
            self._layer_group3.append(AffineCouplingWithCheckerboard(self.n_filters, alt_pattern))
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
        # for layer in self._layer_group3:
        #     z, delta_log_det = layer(z)
        #     log_det += delta_log_det
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
        x = zs
        # for layer in reversed(self._layer_group3):
        #     x = layer.inverse(x)
        # swap squeeze and unsqueeze
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
        return (zs - self._bias) * tf.exp(self._log_scale)


class Squeeze(tf.keras.layers.Layer):
    """
    Squeeze - reshape to channels by dividing into 2 x 2 x c sub squares and then flattening each to 1 x 1 x 4c
    [b, h, w, c] --> [b, h//2, w//2, c*4]
    """

    def call(self, inputs, **kwargs):
        bs, h, w, c = tf.shape(inputs)
        # TODO: simplify un/squeeze?
        inputs_sub_sq = tf.reshape(inputs, (bs, h // 2, 2, w // 2, 2, c))
        return tf.reshape(tf.transpose(inputs_sub_sq, (0, 1, 3, 5, 2, 4)), (bs, h // 2, w // 2, 4 * c))


class Unsqueeze(tf.keras.layers.Layer):
    """
    Reverse of Squeeze()
    [b, h//2, w//2, c*4] --> [b, h, w, c]
    """

    def call(self, inputs, **kwargs):
        bs, h, w, c = tf.shape(inputs)
        inputs_sub_sq = tf.reshape(tf.transpose(inputs, (0, 3, 1, 2)), (bs, c // 4, 2, 2, h, w))
        return tf.reshape(tf.transpose(inputs_sub_sq, (0, 4, 2, 5, 3, 1)), (bs, h * 2, w * 2, c // 4))


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters=128, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters

    def build(self, input_shape):
        self._layers = []
        # TODO (Note): pseudocode had in and out as padding=0 and middle as padding=1
        self._layers.append(Conv2D(self.n_filters, (1, 1), activation="relu", strides=1, padding="VALID"))
        self._layers.append(Conv2D(self.n_filters, (3, 3), activation="relu", strides=1, padding="SAME"))
        self._layers.append(Conv2D(self.n_filters, (1, 1), activation="relu", strides=1, padding="VALID"))

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x + inputs


# TODO: try using tfp.WeightNorm
def Conv2D(*args, **kwargs):
    return tfp.layers.weight_norm.WeightNorm(tf.keras.layers.Conv2D(*args, **kwargs))


class Conv2D1(tf.keras.layers.Conv2D):
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
        # get statistics over filters
        mean_t = tf.reduce_mean(t, axis=[0, 1, 2])
        stddev_t = tf.math.reduce_std(t, axis=[0, 1, 2])
        # params to init weights
        g = 1. / stddev_t
        b = - mean_t / stddev_t
        # init weights
        self.kernel = tf.Variable(v * g / v_norm, name="kernel", trainable=True)
        self.bias = tf.Variable(b, name="bias", trainable=True)
        # check ok
        tf.assert_equal(tf.shape(self.kernel), kernel_shape,
                        "Kernel init to wrong size. Was {}, now {}".format(tf.shape(self.kernel), kernel_shape))
        tf.assert_equal(tf.shape(self.bias), bias_shape,
                        "Bias init to wrong size. Was {}, now {}".format(tf.shape(self.bias), bias_shape))
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
        self._scale = self.add_weight(name="scale", shape=(1,))
        self._scale_shift = self.add_weight(name="scale", shape=(1,))
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

        # TODO: del for debug
        def get_stats(a):
            return (tf.reduce_min(a, [0, 1, 2]).numpy(), tf.reduce_max(a, [0, 1, 2]).numpy())

        def get_mean_std(a):
            return (tf.reduce_mean(a, axis=[0, 1, 2]).numpy(), tf.math.reduce_std(a, axis=[0, 1, 2]).numpy())

        x_mu = get_mean_std(x)
        x_stats = get_stats(x)
        zs_mu = get_mean_std(z)
        zs_stats = get_stats(z)
        weights_list = [w.numpy() for w in self.trainable_variables]
        weights_stats = (np.min([np.min(w) for w in weights_list]), np.max([np.min(w) for w in weights_list]),
                         np.mean([np.min(w) for w in weights_list]), np.std([np.min(w) for w in weights_list]))
        ###
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
        # TODO: del for debug
        x_mu = tf.reduce_mean(x, axis=[0, 1, 2])
        zs_mu = tf.reduce_mean(zs, axis=[0, 1, 2])
        ###
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
#     # TODO: should mask be all ones and split the inputs to forward instead? does this o/w mean the later channels will be half zeros?
#     def get_mask(self, input_shape):
#         mask = np.ones(input_shape[1:])
#         # mask out last half channel
#         mask[:, :, input_shape[-1] // 2:] = 0.
#         return mask

class AffineCouplingWithChannel(tf.keras.layers.Layer):
    def __init__(self, n_filters, alt_pattern, **kwargs):
        """
        :param n_filters: number of filers each conv layer
        :param alt_pattern: if True then masking uses inverse of mask pattern so that each layer alternates
        which parts are masked.
        """
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.alt_pattern = alt_pattern

    def build(self, input_shape):
        self._scale = self.add_weight(name="scale", shape=(1,))
        self._scale_shift = self.add_weight(name="scale", shape=(1,))
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

        # TODO: del for debug
        def get_stats(a):
            return (tf.reduce_min(a, [0, 1, 2]).numpy(), tf.reduce_max(a, [0, 1, 2]).numpy())

        def get_mean_std(a):
            return (tf.reduce_mean(a, axis=[0, 1, 2]).numpy(), tf.math.reduce_std(a, axis=[0, 1, 2]).numpy())

        x_mu = get_mean_std(x)
        x_stats = get_stats(x)
        zs_mu = get_mean_std(z)
        zs_stats = get_stats(z)
        weights_list = [w.numpy() for w in self.trainable_variables]
        weights_stats = (np.min([np.min(w) for w in weights_list]), np.max([np.min(w) for w in weights_list]),
                         np.mean([np.min(w) for w in weights_list]), np.std([np.min(w) for w in weights_list]))
        ###
        return self.join_mask(z, x_off), tf.concat([log_scale, tf.zeros_like(log_scale)], axis=-1)

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
        # TODO: del for debug
        x_mu = tf.reduce_mean(x, axis=[0, 1, 2])
        zs_mu = tf.reduce_mean(zs, axis=[0, 1, 2])
        ###
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
        # TODO: right way round? - pretty sure masking is right way now
        if self.alt_pattern:
            x_off, x_on = tf.split(x, (self.n_out // 2, self.n_out - self.n_out // 2), axis=-1)
            return x_on, x_off
        else:
            return tf.split(x, (self.n_out // 2, self.n_out - self.n_out // 2), axis=-1)

    def join_mask(self, ys, x_off):
        """
        Joins outputs with the masked off inputs
        Order depends on masking order
        Both inputs can be Xs or Zs but in this model should be different
        :param ys: model outputs
        :param x_off: data inputs that were masked off, x_off
        :return: x, joined data and outputs
        """
        if self.alt_pattern:
            return tf.concat([x_off, ys], axis=-1)
        else:
            return tf.concat([ys, x_off], axis=-1)


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


def squeeze_test(n, bs, c):
    """
    :param n: n x n images
    :param bs: batch size
    :param c: # channels
    """
    # this matches image in paper for n=4
    # TODO: implement this as squeeze and unsqueeze
    im = np.arange(1, bs * c * (n ** 2) + 1).reshape((bs,c, n // 2, n // 2, 2, 2)).transpose((0,1,2,4,3,5)).reshape((bs,c,n,n)).transpose((0,2,3,1))
    # TODO: transpose (0,1,3, 5,2,4) so s1c1 s2c1 s3c1 s4c1 s2c1 ... ie. subsquares first - this matches ucb solutions - WHY?
    #   or (0, 1, 3, 2, 4, 5) so s1c1 s1c2 s1c3 s2c1 s2c2 .... ie. channels first - I think channels first
    #   because you want to flatten by pixel and so mask channels to mask pixels not infact channels
    squeezed = im.reshape((bs, n // 2, 2, n // 2, 2, c)).transpose((0, 1, 3, 5, 2, 4)).reshape((bs, n // 2, n // 2, 4*c))
    for i in range(6):
        print(squeezed[0, :, :, i])
    # use reference of squeezed shape
    _, n_sq, n_sq, c_sq = np.shape(squeezed)
    unsqueezed = squeezed.transpose((0,3,1,2)).reshape((bs, c_sq // 4, 2, 2, n_sq, n_sq)).transpose((0,4,2,5,3,1)).reshape((bs,n_sq*2,n_sq*2, c_sq//4))
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
    x = preprocess(x, n)

    for i in range(150):
        loss = real_nvp.train(x).numpy()
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
