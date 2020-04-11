import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# TODO: dequantize the data and scale

# class PixelCNNARFlow:
#     """
#     Wraps the model with function handles and training code
#     PixelCNN outputs with an autoregressive flow of mixture of Gaussians
#     """
#     def __init__(self, H, W, k, lr=10e-3):
#         """
#
#         :param H, W: height and width of image, assumed single, real-valued channel
#         :param k: # gaussians in mixture
#         :param lr: learning rate
#         """
#         self.H, self.W = H, W
#         self.k = k
#         self.optimiser = tf.optimizers.Adam(learning_rate=lr)
#         self.model = self.setup_model()
#
#     def train(self, x):
#         """
#         Run training step
#         Returns loss for batch (1,)
#         """
#         with tf.GradientTape() as tape:
#             loss = self.loss(x)
#         grads = tape.gradient(loss, self.model.trainable_variables)
#         self.optimiser.apply_gradients(zip(grads, self.model.trainable_variables))
#         return loss
#
#     def setup_model(self):
#         return PixelCNNARFlowModel(self.H, self.W, self.k)
#
#     def loss(self, x):
#         """
#         Returns loss for batch (1,) in nats / dim
#         """
#         log_p_x = self.log_p_x(x)
#         n_vars = None  # TODO
#         return - tf.reduce_mean(log_p_x) / n_vars
#
#     def log_p_x(self, x):
#         """
#         Returns log prob of given xs (bs, n_vars)
#         """
#         x = tf.cast(x, tf.float32)
#         # zi are uniform -> p(zi) /propto 1 -> log p (zi) = 0
#         # since f(x) are cdf, derivative of f wrt x is pdf
#         log_det_jac = self.model.log_p_x(x)
#         # sum the log probs to get joint log prob
#         # log p(x1, x2) = log p(x1) + log p(x2|x1)
#         return tf.reduce_sum(log_det_jac, axis=-1)
#
#     def f_x(self, x):
#         """
#         Returns z values for given xs (bs, n_vars)
#         """
#         x = tf.cast(x, tf.float32)
#         # f is cdf
#         return self.model.cdf(x)
#
#
# class PixelCNNARFlowModel(tf.keras.layers.Layer):
#     """
#     PixelCNN outputs a mixture of Gaussians for each (B&W, real valued) pixel for an autoregressive flow
#     """
#     def __init__(self, H, W, k, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
#         """
#         :param H, W: height and width of image, assumed single, real-valued channel
#         :param k: # gaussians in mixture
#         """
#         super().__init__(trainable, name, dtype, dynamic, **kwargs)
#         self.H, self.W = H, W
#         self.k = k
#         self.model = self.setup_model()  # model outputs MoG distrbn params
#
#     def setup_model(self):
#         """
#         Model takes (bs,image shape) inputs and outputs (bs, image shape * 3) params for mixture
#         """
#         return PixelCNNModel(self.H, self.W, self.C)
#
#     def get_distributions(self, x):
#         """
#         :param x: (bs, n_cond)
#         :return: weights, dist
#         weights is a tensor (bs, k)
#         dist is a batched tfp distribution (batch shape (bs, k))
#         """
#         # outputs of model
#         weights_logits, means, log_stddevs = self.model(x)
#         stddevs = tf.exp(log_stddevs)
#         weights = tf.nn.softmax(weights_logits, -1)
#         return weights, tfp.distributions.Normal(means, stddevs)
#
#     def f_x(self, x):
#         """
#         z = f(x)
#         Here this is the cdf of mixture model
#         :param x: (bs, 1) current var's value
#         :param cond_x: (bs, n_cond) conditioned vars' values
#         :return: (bs,)
#         """
#         x = tf.reshape(x, (len(x), 1))  # expand dims
#         cond_x = None
#         weight, dist = self.get_distributions(cond_x)
#         # sum over mixture components
#         cdf = tf.reduce_sum(weight * dist.cdf(x), axis=1)
#         return cdf
#
#     def log_p_x(self, x):
#         """
#         log p(x) = log p(f(x)) + log (det(df/dx))
#         here the log det term is the pdf of mixture model
#         :param x: (bs, 1) current var's value
#         :param cond_x: (bs, n_cond) conditioned vars' values
#         :return: (bs,)
#         """
#         x = tf.reshape(x, (len(x), 1))  # expand dims
#         cond_x = None
#         weight, dist = self.get_distributions(cond_x)
#         # TODO: clip prob if nan? ensure != 0 for log
#         # sum over mixture components
#         pdf = tf.reduce_sum(weight * dist.prob(x), axis=1)
#         log_pdf = tf.math.log(pdf)
#         return log_pdf


"""
PixelCNN
"""

def get_pixelcnn_mask(kernel_size, in_channels, out_channels, isTypeA, n_channels=3, factorised=True):
    """
    Masks are repeated in groups with modulo if channel in or out != n_channels so if
    5 channels then it's R, R, G, G, B etc.
    This is so that when reshaping to (H, W, #channels, #values) we get each channel's values aligning
    For RGB channel taking N values case it's R1, R2, ... RN, G1, G2, ... GN, B1, B2, ... BN which reshapes to
    [R1, R2, ... RN], [G1, G2, ... GN], [B1, B2, ... BN]

    raster ordering on conditioning mask.

    kernel_size: size N of filter N x N
    in_channels: number of channels/filters in
    out_channels: number of channels/filters out
    n_channels: number of channels for masking eg. 3 for RGB masks
    isTypeA: bool, true if type A mask, otherwise type B mask used.
        Type A takes context and previous channels (but not its own channel)
        Type B takes context, prev channels and connected to own channel.
    factorised: bool, if True then we factorise over channels, ie. probs treated independently P(r)p(g)p(b)
        so mask type A all have centre off and B all have it on.
        Otherwise the full joint probs are used p(r)p(g|r)p(b|r,g) which requires conditioning on previous channels
        and A and B masks are different for each channel to allow this.

    Returns masks of shape (kernel_size, kernel_size, # in channels, # out channels)
    """
    channel_masks = np.ones((kernel_size, kernel_size, n_channels, n_channels), dtype=np.int32)
    centre = kernel_size // 2
    # bottom rows 0s
    channel_masks[centre + 1:, :, :, :] = 0
    # right of centre on centre row 0s
    channel_masks[centre:, centre + 1:, :, :] = 0
    # deal with centre based on mask "way": factorised or full
    # rows are channels in prev layer, columns are channels in this layer
    if factorised:
        if isTypeA:
            channel_masks[centre, centre, :, :] = 0
    else:
        # centre depends on mask type A or B
        k = 0 if isTypeA else 1
        # reverse i and j to get RGB ordering (other way would be BGR)
        i, j = np.triu_indices(n_channels, k)
        channel_masks[centre, centre, j, i] = 0.

    # we use repeat not tile because this keeps the correct ordering we need
    tile_shape = (int(np.ceil(in_channels / n_channels)), int(np.ceil(out_channels / n_channels)))
    masks = np.repeat(channel_masks, tile_shape[0], axis=2)
    masks = np.repeat(masks, tile_shape[1], axis=3)
    # tile the masks to potentially more than needed, then retrieve the number of channels wanted
    return masks[:, :, :in_channels, :out_channels]


class MaskedCNN(tf.keras.layers.Conv2D):
    def __init__(self, n_filters, kernel_size, isTypeA, factorised, activation=None, **kwargs):
        """
        n_filters and kernel_size for conv layer
        isTypeA for mask type
        """
        assert isinstance(kernel_size, int), "Masked CNN requires square n x n kernel"
        super(MaskedCNN, self).__init__(n_filters, kernel_size, padding="SAME",
                                        activation=activation, **kwargs)
        self.isTypeA = isTypeA
        self.factorised = factorised

    def build(self, input_shape):
        super().build(input_shape)
        (_, _, in_channels, out_channels) = self.kernel.shape
        self.mask = get_pixelcnn_mask(self.kernel_size[0], in_channels, out_channels, self.isTypeA,
                                      factorised=self.factorised)

    def call(self, inputs):
        # mask kernel for internal conv op, but then return to copy of kernel after for learning
        kernel_copy = self.kernel
        self.kernel = self.kernel * self.mask
        out = super().call(inputs)
        self.kernel = kernel_copy
        return out


class MaskedResidualBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters, factorised):
        super().__init__()
        self.n_filters = n_filters
        self.factorised = factorised

    def build(self, input_shape):
        # 1x1 relu filter, then 3x3 then 1x1
        self.layer1 = MaskedCNN(self.n_filters, 1, False, self.factorised)
        self.layer2 = MaskedCNN(self.n_filters, 7, False, self.factorised)
        self.layer3 = MaskedCNN(self.n_filters * 2, 1, False, self.factorised)

    def call(self, inputs, **kwargs):
        """
        x is the inputs, [image, (cx, cy)]
        """
        # other layers take img and cur pixel location
        x = tf.keras.layers.ReLU()(inputs)
        x = self.layer1(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.layer2(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.layer3(x)
        return inputs + x


class PixelCNNModel(tf.keras.Model):
    """
    Outputs params for AR flow mixture model.
     Models real-valued pixels(N, h*w, c)
    """
    def __init__(self, H, W, C, n_outputs, factorised, n_filters, n_res, *args, **kwargs):
        """
        :param H, W, C: height, width and number of channels
        :param n_outputs: number of outputs for each pixel (usually 3 for mixture of Gaussians)
        :param factorised: whether to have factorised over channels - affects masks
        :param n_filters: number of fitlers each hidden conv layer
        :param n_res: number of residual blocks
        """
        super().__init__(*args, **kwargs)
        self.H = H
        self.W = W
        self.C = C
        self.n_outputs = n_outputs
        self.n_filters = n_filters
        self.n_res = n_res
        self.factorised = factorised

    def build(self, input_shape, **kwargs):
        self.layer1 = MaskedCNN(self.n_filters * 2, 7, True, self.factorised)
        self.res_layers = [MaskedResidualBlock(self.n_filters, self.factorised) for _ in range(self.n_res)]
        # want ReLU applied first as per paper
        self.relu_conv1x1 = [tf.keras.layers.ReLU(),
                             MaskedCNN(self.n_filters, 1, False, self.factorised)]
        # output filter same shape as image, we want 3 filters for mixture of Gaussian params each pixel
        self.output_conv = [tf.keras.layers.ReLU(),
                            MaskedCNN(self.C * self.n_outputs, 1, False, self.factorised)]

    def call(self, inputs, training=None, mask=None):
        """
        Returns output of pixelCNN (bs, H, W, C * n_out)
        :param inputs: (bs, H, W, C) images
        """
        img = tf.cast(inputs, tf.float32)
        x = self.layer1(img)
        for layer in self.res_layers:
            x = layer(x)
        for layer in self.relu_conv1x1:
            x = layer(x)
        for layer in self.output_conv:
            x = layer(x)
        return x
