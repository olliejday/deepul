import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from autoregressive_flow import ARFlow

# TODO: batch norm?
# TODO: nan loss
# TODO: neg loss?
# TODO: dequantize the data and scale, account for in loss scaling

class PixelCNNARFlow(ARFlow):
    """
    Wraps the model with function handles and training code
    PixelCNN outputs with an autoregressive flow of mixture of Gaussians
    """
    def __init__(self, H, W, C, k, lr=10e-3):
        """
        :param H, W, C: height and width and # channels of image, assumed single, real-valued channel
        :param k: # gaussians in mixture
        :param lr: learning rate
        """
        self.H, self.W, self.C = H, W, C
        n_vars = self.H * self.W * self.C  # a var per pixel
        self.k = k
        super().__init__(n_vars, lr)

    def setup_model(self):
        """
        Overwrite with pixelCNN model
        :return:
        """
        return PixelCNNARFlowModel(self.H, self.W, self.C, self.k)

    def sample(self, n, seed=123):
        """
        n number of samples
        seed for PRNG
        """
        return self.model.sample(n, seed)


class PixelCNNARFlowModel(tf.keras.Model):
    """
    PixelCNN outputs a mixture of Gaussians for each (B&W, real valued) pixel for an autoregressive flow
    """
    def __init__(self, H, W, C, k, factorised=True, n_filters=64, n_res=7, **kwargs):
        """
        :param H, W, C: height and width and # channels of image, assumed single, real-valued channel
        :param k: # gaussians in mixture
        :param factorised: whether to have factorised over channels - affects masks (doesn't matter in b&w dataset)
        :param n_filters: number of filters each hidden conv layer
        :param n_res: number of residual blocks
        """
        super().__init__(**kwargs)
        self.H, self.W, self.C = H, W, C
        self.k = k
        self.factorised = factorised
        self.n_filters = n_filters
        self.n_res = n_res
        self.model = self._setup_model()  # model outputs MoG distrbn params

    def _setup_model(self):
        """
        Model takes (bs,image shape) inputs
        Outputs (bs, image shape * 3 * k) for each param (weight, mean, stddev) for each mixture (k)
        """
        return PixelCNNModel(self.H, self.W, self.C, self.factorised, self.n_filters, self.n_res, n_outputs=3*self.k)

    def _get_distribution(self, x):
        """
        :param x: (bs, n_cond)
        :return: weights_dist, dist
        categorical distribution over components tfp.distributions.Categorical
        mixture distribution over gaussians with weights distribution tfp.distributions.Mixture
        """
        # outputs of model (bs, H, W, C * 3 * k)
        outputs = self.model(x)
        # each param of shape (bs, H, W, C * k)
        weights_logits, means, log_stddevs = tf.split(outputs, 3, -1)
        stddevs = tf.exp(log_stddevs)
        # get the component distributions and a distribution for mixture weights
        components_dists = [tfp.distributions.Normal(means[:, :, :, :, i], stddevs[:, :, :, :, i]) for i in range(self.k)]
        weights_dist = tfp.distributions.Categorical(logits=weights_logits)
        # handles batching
        return tfp.distributions.Mixture(weights_dist, components_dists)

    def cdf(self, x):
        """
        z = f(x)
        Here this is the cdf of mixture model
        :param x: (bs, H, W, C) batch image inputs
        :return: (bs,)
        """
        dist = self._get_distribution(x)
        # TODO: check shapes
        return dist.cdf(x)

    def log_pdf(self, x):
        """
        returns log (det(df/dx))
        here the log det term is the pdf of mixture model
        :param x: (bs, H, W, C) batch image inputs
        :return: (bs,)
        """
        # clip for numerical stability
        log_pdf = tf.math.log(tf.maximum(self.pdf(x), 1e-9))
        return log_pdf

    def pdf(self, x):
        dist = self._get_distribution(x)
        # TODO: check shape
        return dist.prob(x)

    def sample(self, n, seed=123):
        """
        :param n: number of samples
        :return: (n, H, W, C) samples
        """
        # first pixel unconditional
        images = np.zeros((n, self.H, self.W, self.C))
        # sample and update iteratively
        for h in range(self.H):
            for w in range(self.W):
                # if factorised over channels then only need one fwd pass
                if self.factorised:
                    dist = self._get_distribution(images)
                    samples = dist.sample(1, seed=seed)[0]
                    images[:, h, w] = samples[:, h, w]
                # o/w we need to condition on prev channels
                else:
                    for c in range(self.C):
                        dist = self._get_distribution(images)
                        samples = dist.sample(1, seed=seed)[0]
                        images[:, h, w, c] = samples[:, h, w, c]
        return images

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
    def __init__(self, H, W, C, factorised, n_filters, n_res, n_outputs=3, *args, **kwargs):
        """
        :param H, W, C: height, width and number of channels
        :param n_outputs: number of outputs for each pixel (usually 3 * k for mixture of k Gaussians)
        :param factorised: whether to have factorised over channels - affects masks
        :param n_filters: number of filters each hidden conv layer
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
        :param inputs: (bs, H, W, C) images
        Returns output of pixelCNN (bs, H, W, C, n_out)
        """
        img = tf.cast(inputs, tf.float32)
        x = self.layer1(img)
        for layer in self.res_layers:
            x = layer(x)
        for layer in self.relu_conv1x1:
            x = layer(x)
        for layer in self.output_conv:
            x = layer(x)
        x = tf.reshape(x, (-1, self.H, self.W, self.C, self.n_outputs))
        return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    H, W, C = 5, 5, 1
    k = 5
    factorised = True
    model = PixelCNNARFlow(H, W, C, k)
    bs = 128
    x = np.stack([np.eye(5)] * bs).reshape((bs, H, W, C))
    # dequantise
    x = x + np.random.random((bs, H, W, C))
    # scale to [0, 1]
    x = x / np.max(x)
    for i in range(50):
        print(model.train(x))
    samples = model.sample(3)
    samples = np.squeeze(samples)
    plt.imshow(np.hstack(samples), cmap="gray")
    plt.title("samples")
    plt.show()
    # same as handout,
    # [0,0.5] represents a black pixel
    # and [0.5,1] represents a white pixel
    plot_im = np.zeros_like(samples)
    plot_im[np.where(samples > 0.5)] = 1.
    plot_im = np.hstack(plot_im)
    plt.imshow(plot_im, cmap="gray")
    plt.title("samples binarised")
    plt.show()