import numpy as np
import tensorflow as tf


class DenseNN(tf.keras.layers.Layer):
    def __init__(self, n_units, n_out, activation=None, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.n_units = n_units
        self.n_out = n_out
        self._activation = activation

    def build(self, input_shape):
        self.layers_list = []
        self.layers_list.append(tf.keras.layers.Dense(self.n_units, activation=self._activation))
        self.layers_list.append(tf.keras.layers.Dense(self.n_units, activation=self._activation))
        self.layers_list.append(tf.keras.layers.Dense(self.n_out))

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x


def train_model(model, train_data, test_data, dset_id, n_epochs, bs):
    """
    model must have .train(batch) returns loss
        .loss(batch) returns loss
        .log_p_x returns log(p(x))
        .f_x returns z = f(x)
    train_data: An (n_train, 2) numpy array of floats in R^2
    test_data: An (n_test, 2) numpy array of floats in R^2
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
            used to set different hyperparameters for different datasets, or
            for plotting a different region of densities

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (?,) of probabilities with values in [0, +infinity).
        Refer to the commented hint.
    - a numpy array of size (n_train, 2) of floats in [0,1]^2. This represents
        mapping the train set data points through our flow to the latent space.
    """
    # create data loaders
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
    train_iter = train_dataset.shuffle(bs * 2).batch(bs)

    # train model
    train_losses = []
    test_losses = []
    for epoch in range(n_epochs):
        for i, batch in enumerate(train_iter):
            train_loss = model.train(batch).numpy()
            train_losses.append(train_loss)
        test_loss = model.loss(test_data).numpy()
        test_losses.append(test_loss)

    # final test loss
    test_loss = model.loss(test_data).numpy()
    test_losses.append(test_loss)

    # heatmap
    dx, dy = 0.025, 0.025
    if dset_id == 1:  # face
        x_lim = (-4, 4)
        y_lim = (-4, 4)
    elif dset_id == 2:  # two moons
        x_lim = (-1.5, 2.5)
        y_lim = (-1, 1.5)
    y, x = np.mgrid[slice(y_lim[0], y_lim[1] + dy, dy),
                    slice(x_lim[0], x_lim[1] + dx, dx)]
    mesh_xs = np.stack([x, y], axis=2).reshape(-1, 2)
    densities = np.exp(model.log_p_x(mesh_xs).numpy())
    print(np.shape(densities))

    # latents
    # unshuffled train dataset
    train_iter = train_dataset.batch(bs)
    zs = []
    for batch in train_iter:
        zs.append(model.f_x(batch))
    latents = np.concatenate(zs)

    return train_losses, test_losses, densities, latents