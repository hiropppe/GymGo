import tensorflow as tf

from tensorflow.keras import layers, models, initializers


class Bias(layers.Layer):
    """Custom keras layer that simply adds a scalar bias to each location in the input

    Largely copied from the keras docs:
    http://keras.io/layers/writing-your-own-keras-layers/#writing-your-own-keras-layers
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='bias',
                                 shape=input_shape[1:],
                                 dtype=np.float32,
                                 initializer=tf.zeros_initializer,
                                 trainable=True)
        super().build(input_shape)  # Be sure to call this somewhere!

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, x):
        return x + self.W


def cnn_policy(**kwargs):
    """construct a convolutional neural network.

    Keword Arguments:
    - input_dim:             depth of features to be processed by first layer (default 48)
    - board:                 width of the go board to be processed (default 19)
    - filters_per_layer:     number of filters used on every layer (default 128)
    - filters_per_layer_K:   (where K is between 1 and <layers>) number of filters
                              used on layer K (default #filters_per_layer)
    - layers:                number of convolutional steps (default 12)
    - filter_width_K:        (where K is between 1 and <layers>) width of filter on
                              layer K (default 3 except 1st layer which defaults to 5).
                              Must be odd.
    """
    defaults = {
        "input_dim": 48,
        "board": 19,
        "filters_per_layer": 192,
        "layers": 12,
        "filter_width_1": 5
    }
    # copy defaults, but override with anything in kwargs
    params = defaults
    params.update(kwargs)

    # create the network:
    # a series of zero-paddings followed by convolutions
    # such that the output dimensions are also board x board
    model = models.Sequential()

    # create input layer
    model.add(layers.Input(shape=(params["board"], params["board"], params["input_dim"])))

    # create first layer
    model.add(layers.Conv2D(
        params["filters_per_layer"],
        (params["filter_width_1"], params["filter_width_1"]),
        kernel_initializer=initializers.RandomUniform(minval=-0.05, maxval=0.05),
        activation='relu',
        padding='same',
        name='Conv2D_1'))

    #model.add(layers.BatchNormalization())

    # create all other layers
    for i in range(2, params["layers"] + 1):
        # use filter_width_K if it is there, otherwise use 3
        filter_key = "filter_width_%d" % i
        filter_width = params.get(filter_key, 3)

        # use filters_per_layer_K if it is there, otherwise use default value
        filter_count_key = "filters_per_layer_%d" % i
        filter_nb = params.get(filter_count_key, params["filters_per_layer"])

        model.add(layers.Conv2D(
            filter_nb,
            (filter_width, filter_width),
            kernel_initializer=initializers.RandomUniform(minval=-0.05, maxval=0.05),
            activation='relu',
            padding='same',
            name='Conv2D_' + str(i)))
    
        #model.add(layers.BatchNormalization())

    # the last layer maps each <filters_per_layer> feature to a number
    model.add(layers.Conv2D(
        1,
        (1, 1),
        kernel_initializer=initializers.RandomUniform(minval=-0.05, maxval=0.05),
        padding='same',
        name='Conv2D_13'))
    
    #model.add(layers.BatchNormalization())
    # reshape output to be board x board
    model.add(layers.Flatten())
    # add a bias to each board location
    #model.add(Bias())
    model.add(layers.Dense(units=params["board"]**2+1, use_bias=True))
    #model.add(layers.BatchNormalization())
    # softmax makes it into a probability distribution
    model.add(layers.Activation('softmax'))

    return model


def resnet_policy(**kwargs):
    """Residual network architecture as per He at al. 2015

    construct a convolutional neural network with Resnet-style skip connections.
    Arguments are the same as with the default CNNPolicy network, except the default
    number of layers is 20 plus a new n_skip parameter

    Keword Arguments:
    - input_dim:             depth of features to be processed by first layer (no default)
    - board:                 width of the go board to be processed (default 19)
    - filters_per_layer:     number of filters used on every layer (default 128)
    - layers:                number of convolutional steps (default 20)
    - filter_width_K:        (where K is between 1 and <layers>) width of filter on
                            layer K (default 3 except 1st layer which defaults to 5).
                            Must be odd.
    - n_skip_K:             (where K is as in filter_width_K) number of convolutional
                            layers to skip with the linear path starting at K. Only valid
                            at K >= 1. (Each layer defaults to 1)

    Note that n_skip_1=s means that the next valid value of n_skip_* is 3

    A diagram may help explain (numbers indicate layer):

        1        2        3           4        5        6
    I--C--B--R--C--B--R--C--M--B--R--C--B--R--C--B--R--C--M  ...  M --R--F--O
        \__________________/ \___________________________/ \ ... /
            [n_skip_1 = 2]          [n_skip_3 = 3]

    I - input
    B - BatchNormalization
    R - ReLU
    C - Conv2D
    F - Flatten
    O - output
    M - merge

    The input is always passed through a Conv2D layer, the output of which
    layer is counted as '1'.  Each subsequent [R -- C] block is counted as
    one 'layer'. The 'merge' layer isn't counted; hence if n_skip_1 is 2,
    the next valid skip parameter is n_skip_3, which will start at the
    output of the merge

    """
    defaults = {
        "input_dim": 48,
        "board": 19,
        "filters_per_layer": 128,
        "layers": 20,
        "filter_width_1": 5
    }
    # copy defaults, but override with anything in kwargs
    params = defaults
    params.update(kwargs)

    # create the network using Keras' functional API,
    # since this isn't 'Sequential'
    model_input = layers.Input(shape=(params["board"], params["board"], params["input_dim"]))

    # create first layer
    convolution_path = layers.Conv2D(
        params["filters_per_layer"],
        (params["filter_width_1"], params["filter_width_1"]),
        kernel_initializer=initializers.RandomUniform(minval=-0.05, maxval=0.05),
        activation='linear',  # relu activations done inside resnet modules
        padding='same')(model_input)

    def add_resnet_unit(path, K, **params):
        """Add a resnet unit to path starting at layer 'K',
        adding as many (ReLU + Conv2D) modules as specified by n_skip_K

        Returns new path and next layer index, i.e. K + n_skip_K, in a tuple
        """
        # loosely based on https://github.com/keunwoochoi/residual_block_keras
        # see also # keras docs here:
        # http://keras.io/getting-started/functional-api-guide/#all-models-are-callable-just-like-layers

        block_input = path
        # use n_skip_K if it is there, default to 1
        skip_key = "n_skip_%d" % K
        n_skip = params.get(skip_key, 1)
        for i in range(n_skip):
            layer = K + i
            # add BatchNorm
            path = layers.BatchNormalization()(path)
            # add ReLU
            path = layers.Activation('relu')(path)
            # use filter_width_K if it is there, otherwise use 3
            filter_key = "filter_width_%d" % layer
            filter_width = params.get(filter_key, 3)
            # add Conv2D
            path = layers.Conv2D(
                params["filters_per_layer"],
                (params["filter_width_1"], params["filter_width_1"]),
                kernel_initializer=initializers.RandomUniform(minval=-0.05, maxval=0.05),
                activation='linear',
                padding='same')(path)
        # Merge 'input layer' with the path
        path = layers.add([block_input, path])
        return path, K + n_skip

    # create all other layers
    layer = 1
    while layer < params['layers']:
        convolution_path, layer = add_resnet_unit(convolution_path, layer, **params)
    if layer > params['layers']:
        print("Due to skipping, ended with {} layers instead of {}"
                .format(layer, params['layers']))

    # since each layer's activation was linear, need one more ReLu
    convolution_path = layers.Activation('relu')(convolution_path)

    # the last layer maps each <filters_per_layer> featuer to a number
    convolution_path = layers.Conv2D(
        1,
        (1, 1),
        kernel_initializer=initializers.RandomUniform(minval=-0.05, maxval=0.05),
        padding='same')(convolution_path)
    # flatten output
    network_output = layers.Flatten()(convolution_path)
    # add a bias to each board location
    network_output = Bias()(network_output)
    # softmax makes it into a probability distribution
    network_output = layers.Activation('softmax')(network_output)

    model = models.Model(model_input, network_output)
    return model

