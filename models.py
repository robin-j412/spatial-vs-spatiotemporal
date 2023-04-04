import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import json
import pandas as pd
from data import PS

REG = None

gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

def backbone_effnet(shape):
    ### THIS CODE COME FROM : https://github.com/bnsreenu/python_for_microscopists/blob/master/Tips_tricks_20_Understanding%20transfer%20learning%20for%20different%20size%20and%20channel%20inputs.py
    ### FROM THIS VIDEO : https://www.youtube.com/watch?v=5kbpoIQUB4Q

    # Notice that the additional weights come from the first conv. layer only.
    # Let us start by copying the config information of the VGG model.
    # This way we can easily edit the model input by copying it into a new model.

    # Import vgg model by not defining an input shape.
    effnet_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet')
    # print(effnet_model.summary())

    # Get the dictionary of config for vgg16
    effnet_config = effnet_model.get_config()

    # Change the input shape to new desired shape
    h, w, c = shape
    effnet_config["layers"][0]["config"]["batch_input_shape"] = (None, h, w, c)
    effnet_config["layers"][3]["inbound_nodes"][0][3]["y"] = np.hstack(
        [effnet_config["layers"][3]["inbound_nodes"][0][3]["y"],
         np.array([np.mean(effnet_config["layers"][3]["inbound_nodes"][0][3]["y"]) for _ in range(c - 3)])])
    # Create new model with the updated configuration
    effnet_updated = tf.keras.models.Model.from_config(effnet_config)
    # print(effnet_updated.summary())

    # Check Weights of first conv layer in the original model...
    orig_model_conv1_block1_wts = effnet_model.layers[5].get_weights()[0]

    # print(orig_model_conv1_block1_wts[:, :, 0, 0])
    # print(orig_model_conv1_block1_wts[:, :, 1, 0])
    # print(orig_model_conv1_block1_wts[:, :, 2, 0])

    # Random weights....

    # New model created with updated input shape but weights are not copied from the original input.

    # Since we have more channels to our input layer, we need to either randomly
    # assign weights or get an average of all existing weights from the input layer
    # and assign to the new channels as starting point.
    # Assigning average of weights may be a better approach.

    # Function that calculates average of weights along the channel axis and then
    # copies it over n number of times. n being the new channels that need to be concatenated with the original channels.

    def avg_and_copy_wts(weights, num_channels_to_fill,layer_name):  # num_channels_to_fill are the extra channels for which we need to fill weights

        if layer_name == 'normalization':
            average_weights = np.mean(weights)
        else:
            average_weights = np.mean(weights, axis=-2).reshape(
                weights[:, :, -1:, :].shape)  # Find mean along the channel axis (second to last axis)
        wts_copied_to_mult_channels = np.tile(average_weights,
                                              (num_channels_to_fill, 1))  # Repeat (copy) the array multiple times
        return (wts_copied_to_mult_channels)

    # Get the configuration for the updated model and extract layer names.
    # We will use these names to copy over weights from the original model.
    effnet_updated_config = effnet_updated.get_config()
    effnet_updated_layer_names = [effnet_updated_config['layers'][x]['name'] for x in
                                  range(len(effnet_updated_config['layers']))]

    # Name of the first convolutional layer.
    # Remember that this is the only layer with new additional weights. All other layers
    # will have same weights as the original model.
    first_conv_name = effnet_updated_layer_names[5]
    norm_layer = effnet_updated_layer_names[2]
    # Update weights for all layers. And for the first conv layer, copy the first
    # three layer weights and fill others with the average of all three.
    for layer in effnet_model.layers:
        if layer.name in effnet_updated_layer_names:

            if layer.get_weights() != []:  # All convolutional layers and layers with weights (no input layer or any pool layers)
                target_layer = effnet_updated.get_layer(layer.name)

                if layer.name in [norm_layer]:  # For the norm layer
                    mean = layer.get_weights()[0]
                    variance = layer.get_weights()[1]

                    mean_extra_channels = np.hstack((mean,
                                                     # Keep the first 3 channel weights as-is and copy the weights for additional channels.
                                                     avg_and_copy_wts(mean, c - 3, layer_name=layer.name).ravel()))
                    variance_extra_channels = np.hstack((variance,
                                                         # Keep the first 3 channel weights as-is and copy the weights for additional channels.
                                                         avg_and_copy_wts(variance, c - 3,
                                                                          layer_name=layer.name).ravel()))

                    target_layer.set_weights(
                        [mean_extra_channels, variance_extra_channels, 0])  # Now set weights for the first conv. layer
                    target_layer.trainable = True  # You can make this trainable if you want.
                elif layer.name in [first_conv_name]:
                    weights = layer.get_weights()[0]
                    # biases = layer.get_weights()[1]

                    weights_extra_channels = np.concatenate((weights,
                                                             # Keep the first 3 channel weights as-is and copy the weights for additional channels.
                                                             avg_and_copy_wts(weights, c - 3, layer_name=layer.name)),
                                                            # - 3 as we already have weights for the 3 existing channels in our model.
                                                            axis=-2)

                    target_layer.set_weights(
                        [weights_extra_channels])  # Now set weights for the first conv. layer
                    target_layer.trainable = True  # You can make this trainable if you want.

                else:
                    target_layer.set_weights(layer.get_weights())  # Set weights to all other layers.
                    target_layer.trainable = True  # You can make this trainable if you want.

        return effnet_updated

def efficient_net_pretrain(shape, out_shape, channels='all', fine_tune_all=True, **kwargs):

    if channels == 'all':
        effnet = backbone_effnet(shape)
    else:
        effnet = tf.keras.applications.EfficientNetB0(weights='imagenet',  include_top=False)

    if not fine_tune_all:
        for layer in effnet.layers[:-3]:
            layer.trainable = False

    inp = tf.keras.layers.Input(shape)

    x = effnet(inp)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(out_shape[0], activation='relu')(x)
    model = tf.keras.models.Model(inputs=inp, outputs=x)
    print(model.summary())
    return model

class PositionalEncoding(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def positional_encoding(self, position, d_model):
        def get_angles(pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))

            return pos * angle_rates

        angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, embeddings, **kwargs):
        _, position, dims = embeddings.shape
        x = embeddings + self.positional_encoding(position, dims)
        return x

class EffNetEmbedder(layers.Layer):

    def __init__(self, shape, key_dim, channels='all', **kwargs):

        super().__init__(**kwargs)
        if channels == 'all':
            self.effnet = backbone_effnet(shape)
        else:
            self.effnet = tf.keras.applications.EfficientNetB0(weights='imagenet', input_shape=shape, include_top=False)
            for layer in self.effnet.layers[:len(self.effnet.layers)//2]:
                layer.trainable = False
        self.shape = shape
        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.dense_1 = tf.keras.layers.Dense(key_dim, activation='relu')

        self.seq = tf.keras.models.Sequential([self.effnet,
                                               self.pooling,
                                               self.dense_1])
        def compute_output_shape(self,):
            return (None, key_dim)
        self.seq.compute_output_shape = compute_output_shape

    def call(self, videos):
        embed = tf.keras.layers.TimeDistributed(self.seq)(videos)
        return embed

class PatchEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv2D(
            filters=embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            kernel_regularizer=REG,
            bias_regularizer=REG,
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches
        #return projected_patches

class ResNetBlock(layers.Layer):

    def __init__(self, kernel_size=3, filters=32, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(kernel_size=kernel_size, filters=filters, padding='same')
        self.layer_norm1 = tf.keras.layers.BatchNormalization() #Layer Normalization falls in a local minima
        self.activation = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv2D(kernel_size=kernel_size, filters=filters, padding='same')
        self.layer_norm2 = tf.keras.layers.BatchNormalization()

    def call(self, images):
        input_copy = tf.identity(images)
        x = self.conv1(images)
        x = self.layer_norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.layer_norm2(x)
        x = tf.keras.layers.Add()([x, input_copy])
        return x


def make_transformer_patch(shape, out_shape, num_heads=8, key_dim=158, depth=6,  **kwargs):
    inp = layers.Input(shape)

    patches = PatchEmbedding(embed_dim=key_dim, patch_size=PS//4)(inp)

    encoded_patches = PositionalEncoding()(patches)

    for _ in range(depth):

        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=0.1)(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])


        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=key_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=key_dim, activation=tf.nn.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x2, x3])

    x = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    x = layers.GlobalAvgPool1D(data_format='channels_first')(x)
    #x = layers.Flatten()(x) #NEVER FLATTEN

    x = tf.keras.layers.Dense(out_shape[0], activation='relu')(x)

    model = tf.keras.models.Model(inputs=inp, outputs=x)
    print(model.summary())
    return model

def make_transformer_spat(shape, out_shape, num_heads=8, key_dim=158, depth=6,  **kwargs):
    inp = layers.Input(shape)

    patches = PatchEmbedding(embed_dim=key_dim, patch_size=PS//4)(inp)

    encoded_patches = PositionalEncoding()(patches)

    for _ in range(depth):

        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=0.1)(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])


        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=key_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=key_dim, activation=tf.nn.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x2, x3])

    x = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    x = layers.GlobalAvgPool1D(data_format='channels_first')(x)
    #x = layers.Flatten()(x) #NEVER FLATTEN

    x = tf.keras.layers.Dense(out_shape[0], activation='relu')(x)

    model = tf.keras.models.Model(inputs=inp, outputs=x)
    print(model.summary())
    return model

def make_transformer_temp_only(shape, out_shape, num_heads=8, key_dim=158, depth=6,  **kwargs):
    inp = layers.Input(shape)
    #patches = tf.keras.layers.AveragePooling3D(pool_size=(1, PS//4, PS//4), strides=(1, PS//4, PS//4))(inp)
    patches = tf.keras.layers.AveragePooling3D(pool_size=(1, PS, PS), strides=(1, PS//4, PS//4))(inp)
    patches = tf.keras.layers.Reshape((-1, shape[-1]))(patches)
    patches = tf.keras.layers.Dense(key_dim)(patches)

    encoded_patches = PositionalEncoding()(patches)

    for _ in range(depth):

        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=0.1)(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])


        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=key_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=key_dim, activation=tf.nn.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x2, x3])

    x = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    x = layers.GlobalAvgPool1D(data_format='channels_first')(x)
    #x = layers.Flatten()(x) #NEVER FLATTEN

    x = tf.keras.layers.Dense(out_shape[0], activation='relu')(x)

    model = tf.keras.models.Model(inputs=inp, outputs=x)
    print(model.summary())
    return model

def make_transformer_effnet(shape, out_shape, num_heads=8, key_dim=158, depth=6, channels='all',  **kwargs):
    inp = layers.Input(shape)

    patches = EffNetEmbedder(shape[1:], key_dim, channels=channels)(inp)

    encoded_patches = PositionalEncoding()(patches)

    for _ in range(depth):

        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=0.1)(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])


        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=key_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=key_dim, activation=tf.nn.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x2, x3])


    x = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    x = layers.GlobalAvgPool1D()(x)
    #x = layers.Flatten()(x) #NEVER FLATTEN

    x = tf.keras.layers.Dense(out_shape[0], activation='relu')(x)

    model = tf.keras.models.Model(inputs=inp, outputs=x)
    print(model.summary())
    return model

def baseline_cnn(shape, out_shape, filters=[6, 32, 64, 158]):

    inp = tf.keras.layers.Input(shape)

    x = layers.Rescaling(1. / 255.)(inp)

    for filter0, filter1 in zip(filters[:-1], filters[1:]):
        x = ResNetBlock(kernel_size=3, filters=filter0)(x)
        x = tf.keras.layers.Conv2D(filters=filter1, kernel_size=3, strides=(2, 2))(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(out_shape[0], activation='relu')(x)

    model = tf.keras.models.Model(inp, x)
    print(model.summary())
    return model



if __name__ == '__main__':

    input_shape = (21, PS, PS, 6)
    output_shape = (21,)

    #model = make_transformer_effnet(input_shape, out_shape=output_shape, num_heads=4, key_dim=72, depth=2, channels='all')
    #make_effnet_spat((PS, PS, 6), out_shape=(1,))

    #baseline_cnn((PS, PS, 6), (1, ))

    #make_transformer_temp_only(input_shape, out_shape=output_shape, num_heads=4, key_dim=72, depth=2)

    make_transformer_spat(input_shape[1:], out_shape=(1,), num_heads=4, key_dim=72, depth=2)

    make_transformer_patch(shape=input_shape, out_shape=output_shape, num_heads=4, key_dim=72, depth=2)
