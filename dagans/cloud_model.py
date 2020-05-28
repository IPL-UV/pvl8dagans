import tensorflow as tf
from dagans import util_model


def build_unet_model_fun(x_init, weight_decay=0.05, batch_norm=True, final_activation="sigmoid"):

    axis_batch_norm = 3

    reg = tf.keras.regularizers.l2(weight_decay)

    conv1 = util_model.conv_blocks(x_init,
                        32,
                        axis_batch_norm,
                        reg,
                        name="input",
                        batch_norm=batch_norm)

    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),name="pooling_1")(conv1)

    conv2 = util_model.conv_blocks(pool1,
                        64,
                        axis_batch_norm,reg,name="pool1",
                        batch_norm=batch_norm)

    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),name="pooling_2")(conv2)


    conv3 = util_model.conv_blocks(pool2,
                        128,
                        axis_batch_norm,reg,name="pool2",
                        batch_norm=batch_norm)

    up8 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2),
                                       padding='same',name="upconv1",
                                       kernel_regularizer=reg)(conv3), conv2],
                      axis=axis_batch_norm,name="concatenate_up_1")

    conv8 = util_model.conv_blocks(up8,
                        64,
                        axis_batch_norm,reg,name="up1",
                        batch_norm=batch_norm)

    up9 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(32, (2, 2),
                                       strides=(2, 2),
                                       padding='same',name="upconv2",
                                       kernel_regularizer=reg)(conv8), conv1],
                      axis=axis_batch_norm,name="concatenate_up_2")

    conv9 = util_model.conv_blocks(up9,
                        32,
                        axis_batch_norm,reg,name="up2",
                        batch_norm=batch_norm)

    conv10 = tf.keras.layers.Conv2D(1, (1, 1),
                                    kernel_regularizer=reg,name="linear_model",
                                    activation=final_activation)(conv9)

    return conv10


def load_model(shape=(32, 32), weight_decay=0.05, final_activation="sigmoid"):
    ip = tf.keras.layers.Input(shape+(4,), name="ip_cloud")
    c11 = tf.keras.layers.Conv2D(4, (1, 1),
                                 name="normalization_cloud", trainable=False)
    x_init = c11(ip)
    c11.set_weights([util_model.ID_KERNEL_INITIALIZER, -util_model.NORM_OFF_PROBAV])
    conv2d10 = build_unet_model_fun(x_init, weight_decay=weight_decay,
                                    final_activation=final_activation,
                                    batch_norm=True)
    return tf.keras.models.Model(inputs=[ip], outputs=[conv2d10], name="UNet-Clouds")
