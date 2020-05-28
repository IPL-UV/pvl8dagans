from tensorflow import keras
import tensorflow as tf
from dagans import util_model


def probav_rgb(img):
    return tf.clip_by_value(tf.stack((img[..., 1],
                                      img[..., 1] * .25 + img[..., 2] * .75,
                                      img[..., 0]),
                            axis=-1),
                   0,1)


def d_layer(layer_input, filters, f_size=4,bn=True,name="d_layer"):
    """Discriminator layer"""
    d = keras.layers.Conv2D(filters,
                            kernel_size=f_size,
                            strides=2, padding='same',
                            name=name+"_conv2d")(layer_input)
    if bn:
        d = keras.layers.BatchNormalization(momentum=0.8,
                                            name = name + "_bn")(d)
        
    d = keras.layers.LeakyReLU(alpha=0.2,name=name+"_leakyrelu")(d)

    return d


def disc_model(shape=(32,32,4), df=16, activation=None,normalization=True,depth=3):
    ip = keras.layers.Input(shape,name="ip_disc")
    
    if normalization:
        c11 = keras.layers.Conv2D(shape[-1], (1, 1),
                                  name="normalization_gen_in", trainable=False)
        x_init = c11(ip)
        c11.set_weights([util_model.ID_KERNEL_INITIALIZER, -util_model.NORM_OFF_PROBAV])
    else:
        x_init = ip
    
    d1 = d_layer(x_init, df, bn=False,name="d_layer_1")
    
    for cd in range(depth):
        d1 = d_layer(d1, df * 2**(cd+1),
                     name="d_layer_%d"%(cd+2))

    validity = keras.layers.Conv2D(1,
                                   kernel_size=1,
                                   strides=1,
                                   activation=activation,
                                   name="conv1x1")(d1) # (None, 2, 2, 1)

    # validity = keras.layers.GlobalAveragePooling2D()(validity) # (None,  1)

    return keras.models.Model(inputs=[ip],outputs=[validity],
                              name="discriminator")


def generator_simple(shape=(32, 32, 4), df=16, l2reg=None, normtype="batchnorm"):
    ip = keras.layers.Input(shape, name="ip_gen")

    if l2reg is None:
        reg = None
    else:
        reg = tf.keras.regularizers.l2(l2reg)

    # Input centering
    c11 = keras.layers.Conv2D(shape[-1], (1, 1),
                              name="normalization_gen_in", trainable=False)
    x_init = c11(ip)
    c11.set_weights([util_model.ID_KERNEL_INITIALIZER, -util_model.NORM_OFF_PROBAV])

    x = util_model.conv_blocks(x_init, df, -1, reg=reg,
                               name="generator_block_1", batch_norm=normtype != "no",
                               normtype=normtype)
    x2 = keras.layers.concatenate([x_init, x], axis=-1, name="gen_concatenate_1")

    x3 = util_model.conv_blocks(x2, df, -1, reg=reg,
                                name="generator_block_2", batch_norm=normtype != "no",
                                dilation_rate=(2,2),
                                normtype=normtype)
    x3 = keras.layers.concatenate([x3, x2], axis=-1, name="gen_concatenate_2")

    out_conv = keras.layers.Conv2D(shape[-1], (1, 1), name="out_generator",
                                   kernel_regularizer=reg)
    out = out_conv(x3)

    # Output centering
    c11 = keras.layers.Conv2D(4, (1, 1),
                              name="normalization_gen_out", trainable=False)
    out = c11(out)
    c11.set_weights([util_model.ID_KERNEL_INITIALIZER, util_model.NORM_OFF_PROBAV])

    return keras.models.Model(inputs=[ip],outputs=[out])


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_real_output), 
                                                                       logits=disc_real_output))
    generated_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(disc_generated_output),
                                                                            logits=disc_generated_output))

    total_disc_loss = tf.reduce_mean(.5*(real_loss + generated_loss))
    
    return total_disc_loss


def generator_gan_loss(disc_generated_output):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_generated_output),
                                                                  logits=disc_generated_output))


def mae_loss(gen_input, gen_output):
    return tf.reduce_mean(tf.abs(gen_input - gen_output))


def _interpolate(a, b):
    shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
    alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
    inter = a + alpha * (b - a)
    inter.set_shape(a.shape)
    return inter


def _gradient_penalty(f, inputs, center=0.):
    with tf.GradientTape() as t:
        t.watch(inputs)
        pred = tf.reduce_mean(f(inputs)) # the discriminator is PatchGAN
    grad = t.gradient(pred, inputs)
    norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
    gp = tf.reduce_mean((norm - center)**2)

    return gp


def gradient_penalty_fun(mode):
    """
    Implementation taken from: https://github.com/LynnHo/CycleGAN-Tensorflow-2/blob/master/tf2gan/loss.py
    :param discriminator:
    :param real:
    :param fake:
    :param mode:
    :return:
    """

    if mode == 'none':
        gp = lambda discriminator, real, fake: tf.constant(0, dtype=real.dtype)
    elif mode == "meschederreal": # https://arxiv.org/pdf/1801.04406.pdf
        gp = lambda discriminator, real, fake: _gradient_penalty(discriminator, real, center=0.)
    else:
        raise NotImplementedError("Gradient penalty loss %s not recognized" % mode)

    return gp