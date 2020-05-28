import tensorflow as tf
import numpy as np

NORM_OFF_PROBAV = np.array([0.43052389, 0.40560079, 0.46504526, 0.23876471])
ID_KERNEL_INITIALIZER =np.expand_dims(np.expand_dims(np.eye(4),
                                                     axis=0),
                                      axis=0)


class InstanceNormalization(tf.keras.layers.Layer):
  """
  Instance Normalization Layer (https://arxiv.org/abs/1607.08022).
  Taken from : tensorflow_examples.models.pix2pix import pix2pix.
  """

  def __init__(self, **kwargs):
    super(InstanceNormalization, self).__init__(**kwargs)
    self.epsilon = 1e-5

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)


def conv_blocks(ip_, nfilters, axis_batch_norm, reg, name, batch_norm,
                remove_bias_if_batch_norm=False, dilation_rate=(1,1),normtype="batchnorm"):
    use_bias = not (remove_bias_if_batch_norm and batch_norm)

    conv = tf.keras.layers.SeparableConv2D(nfilters, (3, 3),
                                           padding='same',
                                           name=name+"_conv_1",
                                           depthwise_regularizer=reg,
                                           pointwise_regularizer=reg,
                                           dilation_rate=dilation_rate,
                                           use_bias=use_bias)(ip_)

    if batch_norm:
        if normtype == "batchnorm":
            conv = tf.keras.layers.BatchNormalization(axis=axis_batch_norm,name=name + "_bn_1")(conv)
        elif normtype == "instancenorm":
            conv = InstanceNormalization(name=name + "_bn_1")(conv)
        else:
            raise NotImplementedError("Unknown norm %s" % normtype)

    conv = tf.keras.layers.Activation('relu',name=name + "_act_1")(conv)


    conv = tf.keras.layers.SeparableConv2D(nfilters, (3, 3),
                           padding='same',name=name+"_conv_2",
                           use_bias=use_bias,dilation_rate=dilation_rate,
                           depthwise_regularizer=reg,pointwise_regularizer=reg)(conv)

    if batch_norm:
        if normtype == "batchnorm":
            conv = tf.keras.layers.BatchNormalization(axis=axis_batch_norm,name=name + "_bn_2")(conv)
        elif normtype == "instancenorm":
            conv = InstanceNormalization(name=name + "_bn_2")(conv)
        else:
            raise NotImplementedError("Unknown norm %s" % normtype)


    return tf.keras.layers.Activation('relu',name=name + "_act_2")(conv)