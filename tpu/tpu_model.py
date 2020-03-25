# Based on WaveGAN codes

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def tf_repeat(output, idx, dim1, dim2, bias):
    # tensor equivalent of np.repeat
    # 1d to 3d array tensor
    if bias:
        idx = tf.tile(idx, [1, dim1 * dim2])
        idx = tf.reshape(idx, [-1, dim1, dim2])
        return output * idx
    else:
        return output

def conv1d_transpose(
    inputs,
    filters,
    kernel_width,
    stride=4,
    padding='same',
    upsample='zeros'):
    if upsample == 'zeros':
        return tf.layers.conv2d_transpose(
            tf.expand_dims(inputs, axis=1),
            filters,
            (1, kernel_width),
            strides=(1, stride),
            padding='same'
        )[:, 0]
    else:
        raise NotImplementedError


def lrelu(inputs, alpha=0.2):
  return tf.maximum(alpha * inputs, inputs)


def apply_phaseshuffle(x, rad, pad_type='reflect'):
  b, x_len, nch = x.get_shape().as_list()

  phase = tf.random_uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
  pad_l = tf.maximum(phase, 0)
  pad_r = tf.maximum(-phase, 0)
  phase_start = pad_r
  x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

  x = x[:, phase_start:phase_start+x_len]
  x.set_shape([b, x_len, nch])

  return x


"""
  Input: [None, 8192, 1]
  Output: [None] (linear output)
"""
def discriminator_wavegan(
    x,
    labels,
    kernel_len=25,
    dim=64,
    use_batchnorm=True,
    phaseshuffle_rad=0,
    reuse=False,
    scope='Discriminator',
    bias=False):
    with tf.variable_scope(scope, reuse=reuse):
        batch_size = tf.shape(x)[0]

        if use_batchnorm:
            batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
        else:
            batchnorm = lambda x: x

        if phaseshuffle_rad > 0:
            phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
        else:
            phaseshuffle = lambda x: x

        with tf.variable_scope('discriminator_0', reuse=reuse):
            # Layer 0
            # [8192, 1] -> [4096, 64]
            output = x
            output = tf.layers.conv1d(output, dim, kernel_len, 2, padding='SAME', name='downconv_0')
            output = tf_repeat(output, labels, 4096, dim, bias)
            output = lrelu(output)
            output = phaseshuffle(output)

            # Layer 1
            # [4096, 64] -> [1024, 128]
            output = tf.layers.conv1d(output, dim * 2, kernel_len, 4, padding='SAME', name='downconv_1')
            output = tf_repeat(output, labels, 1024, dim * 2, bias)
            output = batchnorm(output)
            output = lrelu(output)
            output = phaseshuffle(output)

            # Layer 2
            # [1024, 128] -> [256, 256]
            output = tf.layers.conv1d(output, dim * 4, kernel_len, 4, padding='SAME', name='downconv_2')
            output = tf_repeat(output, labels, 256, dim * 4, bias)
            output = batchnorm(output)
            output = lrelu(output)
            output = phaseshuffle(output)

            # Layer 3
            # [256, 256] -> [64, 512]
            output = tf.layers.conv1d(output, dim * 8, kernel_len, 4, padding='SAME', name='downconv_3')
            output = tf_repeat(output, labels, 64, dim * 8, bias)
            output = batchnorm(output)
            output = lrelu(output)
            output = phaseshuffle(output)

            # Layer 4
            # [64, 512] -> [16, 1024]
            output = tf.layers.conv1d(output, dim * 16, kernel_len, 4, padding='SAME', name='downconv_4')
            output = tf_repeat(output, labels, 16, dim * 16, bias)
            output = batchnorm(output)
            output = lrelu(output)

        # Flatten
        output = tf.reshape(output, [batch_size, 4 * 4 * dim * 16])

        # Connect to single logit
        with tf.variable_scope('output', reuse=reuse):
            output = tf.layers.dense(output, 1)[:, 0]

        # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

        return output


"""
  Input: [None, 100]
  Output: [None, 8192, 1]
"""
def generator_wavegan(
    z,
    labels,
    kernel_len=25,
    dim=64,
    use_batchnorm=True,
    upsample='zeros',
    train=False,
    scope='Generator',
    bias=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        batch_size = tf.shape(z)[0]

        if use_batchnorm:
            batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
        else:
            batchnorm = lambda x: x

        # FC and reshape for convolution
        # [100] -> [16, 1024]
        output = z
        with tf.variable_scope('z_project'):
            output = tf.layers.dense(output, 4 * 4 * dim * 16)
            output = tf.reshape(output, [batch_size, 16, dim * 16])
            output = batchnorm(output)
        output = tf_repeat(output, labels, 16, dim * 16, bias)
        output = tf.nn.relu(output)

        # Layer 0
        # [16, 1024] -> [64, 512]
        with tf.variable_scope('upconv_0'):
            output = conv1d_transpose(output, dim * 8, kernel_len, 4, upsample=upsample)
            output = batchnorm(output)
        output = tf_repeat(output, labels, 64, dim * 8, bias)
        output = tf.nn.relu(output)

        # Layer 1
        # [64, 512] -> [256, 256]
        with tf.variable_scope('upconv_1'):
            output = conv1d_transpose(output, dim * 4, kernel_len, 4, upsample=upsample)
            output = batchnorm(output)
        output = tf_repeat(output, labels, 256, dim * 4, bias)
        output = tf.nn.relu(output)

        # Layer 2
        # [256, 256] -> [1024, 128]
        with tf.variable_scope('upconv_2'):
            output = conv1d_transpose(output, dim * 2, kernel_len, 4, upsample=upsample)
            output = batchnorm(output)
        output = tf_repeat(output, labels, 1024, dim * 2, bias)
        output = tf.nn.relu(output)

        # Layer 3
        # [1024, 128] -> [4096, 64]
        with tf.variable_scope('upconv_3'):
            output = conv1d_transpose(output, dim, kernel_len, 4, upsample=upsample)
            output = batchnorm(output)
        output = tf_repeat(output, labels, 4096, dim, bias)
        output = tf.nn.relu(output)

        # Layer 4
        # [4096, 64] -> [8192, 1]
        with tf.variable_scope('upconv_4'):
            # output = conv1d_transpose(output, 1, kernel_len, 4, upsample=upsample)
            output = conv1d_transpose(output, 1, kernel_len, 2, upsample=upsample)
        output = tf_repeat(output, labels, 8192, 1, bias)
        output = tf.nn.tanh(output)

        # Automatically update batchnorm moving averages every time G is used during training
        if train and use_batchnorm:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if len(update_ops) != 10:
                raise Exception('Other update ops found in graph')
            with tf.control_dependencies(update_ops):
                output = tf.identity(output)

        return output

