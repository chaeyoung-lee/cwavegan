# Based on TensorFlow released DCGAN codes

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

NUM_TRAIN_AUDIO = 60000
NUM_EVAL_AUDIO = 10000


class InputFunction(object):
  """Wrapper class that is passed as callable to Estimator."""

  def __init__(self, is_training, noise_dim, bias):
    self.is_training = is_training
    self.noise_dim = noise_dim
    self.bias = bias
    mode = ('train' if is_training
            else 'test')

  def __call__(self, params):
    """Creates a simple Dataset pipeline."""
    window_len = 8192 if self.bias else 8182
    def parser(serialized_example):
      features = {'samples': tf.FixedLenSequenceFeature([1], tf.float32, allow_missing=True)}
      if self.bias:
        features['label'] = tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
      else:
        features['label'] = tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True)

      example = tf.parse_single_example(serialized_example, features)
      wav = example['samples']
      label = example['label']

      # first window
      wav = wav[:window_len]
      wav = tf.pad(wav, [[0, window_len - tf.shape(wav)[0]], [0, 0]])
      wav.set_shape([window_len, 1])

      if not self.bias:
        label.set_shape(10)

      return wav, label

    batch_size = params['batch_size']
    data_file = 'gs://sc09_tf_int' if self.bias else 'gs://sc09_tf/'
    data_files = []
    for i in range(128):
      data_root = data_file + 'train-{}-of-128.tfrecord'.format(str(i).zfill(3))
      data_files.append(data_root)

    dataset = tf.data.TFRecordDataset(data_files)
    dataset = dataset.map(parser).cache()
    if self.is_training:
        dataset = dataset.repeat()
    dataset = dataset.shuffle(1024)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    wav, labels = dataset.make_one_shot_iterator().get_next()
    random_noise = tf.random_uniform([batch_size, self.noise_dim], -1., 1., dtype=tf.float32)

    if self.bias:
      labels = labels + tf.constant(10, name='fixed', dtype=tf.int64)
      labels = tf.cast(labels, dtype=tf.float32)
      labels = tf.reshape(labels, [batch_size, 1])

    features = {
        'real_audio': wav,
        'random_noise': random_noise}

    return features, labels
