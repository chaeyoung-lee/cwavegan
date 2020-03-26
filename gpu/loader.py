import tensorflow as tf


"""
  Data loader
  fps: List of tfrecords
  batch_size: Resultant batch size
  window_len: Size of slice to take from each example
  first_window: If true, always take the first window in the example, otherwise take a random window
  repeat: If false, only iterate through dataset once
  labels: If true, return (x, y), else return x
  buffer_size: Number of examples to queue up (larger = more random)
"""
def get_batch(
    fps,
    batch_size,
    window_len,
    first_window=False,
    repeat=True,
    labels=True,
    buffer_size=8192):
  def _mapper(example_proto):
    features = {'samples': tf.FixedLenSequenceFeature([1], tf.float32, allow_missing=True)}
    if labels:
      features['label'] = tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True)

    example = tf.parse_single_example(example_proto, features)
    wav = example['samples']
    if labels:
      label = example['label']

    if first_window:
      # Use first window
      wav = wav[:window_len]
    else:
      # Select random window
      wav_len = tf.shape(wav)[0]

      start_max = wav_len - window_len
      start_max = tf.maximum(start_max, 0)

      start = tf.random_uniform([], maxval=start_max + 1, dtype=tf.int32)

      wav = wav[start:start+window_len]

    wav = tf.pad(wav, [[0, window_len - tf.shape(wav)[0]], [0, 0]])

    wav.set_shape([window_len, 1])
    # label.set_shape(10)

    if labels:
      return wav, label
    else:
      return wav

  dataset = tf.data.TFRecordDataset(fps)
  dataset = dataset.map(_mapper)
  if repeat:
    dataset = dataset.shuffle(buffer_size=buffer_size)
  dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  if repeat:
    dataset = dataset.repeat()
  wav, labels = dataset.make_one_shot_iterator().get_next()
  
  # bias scaling
  labels = labels + tf.constant(10, name='fixed', dtype=tf.int64)
  labels = tf.cast(labels, dtype=tf.float32)
  labels = tf.reshape(labels, [batch_size, 1])

  return wav, labels