from __future__ import print_function
try:
	import cPickle as pickle
except:
	import pickle
import os
import time

import numpy as np
import tensorflow as tf
from six.moves import xrange

import loader
from wavegan import WaveGANGenerator, WaveGANDiscriminator
from functools import reduce


"""
  Constants
"""
_FS = 16000
_WINDOW_LEN = 8192
_D_Z = 90
_D_Y = 10


"""
  Trains a WaveGAN
"""
def train(fps, args):
  with tf.name_scope('loader'):
    x, y = loader.get_batch(fps, args.train_batch_size, _WINDOW_LEN, args.data_first_window, labels=True)

  # bias scaling
  y = y + tf.constant(10, name='fixed', dtype=tf.int64)
  y = tf.cast(y, dtype=tf.float32)
  y = tf.reshape(y, [args.train_batch_size, 1])
  
  # Make inputs
  z = tf.random_uniform([args.train_batch_size, _D_Z], -1., 1., dtype=tf.float32)

  # Concatenate labels
  # y_fill = tf.expand_dims(y, axis=2)
  # x = tf.concat([x, y_fill], 1)
  # z = tf.concat([z, y], 1)

  # Make generator
  with tf.variable_scope('G'):
    G_z = WaveGANGenerator(z, y, train=True, **args.wavegan_g_kwargs)
    if args.wavegan_genr_pp:
      with tf.variable_scope('pp_filt'):
        G_z = tf.layers.conv1d(G_z, 1, args.wavegan_genr_pp_len, use_bias=False, padding='same')
  G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

  # Print G summary
  print('-' * 80)
  print('Generator vars')
  nparams = 0
  for v in G_vars:
    v_shape = v.get_shape().as_list()
    v_n = reduce(lambda x, y: x * y, v_shape)
    nparams += v_n
    print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
  print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))

  # Summarize
  tf.summary.audio('x', x, _FS)
  tf.summary.audio('G_z', G_z, _FS)
  G_z_rms = tf.sqrt(tf.reduce_mean(tf.square(G_z[:, :, 0]), axis=1))
  x_rms = tf.sqrt(tf.reduce_mean(tf.square(x[:, :, 0]), axis=1))
  tf.summary.histogram('x_rms_batch', x_rms)
  tf.summary.histogram('G_z_rms_batch', G_z_rms)
  tf.summary.scalar('x_rms', tf.reduce_mean(x_rms))
  tf.summary.scalar('G_z_rms', tf.reduce_mean(G_z_rms))

  # Make real discriminator
  with tf.name_scope('D_x'), tf.variable_scope('D'):
    D_x = WaveGANDiscriminator(x, y, **args.wavegan_d_kwargs)
  D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')

  # Print D summary
  print('-' * 80)
  print('Discriminator vars')
  nparams = 0
  for v in D_vars:
    v_shape = v.get_shape().as_list()
    v_n = reduce(lambda x, y: x * y, v_shape)
    nparams += v_n
    print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
  print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))
  print('-' * 80)

  # Make fake discriminator
  with tf.name_scope('D_G_z'), tf.variable_scope('D', reuse=True):
    D_G_z = WaveGANDiscriminator(G_z, y, **args.wavegan_d_kwargs)

  # Create loss
  D_clip_weights = None
  if args.wavegan_loss == 'dcgan':
    fake = tf.zeros([args.train_batch_size], dtype=tf.float32)
    real = tf.ones([args.train_batch_size], dtype=tf.float32)

    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=D_G_z,
      labels=real
    ))

    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=D_G_z,
      labels=fake
    ))
    D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=D_x,
      labels=real
    ))

    D_loss /= 2.
  elif args.wavegan_loss == 'lsgan':
    G_loss = tf.reduce_mean((D_G_z - 1.) ** 2)
    D_loss = tf.reduce_mean((D_x - 1.) ** 2)
    D_loss += tf.reduce_mean(D_G_z ** 2)
    D_loss /= 2.
  elif args.wavegan_loss == 'wgan':
    G_loss = -tf.reduce_mean(D_G_z)
    D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)

    with tf.name_scope('D_clip_weights'):
      clip_ops = []
      for var in D_vars:
        clip_bounds = [-.01, .01]
        clip_ops.append(
          tf.assign(
            var,
            tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
          )
        )
      D_clip_weights = tf.group(*clip_ops)
  elif args.wavegan_loss == 'wgan-gp':
    G_loss = -tf.reduce_mean(D_G_z)
    D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)

    alpha = tf.random_uniform(shape=[args.train_batch_size, 1, 1], minval=0., maxval=1.)
    differences = G_z - x
    interpolates = x + (alpha * differences)
    with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
      D_interp = WaveGANDiscriminator(interpolates, y, **args.wavegan_d_kwargs)

    LAMBDA = 10
    gradients = tf.gradients(D_interp, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
    D_loss += LAMBDA * gradient_penalty
  else:
    raise NotImplementedError()

  tf.summary.scalar('G_loss', G_loss)
  tf.summary.scalar('D_loss', D_loss)

  # Create (recommended) optimizer
  if args.wavegan_loss == 'dcgan':
    G_opt = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5)
    D_opt = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5)
  elif args.wavegan_loss == 'lsgan':
    G_opt = tf.train.RMSPropOptimizer(
        learning_rate=1e-4)
    D_opt = tf.train.RMSPropOptimizer(
        learning_rate=1e-4)
  elif args.wavegan_loss == 'wgan':
    G_opt = tf.train.RMSPropOptimizer(
        learning_rate=5e-5)
    D_opt = tf.train.RMSPropOptimizer(
        learning_rate=5e-5)
  elif args.wavegan_loss == 'wgan-gp':
    G_opt = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9)
    D_opt = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9)
  else:
    raise NotImplementedError()

  # Create training ops
  G_train_op = G_opt.minimize(G_loss, var_list=G_vars,
      global_step=tf.train.get_or_create_global_step())
  D_train_op = D_opt.minimize(D_loss, var_list=D_vars)

  # Run training
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=args.train_dir,
      save_checkpoint_secs=args.train_save_secs,
      save_summaries_secs=args.train_summary_secs) as sess:
    while True:
      # Train discriminator
      for i in xrange(args.wavegan_disc_nupdates):
        sess.run(D_train_op)
        
        # Enforce Lipschitz constraint for WGAN
        if D_clip_weights is not None:
          sess.run(D_clip_weights)
        
      # Train generator
      sess.run(G_train_op)


"""
  Creates and saves a MetaGraphDef for simple inference
  Tensors:
    'samp_z_n' int32 []: Sample this many latent vectors
    'samp_z' float32 [samp_z_n, 100]: Resultant latent vectors
    'z:0' float32 [None, 100]: Input latent vectors
    'y:0' float32 [None, 1]: Label vectors
	  'flat_pad:0' int32 []: Number of padding samples to use when flattening batch to a single audio file
    'G_z:0' float32 [None, 16384, 1]: Generated outputs
    'G_z_int16:0' int16 [None, 16384, 1]: Same as above but quantizied to 16-bit PCM samples
    'G_z_flat:0' float32 [None, 1]: Outputs flattened into single audio file
    'G_z_flat_int16:0' int16 [None, 1]: Same as above but quantized to 16-bit PCM samples
  Example usage:
    import tensorflow as tf
    tf.reset_default_graph()

    saver = tf.train.import_meta_graph('infer.meta')
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession()
    saver.restore(sess, 'model.ckpt-10000')

    z_n = graph.get_tensor_by_name('samp_z_n:0')
    _z = sess.run(graph.get_tensor_by_name('samp_z:0'), {z_n: 10})

    z = graph.get_tensor_by_name('G_z:0')
    _G_z = sess.run(graph.get_tensor_by_name('G_z:0'), {z: _z})
"""
def infer(args):
  infer_dir = os.path.join(args.train_dir, 'infer')
  if not os.path.isdir(infer_dir):
    os.makedirs(infer_dir)

  # Subgraph that generates latent vectors
  samp_z_n = tf.placeholder(tf.int32, [], name='samp_z_n')
  samp_z = tf.random_uniform([samp_z_n, _D_Z], -1.0, 1.0, dtype=tf.float32, name='samp_z')

  # Input zo
  z = tf.placeholder(tf.float32, [None, _D_Z], name='z')
  flat_pad = tf.placeholder(tf.int32, [], name='flat_pad')
  y = tf.placeholder(tf.float32, [None, 1], name='y')

  # Execute generator
  with tf.variable_scope('G'):
    G_z = WaveGANGenerator(z, y, train=False, **args.wavegan_g_kwargs)
    if args.wavegan_genr_pp:
      with tf.variable_scope('pp_filt'):
        G_z = tf.layers.conv1d(G_z, 1, args.wavegan_genr_pp_len, use_bias=False, padding='same')
  G_z = tf.identity(G_z, name='G_z')

  # Flatten batch
  nch = int(G_z.get_shape()[-1])
  G_z_padded = tf.pad(G_z, [[0, 0], [0, flat_pad], [0, 0]])
  G_z_flat = tf.reshape(G_z_padded, [-1, nch], name='G_z_flat')

  # Encode to int16
  def float_to_int16(x, name=None):
    x_int16 = x * 32767.
    x_int16 = tf.clip_by_value(x_int16, -32767., 32767.)
    x_int16 = tf.cast(x_int16, tf.int16, name=name)
    return x_int16
  G_z_int16 = float_to_int16(G_z, name='G_z_int16')
  G_z_flat_int16 = float_to_int16(G_z_flat, name='G_z_flat_int16')

  # Create saver
  G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G')
  global_step = tf.train.get_or_create_global_step()
  saver = tf.train.Saver(G_vars + [global_step])

  # Export graph
  tf.train.write_graph(tf.get_default_graph(), infer_dir, 'infer.pbtxt')

  # Export MetaGraph
  infer_metagraph_fp = os.path.join(infer_dir, 'infer.meta')
  tf.train.export_meta_graph(
      filename=infer_metagraph_fp,
      clear_devices=True,
      saver_def=saver.as_saver_def())

  # Reset graph (in case training afterwards)
  tf.reset_default_graph()


"""
  Generates a preview audio file every time a checkpoint is saved
"""
def preview(args):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from scipy.io.wavfile import write as wavwrite
  from scipy.signal import freqz

  preview_dir = os.path.join(args.train_dir, 'preview')
  if not os.path.isdir(preview_dir):
    os.makedirs(preview_dir)

  # Load graph
  infer_metagraph_fp = os.path.join(args.train_dir, 'infer', 'infer.meta')
  graph = tf.get_default_graph()
  saver = tf.train.import_meta_graph(infer_metagraph_fp)

  # Generate or restore z_i and z_o
  z_fp = os.path.join(preview_dir, 'z.pkl')
  if os.path.exists(z_fp):
    with open(z_fp, 'rb') as f:
      _zs = pickle.load(f)
  else:
    # Sample z
    samp_feeds = {}
    samp_feeds[graph.get_tensor_by_name('samp_z_n:0')] = args.preview_n
    samp_fetches = {}
    samp_fetches['zs'] = graph.get_tensor_by_name('samp_z:0')
    with tf.Session() as sess:
      _samp_fetches = sess.run(samp_fetches, samp_feeds)
    _zs = _samp_fetches['zs']

    # Save z
    with open(z_fp, 'wb') as f:
      pickle.dump(_zs, f)

  # Create labels
  sample_n = 20
  _zs = _zs[:sample_n]
  _ys = np.zeros([sample_n])
  for i in range(10):
    # one-hot vector
    # _ys[2 * i + 1][i] = 1
    # _ys[2 * i][i] = 1

    # integer labels
    _ys[2 * i] = 10 + i
    _ys[2 * i + 1] = 10 + i
  _ys = np.expand_dims(_ys, axis=1)

  # Set up graph for generating preview images
  feeds = {}
  feeds[graph.get_tensor_by_name('z:0')] = _zs
  feeds[graph.get_tensor_by_name('y:0')] = _ys
  feeds[graph.get_tensor_by_name('flat_pad:0')] = _WINDOW_LEN // 2
  fetches = {}
  fetches['step'] = tf.train.get_or_create_global_step()
  fetches['G_z'] = graph.get_tensor_by_name('G_z:0')
  fetches['G_z_flat_int16'] = graph.get_tensor_by_name('G_z_flat_int16:0')
  if args.wavegan_genr_pp:
    fetches['pp_filter'] = graph.get_tensor_by_name('G/pp_filt/conv1d/kernel:0')[:, 0, 0]

  # Summarize
  G_z = graph.get_tensor_by_name('G_z_flat:0')
  summaries = [
      tf.summary.audio('preview', tf.expand_dims(G_z, axis=0), _FS, max_outputs=1)
  ]
  fetches['summaries'] = tf.summary.merge(summaries)
  summary_writer = tf.summary.FileWriter(preview_dir)

  # PP Summarize
  if args.wavegan_genr_pp:
    pp_fp = tf.placeholder(tf.string, [])
    pp_bin = tf.read_file(pp_fp)
    pp_png = tf.image.decode_png(pp_bin)
    pp_summary = tf.summary.image('pp_filt', tf.expand_dims(pp_png, axis=0))

  # Loop, waiting for checkpoints
  ckpt_fp = None
  while True:
    latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
    if latest_ckpt_fp != ckpt_fp:
      print('Preview: {}'.format(latest_ckpt_fp))

      with tf.Session() as sess:
        saver.restore(sess, latest_ckpt_fp)

        _fetches = sess.run(fetches, feeds)

        _step = _fetches['step']

      gen_speech = _fetches['G_z_flat_int16']
      gen_len = int(len(gen_speech) / sample_n)

      for i in range(sample_n):
        label = int(i / 2)
        start = i * gen_len
        end = start + gen_len
        preview_fp = os.path.join(preview_dir, '{}_{}_{}.wav'.format(str(label), str(_step), str(i)))
        wavwrite(preview_fp, _FS, gen_speech[start:end])

      summary_writer.add_summary(_fetches['summaries'], _step)

      if args.wavegan_genr_pp:
        w, h = freqz(_fetches['pp_filter'])

        fig = plt.figure()
        plt.title('Digital filter frequncy response')
        ax1 = fig.add_subplot(111)

        plt.plot(w, 20 * np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [rad/sample]')

        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        plt.plot(w, angles, 'g')
        plt.ylabel('Angle (radians)', color='g')
        plt.grid()
        plt.axis('tight')

        _pp_fp = os.path.join(preview_dir, '{}_ppfilt.png'.format(str(_step).zfill(8)))
        plt.savefig(_pp_fp)

        with tf.Session() as sess:
          _summary = sess.run(pp_summary, {pp_fp: _pp_fp})
          summary_writer.add_summary(_summary, _step)

      print('Done')

      ckpt_fp = latest_ckpt_fp

    time.sleep(1)


if __name__ == '__main__':
  import argparse
  import glob
  import sys

  parser = argparse.ArgumentParser()

  parser.add_argument('mode', type=str, choices=['train', 'preview', 'incept', 'infer'])
  parser.add_argument('train_dir', type=str,
      help='Training directory')

  data_args = parser.add_argument_group('Data')
  data_args.add_argument('--data_dir', type=str,
      help='Data directory')
  data_args.add_argument('--data_first_window', action='store_true', dest='data_first_window',
      help='If set, only use the first window from each audio example')

  wavegan_args = parser.add_argument_group('WaveGAN')
  wavegan_args.add_argument('--wavegan_kernel_len', type=int,
      help='Length of 1D filter kernels')
  wavegan_args.add_argument('--wavegan_dim', type=int,
      help='Dimensionality multiplier for model of G and D')
  wavegan_args.add_argument('--wavegan_batchnorm', action='store_true', dest='wavegan_batchnorm',
      help='Enable batchnorm')
  wavegan_args.add_argument('--wavegan_disc_nupdates', type=int,
      help='Number of discriminator updates per generator update')
  wavegan_args.add_argument('--wavegan_loss', type=str, choices=['dcgan', 'lsgan', 'wgan', 'wgan-gp'],
      help='Which GAN loss to use')
  wavegan_args.add_argument('--wavegan_genr_upsample', type=str, choices=['zeros', 'nn', 'lin', 'cub'],
      help='Generator upsample strategy')
  wavegan_args.add_argument('--wavegan_genr_pp', action='store_true', dest='wavegan_genr_pp',
      help='If set, use post-processing filter')
  wavegan_args.add_argument('--wavegan_genr_pp_len', type=int,
      help='Length of post-processing filter for DCGAN')
  wavegan_args.add_argument('--wavegan_disc_phaseshuffle', type=int,
      help='Radius of phase shuffle operation')

  train_args = parser.add_argument_group('Train')
  train_args.add_argument('--train_batch_size', type=int,
      help='Batch size')
  train_args.add_argument('--train_save_secs', type=int,
      help='How often to save model')
  train_args.add_argument('--train_summary_secs', type=int,
      help='How often to report summaries')

  preview_args = parser.add_argument_group('Preview')
  preview_args.add_argument('--preview_n', type=int,
      help='Number of samples to preview')

  incept_args = parser.add_argument_group('Incept')
  incept_args.add_argument('--incept_metagraph_fp', type=str,
      help='Inference model for inception score')
  incept_args.add_argument('--incept_ckpt_fp', type=str,
      help='Checkpoint for inference model')
  incept_args.add_argument('--incept_n', type=int,
      help='Number of generated examples to test')
  incept_args.add_argument('--incept_k', type=int,
      help='Number of groups to test')

  parser.set_defaults(
    data_dir=None,
    data_first_window=False,
    wavegan_kernel_len=25,
    wavegan_dim=64,
    wavegan_batchnorm=False,
    wavegan_disc_nupdates=5,
    wavegan_loss='wgan-gp',
    wavegan_genr_upsample='zeros',
    wavegan_genr_pp=False,
    wavegan_genr_pp_len=512,
    wavegan_disc_phaseshuffle=2,
    train_batch_size=64,
    train_save_secs=300,
    train_summary_secs=120,
    preview_n=32,
    incept_metagraph_fp='./eval/inception/infer.meta',
    incept_ckpt_fp='./eval/inception/best_acc-103005',
    incept_n=5000,
    incept_k=10)

  args = parser.parse_args()

  # Make train dir
  if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)

  # Save args
  with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

  # Make model kwarg dicts
  setattr(args, 'wavegan_g_kwargs', {
      'kernel_len': args.wavegan_kernel_len,
      'dim': args.wavegan_dim,
      'use_batchnorm': args.wavegan_batchnorm,
      'upsample': args.wavegan_genr_upsample
  })
  setattr(args, 'wavegan_d_kwargs', {
      'kernel_len': args.wavegan_kernel_len,
      'dim': args.wavegan_dim,
      'use_batchnorm': args.wavegan_batchnorm,
      'phaseshuffle_rad': args.wavegan_disc_phaseshuffle
  })

  # Assign appropriate split for mode
  if args.mode == 'train':
    split = 'train'
  else:
    split = None

  # Find fps for split
  if split is not None:
    fps = glob.glob(os.path.join(args.data_dir, split) + '*.tfrecord')

  if args.mode == 'train':
    if len(fps) == 0:
      raise Exception('Did not find any audio files in specified directory')
    print('Found {} audio files in specified directory'.format(len(fps)))
    infer(args)
    train(fps, args)
  elif args.mode == 'preview':
    preview(args)
  elif args.mode == 'infer':
    infer(args)
  else:
    raise NotImplementedError()
