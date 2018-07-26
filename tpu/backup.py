from __future__ import print_function
from google.cloud import storage
import sys

def copy_blob(bucket_name, blob_name, new_bucket_name, new_blob_name):
    """Copies a blob from one bucket to another with a new name."""
    storage_client = storage.Client()
    source_bucket = storage_client.get_bucket(bucket_name)
    source_blob = source_bucket.blob(blob_name)
    destination_bucket = storage_client.get_bucket(new_bucket_name)

    new_blob = source_bucket.copy_blob(
        source_blob, destination_bucket, new_blob_name)

    print('Blob {} in bucket {} copied to blob {} in bucket {}.'.format(
        source_blob.name, source_bucket.name, new_blob.name,
        destination_bucket.name))

def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs()
    return blobs

if __name__ == '__main__':
  import os
  import time

  import tensorflow as tf

  ckpt, nmin = sys.argv[1:3]
  train_dir = 'gs://' + ckpt
  nsec = int(float(nmin) * 60.)

  while tf.train.latest_checkpoint(train_dir) is None:
    print('Waiting for first checkpoint')
    time.sleep(1)

  while True:
    latest_ckpt = tf.train.latest_checkpoint(train_dir)

    # Sleep for two seconds in case file flushing
    time.sleep(2)

    files = list_blobs(ckpt)
    for file in files:
        name = file.name
        _, latest_ckpt = os.path.split(latest_ckpt)
        if latest_ckpt in name:
          print("copied successfully\n", name)
          copy_blob(ckpt, name, ckpt + '-backup', name)
    print('-' * 80)

    # Sleep for an hour
    time.sleep(nsec)
