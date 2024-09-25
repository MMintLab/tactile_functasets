# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This file has been modified by Sikai Li, MMint Lab in 2024 from the original version. The original version can be found at:
#
#   https://github.com/google-deepmind/functa
#
# Modifications Copyright 2024 Sikai Li
#
# This modified version is licensed under the Apache License 2.0.
# ==============================================================================

"""Utils for loading and processing datasets."""

from typing import Mapping, Optional
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


Array = jnp.ndarray
Batch = Mapping[str, np.ndarray]
COMMON_HEIGHT, COMMON_WIDTH = 96, 96
DATASET_ATTRIBUTES = {
    'bubble_pt_dataset': {
        'num_channels': 1,
        'height': 70,
        'width': 88,
        'type': 'image',
    },
    'gelslim_pt_dataset': {
        'num_channels': 1,
        'height': 80,
        'width': 107,
        'type': 'image',
    },
    'combined_pt_dataset': {
        'num_channels': 1,
        'height': COMMON_HEIGHT,
        'width': COMMON_WIDTH,
        'type': 'image',
    },
}


def load_dataset(dataset_name: str,
                 subset: str,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 repeat: bool = False,
                 num_examples: Optional[int] = None,
                 shuffle_buffer_size: int = 10000):
  """Tensorflow dataset loaders.

  Args:
    dataset_name (string): One of elements of DATASET_NAMES.
    subset (string): One of 'train', 'test'.
    batch_size (int):
    shuffle (bool): Whether to shuffle dataset.
    repeat (bool): Whether to repeat dataset.
    num_examples (int): If not -1, returns only the first num_examples of the
      dataset.
    shuffle_buffer_size (int): Buffer size to use for shuffling dataset.

  Returns:
    Tensorflow dataset iterator.
  """

  # Load dataset
  if dataset_name.startswith('bubble'):
    if subset == 'train':
      subset = 'train[:90%]'
    elif subset == 'test':
      subset = 'train[90%:]'
    ds = tfds.load(dataset_name, split=subset)
    ds = ds.map(process_bubble,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  elif dataset_name.startswith('gelslim'):
    if subset == 'train':
      subset = 'train[:90%]'
    elif subset == 'test':
      subset = 'train[90%:]'
    ds = tfds.load(dataset_name, split=subset)
    ds = ds.map(process_gelslim,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  elif dataset_name.startswith('combined'):
    bubble_ds = load_single_dataset('bubble_pt_dataset', subset)
    gelslim_ds = load_single_dataset('gelslim_pt_dataset', subset)
    
    zipped_ds = tf.data.Dataset.zip((bubble_ds, gelslim_ds))
    
    ds = zipped_ds.flat_map(
      lambda x, y: tf.data.Dataset.from_tensors(x).concatenate(tf.data.Dataset.from_tensors(y))
    )

  # Optionally subsample dataset
  if num_examples is not None:
    ds = ds.take(num_examples)

  # Optionally shuffle dataset
  if shuffle:
    ds = ds.shuffle(shuffle_buffer_size)

  # Optionally repeat dataset if repeat
  if repeat:
    ds = ds.repeat()

  if batch_size is not None:
    ds = ds.batch(batch_size)

  # Convert from tf.Tensor to numpy arrays for use with Jax
  return iter(tfds.as_numpy(ds))

def load_single_dataset(dataset_name: str, subset: str):
  if subset == 'train':
    subset = 'train[:90%]'
  elif subset == 'test':
    subset = 'train[90%:]'
    
  ds = tfds.load(dataset_name, split=subset)
  if dataset_name.startswith('bubble') or dataset_name.startswith('gelslim'):
    return ds.map(process_common,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  else:
    raise ValueError(f'Unknown dataset name: {dataset_name}')
    

def process_bubble(batch: Batch):
  image = tf.cast(batch['image'], tf.float32)
  # Resize resolution.
  image = tf.image.resize(image, [70, 88])

  pose = tf.cast(batch['pose'], tf.float32)
  return {'array': image, 'pose': pose}

def process_gelslim(batch: Batch):
  image = tf.cast(batch['image'], tf.float32)
  # Resize resolution.
  image = tf.image.resize(image, [80, 107])  # [64, 64, 3]

  pose = tf.cast(batch['pose'], tf.float32)
  return {'array': image, 'pose': pose}

def process_common(batch: Batch):
  image = tf.cast(batch['image'], tf.float32)
  # Resize resolution.
  image = tf.image.resize(image, [COMMON_HEIGHT, COMMON_WIDTH])

  pose = tf.cast(batch['pose'], tf.float32)
  return {'array': image, 'pose': pose}

