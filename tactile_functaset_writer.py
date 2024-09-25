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

"""Create modulation dataset for bubble/gelslim."""

import os

from absl import app
from absl import flags
import dill
import pickle
import haiku as hk
import numpy as np
import optax
from ml_collections import config_dict

import data_utils
import function_reps
import helpers
import pytree_conversions

flags.DEFINE_string('type', 'combined',
                    'data type: bubble, combined or gelslim')

FLAGS = flags.FLAGS


# Define function that creates a dict of modulations & psnrs for each dataset
def create_modulation_dataset(model, params, ds, num_steps, coords, lr,
                              l2_weight, noise_std):
  """Creates a dataset of modulations and corresponding psnr values.

  Args:
    model: Haiku transformed model that outputs rgb given 2d pixel coord inputs.
    params: Parameters of ModulatedSiren or LatentModulatedSiren model.
    ds: Dataset iterator that gives a single image at each iteration.
    num_steps: Number of SGD steps to use for fitting each modulation.
    coords: 2D pixel coordinates of shape (H, W, 2).
    lr: Learning rate of SGD optimizer.
    l2_weight: Weight for L2 regularisation of modulations.
    noise_std: standard deviation of Gaussian noise applied to modulations.

  Returns:
    mod_data: Array of modulations shape (data_size, mod_dim).
    psnr_vals: Array of psnrs shape (data_size,).
    psnr_mean: psnr corresponding to the mean rec loss across the dataset.
  """
  # Define sgd optimizer that carries out 3 gradient steps wrt modulations
  opt_inner = optax.sgd(lr)
  mod_list = []
  psnr_list = []
  pose_list = []
  rec_loss_list = []
  for i, datum in enumerate(ds):
    fitted_params, _, psnr = helpers.inner_loop(
        params=params,
        model=model,
        opt_inner=opt_inner,
        inner_steps=num_steps,
        coords=coords,
        targets=datum['array'],
        return_all_psnrs=False,
        return_all_losses=False,
        l2_weight=l2_weight,
        noise_std=noise_std)
    rec_loss = helpers.inverse_psnr_fn(psnr)
    _, modulations = function_reps.partition_params(fitted_params)
    modulations, _, _ = pytree_conversions.pytree_to_array(modulations)
    mod_list.append(modulations)
    psnr_list.append(psnr)
    pose_list.append(datum['pose'])
    rec_loss_list.append(rec_loss)
    print(f'data point {(i+1):5d} has psnr {psnr:2.2f} dB')
  mod_data = np.stack(mod_list)  # [num_data, mod_dim]
  psnr_vals = np.array(psnr_list)  # [num_data]
  pose_data = np.stack(pose_list)  # [num_data, 3]
  rec_losses = np.array(rec_loss_list)  # [num_data]
  mean_rec_loss = np.mean(rec_losses)
  psnr_mean = helpers.psnr_fn(mean_rec_loss)
  return mod_data, psnr_vals, psnr_mean, pose_data


def main(_):
  # Load params of LatentModulatedSiren model
  data_type = FLAGS.type
  # Relative path to trunk network checkpoint (Or change the path to your own checkpoint)
  path = f'./tmp/training/{data_type}_pt_dataset/checkpoint.npz'
  ## Check that checkpoint file exists
  assert os.path.exists(path), 'Pretrained weights file does not exist.'
  with open(path, 'rb') as f:
    ckpt = dill.load(f)
  print("Successfully loaded checkpoint")
  params = ckpt['params']
  config = ckpt['config']
  mod_dim = config['model']['latent_dim']
  assert config['model']['type'] == 'latent_modulated_siren'
  print(f'Loaded params for model with {mod_dim} latent dimensions.')
  ## Create haiku transformed model that runs the forward pass.
  ## Only keep configs needed for model construction from model config `None` below ensures no error is given when already removed
  model_config = dict(config['model']).copy()
  model_config.pop('type', None)
  model_config.pop('l2_weight', None)
  model_config.pop('noise_std', None)

  def model_net(coords):
    hk_model = function_reps.LatentModulatedSiren(
        out_channels=config['dataset']['num_channels'], **model_config)
    return hk_model(coords)
  model = hk.without_apply_rng(hk.transform(model_net))

  # Check that user specified directory exists if specified
  data_dir = os.getcwd() + f'/data/functasets/{data_type}/'
  assert os.path.isdir(
      data_dir
  ), f'User specified directory {data_dir} does not exist.'

  # Setup dataset
  train_ds = data_utils.load_dataset(config['dataset']['name'], subset='train')
  test_ds = data_utils.load_dataset(config['dataset']['name'], subset='test')

  # Iterate across training set to produce train modulations
  train_mod_data, train_psnr_vals, train_psnr_mean, train_pose = create_modulation_dataset(
      model=model,
      params=params,
      ds=train_ds,
      num_steps=4,
      coords=function_reps.get_coordinate_grid(config['dataset']['height'], config['dataset']['width']),
      lr=config['opt_inner']['lr'],
      l2_weight=config['model']['l2_weight'],
      noise_std=config['model']['noise_std'],
  )
  print(f'Training set psnr: {train_psnr_mean}')
  print(f'Training set modulations shape: {train_mod_data.shape}')
  print(f'Training set poses shape: {train_pose.shape}')
  print(f'Training set mean mse loss: {helpers.inverse_psnr_fn(train_psnr_mean)}')

  # Repeat with test set
  test_mod_data, test_psnr_vals, test_psnr_mean, test_pose = create_modulation_dataset(
      model=model,
      params=params,
      ds=test_ds,
      num_steps=4,
      coords=function_reps.get_coordinate_grid(config['dataset']['height'], config['dataset']['width']),
      lr=config['opt_inner']['lr'],
      l2_weight=config['model']['l2_weight'],
      noise_std=config['model']['noise_std'],
  )
  print(f'Test set psnr: {test_psnr_mean}')
  print(f'Test set modulations shape: {test_mod_data.shape}')
  print(f'Test set poses shape: {test_pose.shape}')
  print(f'Test set mean mse loss: {helpers.inverse_psnr_fn(test_psnr_mean)}')

  # Save modulations to user specified directory
  train_dict = dict(modulation=train_mod_data, psnr=train_psnr_vals, pose=train_pose)
  test_dict = dict(modulation=test_mod_data, psnr=test_psnr_vals, pose=test_pose)
  modulation_data = dict(train=train_dict, test=test_dict)
  path = os.path.join(data_dir, f'{data_type}_modulations_{mod_dim}_latents.npz')
  with open(path, 'wb') as f:
    dill.dump(modulation_data, f)


if __name__ == '__main__':
  app.run(main)
