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

"""Jaxline meta-learning experiment for tactile functa."""
import sys
import os
from typing import Generator, List, Mapping, Text, Tuple, Union
from absl import app
from absl import flags
from absl import logging
import functools

import chex # type checking
import haiku as hk
import jax
import jax.numpy as jnp
from jaxline import base_config
from jaxline import experiment
from jaxline import platform
from jaxline import utils
from ml_collections import config_dict
import numpy as np
import dill
import optax
import gc

import data_utils
import function_reps
import helpers
import pytree_conversions

FLAGS = flags.FLAGS

Array = jnp.ndarray
Batch = Mapping[str, Array]
OptState = optax.OptState
PRNGKey = chex.PRNGKey
Scalars = Mapping[Text, Array]

################################################
################################################
# Run the experiment with:
# python -m tactile_meta_learning --config=tactile_meta_learning.py
# Use tensorboard to visualize the results:
# For example: tensorboard --logdir=./tmp/training/bubble_pt_dataset/train
################################################
################################################

def get_config():
    """Returns the default config for the experiment."""
    config = base_config.get_base_config()
    
    config.experiment_kwargs = config_dict.ConfigDict()
    exp = config.experiment_kwargs.config = config_dict.ConfigDict()
    
    # Dataset config
    exp.dataset = config_dict.ConfigDict()
    exp.dataset.name = 'combined_pt_dataset' # 'gelslim_pt_dataset', 'bubble_pt_dataset' and 'combined_pt_dataset' are the only options
    exp.dataset.num_channels = data_utils.DATASET_ATTRIBUTES[
        exp.dataset.name]['num_channels']
    exp.dataset.height = data_utils.DATASET_ATTRIBUTES[
        exp.dataset.name]['height']
    exp.dataset.width = data_utils.DATASET_ATTRIBUTES[
        exp.dataset.name]['width']
    # dataset type should be 'image'
    exp.dataset.type = data_utils.DATASET_ATTRIBUTES[
        exp.dataset.name]['type']
    
    # Optimizer config
    exp.opt_inner = config_dict.ConfigDict()
    exp.opt_inner.lr = 1e-2
    exp.opt_outer = config_dict.ConfigDict()
    exp.opt_outer.lr = 3e-6
    
    # Model config
    exp.model = config_dict.ConfigDict()
    exp.model.type = 'latent_modulated_siren'
    exp.model.w0 = 30.0
    exp.model.width = 512
    exp.model.depth = 10
    exp.model.modulate_scale = False
    exp.model.modulate_shift = True
    exp.model.l2_weight = 1e-5
    exp.model.noise_std = 0.
    
    # latent_modulated_siren specific config
    exp.model.latent_dim = 512
    exp.model.layer_sizes = () # emply tuple -> linear map -> better PSNR
    exp.model.latent_init_scale = 0.
    
    # meta-SGD config
    exp.model.use_meta_sgd = True
    exp.model.meta_sgd_init_range = (0.005, 0.1)
    exp.model.meta_sgd_clip_range = (0., 1.)
    
    # Training config
    per_device_batch_size = 16
    exp.training = config_dict.ConfigDict()
    exp.training.per_device_batch_size = per_device_batch_size
    exp.training.inner_steps = 3
    exp.training.repeat = True
    exp.training.coord_noise = False
    exp.training.subsample = False
    
    # Evaluation config
    exp.evaluation = config_dict.ConfigDict()
    exp.evaluation.batch_size = per_device_batch_size
    exp.evaluation.inner_steps = 3
    exp.evaluation.num_examples = -1 # evaluate on all examples
    exp.evaluation.shuffle = True
    
    # Training loop config: log and checkpoint every minute
    config.training_steps = int(50000)
    config.interval_type = 'steps'
    config.log_train_data_interval = 100
    config.log_tensors_interval = 100
    config.save_checkpoint_interval = 100
    config.train_checkpoint_all_hosts = False
    config.checkpoint_dir = f'./data/meta_learned/{exp.dataset.name}'
    config.eval_specific_checkpoint_dir = f'./data/meta_learned/{exp.dataset.name}'
    
    return config

class Experiment(experiment.AbstractExperiment):
    """Meta-learning experiment for tactile functa."""
    
    CHECKPOINT_ATTRS = {
        '_params': 'params',
        '_opt_state': 'opt_state',
    }
    
    def __init__(self, mode, init_rng, config):
        """Initializes the experiment."""
        
        super().__init__(mode=mode, init_rng=init_rng)
        self.mode = mode
        self.init_rng = init_rng
        
        # This config holds all the experiment specific keys defined in get_config
        self.config = config
        # Use all local devices to train the model, then uncomment the following line
        # self.num_devices = jax.local_device_count()
        
        self.num_devices = len(jax.devices())
        print(f'Number of devices: {self.num_devices}')
        
        # Use without_apply_rng since the forward function is deterministic
        self.forward = hk.without_apply_rng(hk.transform(self._forward_fn))
        
        # Define coordinates of image
        if config.dataset.type == 'image':
            self.coords = function_reps.get_coordinate_grid(config.dataset.height, config.dataset.width)
        else:
            raise f'Unknown dataset type: {config.dataset.type}'
        
        # Inner optimizer is used both for training and validation
        self._opt_inner = optax.sgd(learning_rate=config.opt_inner.lr)
        
        self.bs = self.num_devices * config.training.per_device_batch_size
        self.bs_coords = jnp.stack([self.coords for _ in range(self.bs)]).reshape(
            self.num_devices, config.training.per_device_batch_size, *self.coords.shape
        )
        
        if self.mode == 'train':
            # broadcast the rng key for random number generation to all devices
            init_rng = utils.bcast_local_devices(self.init_rng)
            # Initialize the model and optimizer
            self._params = jax.pmap(self.forward.init)(init_rng, utils.bcast_local_devices(self.coords))
            
            self._opt_outer = optax.adam(learning_rate=config.opt_outer.lr)
            
            # Get parameters (weights without modulations) of the model. Only outer optimizer has a state. Optimizer for inner loop is reset at each iteration.
            weights, _ = function_reps.partition_params(self._params)
            self._opt_state = jax.pmap(self._opt_outer.init)(weights)
            
            # We require an axis name as this will later be used to determine which axis to average the gradients over
            self._update_func = jax.pmap(self._update_func, axis_name='i')
            
            # Set up training dataset
            self._train_input = self._build_train_input(
                self.bs)
        else:
            self._params = None
            self._opt_state = None
            self._eval_batch = jax.jit(self._eval_batch)
            
        
        
        
    def _forward_fn(self, coords: Array) -> Array:
        """Forward pass of the 'latent_modulated_siren' model."""
        assert self.config.model.type == 'latent_modulated_siren'
        model = function_reps.LatentModulatedSiren(
            width=self.config.model.width,
            depth=self.config.model.depth,
            out_channels=self.config.dataset.num_channels,
            w0=self.config.model.w0,
            modulate_scale=self.config.model.modulate_scale,
            modulate_shift=self.config.model.modulate_shift,
            latent_dim=self.config.model.latent_dim,
            layer_sizes=self.config.model.layer_sizes,
            latent_init_scale=self.config.model.latent_init_scale,
            use_meta_sgd=self.config.model.use_meta_sgd,
            meta_sgd_init_range=self.config.model.meta_sgd_init_range,
            meta_sgd_clip_range=self.config.model.meta_sgd_clip_range)
        return model(coords)
    
    # Training
    #  _             _
    # | |_ _ __ __ _(_)_ __
    # | __| '__/ _` | | '_ \
    # | |_| | | (_| | | | | |
    #  \__|_|  \__,_|_|_| |_|
    #
    
    def _build_train_input(self, batch_size: int) -> Generator[Array, None, None]:
        """Builds the training input pipeline.
        All data are images.
        """
        assert self.config.dataset.type == 'image'
        shuffle_buffer_size = 10_000
        return data_utils.load_dataset(
            self.config.dataset.name,
            'train',
            batch_size=batch_size,
            shuffle=True,
            repeat=self.config.training.repeat,
            shuffle_buffer_size=shuffle_buffer_size)
        
    def step(self, global_step, rng, *unused_args, **unused_kwargs):
        """Runs a training step."""
        per_device_batch_size = self.config.training.per_device_batch_size
        
        assert self.config.dataset.type == 'image'
        # Get the next batch of data
        train_batch_dict = next(self._train_input)
        train_batch = train_batch_dict['array']
        train_batch = train_batch.reshape(
            self.num_devices, per_device_batch_size, 
            *train_batch.shape[1:]
        )
        
        self._params, self._opt_state, scalars = (
            self._update_func(self._params, self._opt_state, train_batch, self.bs_coords, rng)
        )
        
        scalars = utils.get_first(scalars)
        
        # Print losses
        global_step = utils.get_first(global_step)
        logging.info(f'global_step: {global_step}, train PSNR: {scalars["train_psnr"]:.2f}dB, mse loss: {scalars["mse_loss"]}')
        
        # Save final checkpoint
        if global_step == FLAGS.config.get('training_steps', 1) - 1:
            f_np = lambda x: np.array(jax.device_get(utils.get_first(x)))
            np_params = jax.tree_map(f_np, self._params)
            
            ckpt_data = {
                'params': np_params,
                'config': self.config
            }
            
            path_npy = os.path.join(FLAGS.config.checkpoint_dir, 'checkpoint.npz')
            with open(path_npy, 'wb') as f:
                dill.dump(ckpt_data, f)
            logging.info('Saved final checkpoint and config to %s', path_npy)
        
        return scalars
        
        
    def _update_func(self, params: hk.Params, opt_outer_state: OptState, train_batch: Array, coords: Array, rng: PRNGKey) -> Tuple[hk.Params, OptState, Scalars]:
        """Updates meta-learned init of params.
        Only updates weights, not modulations.
        Returns:
            Updated params, optimization state and scalars (losses).
        """
        # Compute gradients and loss
        weights, modulations = function_reps.partition_params(params)
        # value_and_grad returns the loss and the gradient of the loss with respect to the weights (the first argument)
        _, model_grad = jax.value_and_grad(
            self._loss_func)(
                weights, modulations, train_batch, coords, rng)
        
        # Deal with parallelism
        model_grad = jax.lax.pmean(model_grad, axis_name='i')
        # Update the weights
        updates, opt_outer_state = self._opt_outer.update(model_grad, opt_outer_state)
        # Apply the updates to the weights
        weights = optax.apply_updates(weights, updates)
        # Merge the weights and original modulations
        params = function_reps.merge_params(weights, modulations)
        
        # Track training loss
        fitted_params, loss = self._fit_params(params, train_batch, coords, rng)
        mse_loss = jnp.mean(loss) # not considering l2 norm
        scalars = {'train_psnr': helpers.psnr_fn(mse_loss),
                   'mse_loss': mse_loss}
        # Average scalars across devices
        scalars = jax.lax.pmean(scalars, axis_name='i')
        return params, opt_outer_state, scalars
        
    def _loss_func(self, weights: hk.Params, modulations: hk.Params, train_batch: Array, coords: Array, rng: PRNGKey) -> Tuple[Array, Array]:
        """loss function (which only meta-learns weights, not modulations).
        
        Taking the gradient with respect to this loss function will backpropagate through the entire inner loop.
        
        Returns:
            loss.
        """
        params = function_reps.merge_params(weights, modulations)
        _, loss = self._fit_params(params, train_batch, coords, rng)
        return jnp.mean(loss)
    
    def _fit_params(self, params: hk.Params, train_batch: Array, coords: Array, rng: PRNGKey) -> Tuple[hk.Params, Array]:
        """Fits params of a model by running inner loop.
        Returns:
            fitted_params (bs, ...). Not used in the outer loop.
            loss (bs)
        """
        rng = jax.random.split(rng, num=train_batch.shape[0]) # [bs, 2]
        fitted_params, loss, _ = jax.vmap(
            self._inner_loop, in_axes=[None, 0, 0, 0])(params, train_batch, coords, rng)
        return fitted_params, loss
    
    def _inner_loop(self, params: hk.Params, targets: Array, coords: Array, rng: PRNGKey) -> Tuple[hk.Params, Array, Array]:
        """Inner loop of the model. MAML
        
        This function takes `self.inner_steps` SGD steps in the inner loop to update modulations while keeping weights fixed. This function is applied to a single image.
        
        Returns:
            Updated_params, loss, PSNR
        """
        return helpers.inner_loop(params, self.forward, self._opt_inner,
                              self.config.training.inner_steps, coords,
                              targets,
                              render_config=None,
                              l2_weight=self.config.model.l2_weight,
                              noise_std=self.config.model.noise_std,
                              rng=rng,
                              coord_noise=self.config.training.coord_noise)
    
    # Evaluation
    #                  _
    #   _____   ____ _| |
    #  / _ \ \ / / _` | |
    # |  __/\ V / (_| | |
    #  \___| \_/ \__,_|_|
    #

    def evaluate(self, global_step, rng, **unused_kwargs):
        """Runs evaluation."""
        assert self.config.dataset.type == 'image'
        global_step = utils.get_first(global_step)
        log_dict = jax.device_get(self._eval_epoch(rng))
        scalars = log_dict['scalars']

        print(f'Step: {global_step}, val PSNR: {scalars["val_psnr"]:.2f}dB, mse loss: {scalars["mse_loss"]}')
        logging.info('[Step %d] Eval scalars: %s', global_step, scalars['val_psnr'])

        return scalars


    def _eval_epoch(self, rng: Array):
        """Evaluates an epoch."""
        num_samples = 0.
        summed_scalars = None
        rng = rng[0]
        params = utils.get_first(self._params)

        for i, val_batch_dict in enumerate(self._build_eval_input()):
            rng, _ = jax.random.split(rng) # use new rng for each batch
            val_batch = val_batch_dict['array']
            num_samples += val_batch.shape[0]
            log_dict = self._eval_batch(params, val_batch_dict, rng)
            scalars = log_dict['scalars']
            scalars = jax.tree_map(lambda x: jnp.sum(x, axis=0), scalars)
            if summed_scalars is None:
                summed_scalars = scalars
            else:
                summed_scalars = jax.tree_map(jnp.add, summed_scalars, scalars)
            print(f'============= {i} eval iterations done =============')
            logging.info('%d eval iterations done', i)

        mean_scalars = jax.tree_map(lambda x: x / num_samples, summed_scalars)
        return {'scalars': mean_scalars}


    def _build_eval_input(self) -> Generator[Array, None, None]:
        assert self.config.dataset.type == 'image'
        shuffle_buffer_size = 10_000
        return data_utils.load_dataset(
            self.config.dataset.name,
            'test',
            batch_size=self.config.evaluation.batch_size,
            shuffle=self.config.evaluation.shuffle,
            num_examples=self.config.evaluation.num_examples,
            shuffle_buffer_size=shuffle_buffer_size)
    
    def _eval_batch(self, params: hk.Params, val_batch_dict: Mapping[str, Array], rng: Array) -> Scalars:
        """Evaluates the model on a batch of data."""
        val_batch = val_batch_dict['array']
        out = jax.vmap(
            self._eval_inner_loop,
            in_axes=[None, 0, None]
        )(params, val_batch, self.coords)

        # Unpack the results
        new_params, loss, val_psnrs, val_losses = out
        scalars = {}
        _, mods = function_reps.partition_params(new_params)
        mods_array = jax.vmap(lambda x: pytree_conversions.pytree_to_array(x)[0])(mods)
        l2_norm = jnp.sqrt(jnp.sum(mods_array**2, axis=-1))
        scalars['mod_l2_norm'] = l2_norm

        for i in range(len(val_psnrs)):
            scalars[f'val_psnr_{str(i).zfill(2)}'] = val_psnrs[i]
            scalars[f'loss_{str(i).zfill(2)}'] = val_losses[i]
        
        scalars['val_psnr'] = val_psnrs[self.config.training.inner_steps]
        scalars['loss'] = loss
        log_dict = {'scalars': scalars}

    def _eval_inner_loop(self, params: hk.Params, image: Array, coords: Array) -> Union[Tuple[hk.Params, Array, List[Array]], Tuple[
      hk.Params, Array, List[Array], List[Array]]]:
        """Inner loop of the model for evaluation."""
        return helpers.inner_loop(params,
            self.forward,
            self._opt_inner,
            self.config.evaluation.inner_steps,
            coords,
            image,
            render_config=None,
            l2_weight=self.config.model.l2_weight,
            return_all_losses=True,
            return_all_psnrs=True)

if __name__ == '__main__':
    jax.clear_backends()
    gc.collect()
    # Parse command line arguments
    flags.mark_flag_as_required('config')
    app.run(functools.partial(platform.main, Experiment))
    