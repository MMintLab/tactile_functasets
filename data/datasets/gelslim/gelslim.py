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

import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import numpy as np
import pathlib

"""
Dataset name: gelslim_pt_dataset
Run "tfds build --register_checksums" to build the dataset.
"""

_DESCRIPTION = """
Convert .pt files from gelslim sensors into a tensorflow dataset (tfds).
"""

_CITATION = """
@misc{rodriguez2024touch2touchcrossmodaltactilegeneration,
      title={Touch2Touch: Cross-Modal Tactile Generation for Object Manipulation}, 
      author={Samanta Rodriguez and Yiming Dou and Miquel Oller and Andrew Owens and Nima Fazeli},
      year={2024},
      eprint={2409.08269},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.08269}, 
}
"""

class GelslimPtDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for custom dataset created from .pt files."""
    
    VERSION = tfds.core.Version('1.1.1')
    RELEASE_NOTES = {
        '1.1.1': 'Entire gelslim.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(320, 427, 1), dtype=tf.float32),
                'pose': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
            }),
            supervised_keys=None,  # Set to `None` to disable
            homepage='https://sikai-li.gitbook.io/sikai_li',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Path to your .pt files
        data_dir = pathlib.Path("/data/functa/gelslim")
        return {
            'train': self._generate_examples(data_dir),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        for pt_file in path.glob('*.pt'):
            # Load the tensor from the .pt file
            # image_tensor = torch.load(pt_file)
            data = torch.load(pt_file, weights_only=True)
            
            img1, _ = data["gelslim_ref"][0], data["gelslim_ref"][1]
            img3, _ = data["gelslim"][0], data["gelslim"][1]
            theta, x, y = data["theta"], data["x"], data["y"]
            pose_vec = torch.tensor([theta, x, y]).numpy().astype(np.float32)
            
            left_img_diff = img1 - img3

            left_img_diff = rgb2gray(left_img_diff)
            
            normalized_left_img_diff = normalize_image(left_img_diff)
            
            # Convert the tensor to a NumPy array and then to float
            image_np = normalized_left_img_diff[:, :, np.newaxis].numpy().astype(np.float32)
            
            # Yields (key, example)
            yield pt_file.name, {'image': image_np, 'pose': pose_vec}
            
def normalize_image(img):
    min_val = torch.min(img)
    max_val = torch.max(img)
    normalized_img = (img - min_val) / (max_val - min_val)
    return normalized_img

def rgb2gray(rgb):
    return 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]