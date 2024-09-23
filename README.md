# Tactile Functaset

[![Python Version][python-image]][python-url]
[![Package License][package-license-image]][package-license-url]

This repository contains main code for the paper "Tactile Functasets: Neural Implicit Representations of Tactile Datasets" by Sikai Li, Samanta Rodriguez, Yiming Dou, Andrew Owens, Nima Fazeli.

## Overview
🦾 Tactile functaset (TactFunc) reconstructs the high-dimensional raw tactile dataset by training neural implicit functions. It produces compact representations that capture the underlying structure of the tactile sensory inputs. We demonstrate the efficacy of this representation on the downstream task of in-hand object pose estimation, achieving improved performance over image-based methods while simplifying downstream models.

This codebase contains implementations of:

1. Meta-learning for Bubble and Gelslim tactile datasets.
2. Conversion from raw tactile datasets to functasets.
3. Inference over tactile functasets.
4. Downstream models for in-hand object pose estimation.
5. Baselines: ResNet-18, Variational Autoencoder and T3 model.

Bubble and Gelslim datasets are from ["Touch2Touch: Cross-Modal Tactile Generation for Object Manipulation"](https://www.arxiv.org/abs/2409.08269) and can be found [here](https://drive.google.com/drive/folders/15vWo5AWw9xVKE1wHbLhzm40ClPyRBYk5?usp=sharing ).

## Contents
- [Setup](#setup)
- [Data](#data)
- [Experiments](#experiments)
- [Demos](#demos)

### ⛏️ In progress...

## Setup

## Data

## Experiments

## Demos

## Citation
```
```

## Acknowledgement
This work is supported by NSF GRFP \#2241144, NSF CAREER Awards \#2339071 and \#2337870, and NSF NRI \#2220876.

## License
The source code is licensed under Apache 2.0.

## Contact
For more information please contact skevinci@umich.edu.

[python-image]: https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg
[python-url]: https://docs.python.org/3.10/
[package-license-image]: https://img.shields.io/badge/License-Apache_2.0-blue.svg
[package-license-url]: https://github.com/camel-ai/camel/blob/master/licenses/LICENSE
