# PyTorch Tutorial

A hands-on introduction to PyTorch in two 45-minute sessions.

| Session | Notebook | Topics |
|---|---|---|
| 1 | [01_tensors_and_simple_nn.ipynb](notebooks/01_tensors_and_simple_nn.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilievanadezhda/pytorch-tutorial/blob/main/notebooks/01_tensors_and_simple_nn.ipynb) | Tensors, autograd, writing a model with `nn.Module`, training an MLP on MNIST |
| 2 | [02_cnn_satellite_classification.ipynb](notebooks/02_cnn_satellite_classification.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilievanadezhda/pytorch-tutorial/blob/main/notebooks/02_cnn_satellite_classification.ipynb) | Custom datasets, data augmentation, convolutions, training and evaluating a CNN on satellite images |

**Prerequisites:** Python, NumPy, and the basic understanding of a neural network and gradient descent.

## Running the notebooks

**Google Colab (recommended):** click a badge above. Everything is preinstalled, and the datasets download automatically. For notebook 2, use a GPU runtime (*Runtime → Change runtime type → T4 GPU*).

**Locally:**

```bash
pip install torch torchvision matplotlib kagglehub jupyter
jupyter notebook notebooks/
```

Both notebooks also run on a CPU (training takes well under a minute each).

## Data

- **MNIST**, downloaded by `torchvision`.
- **[Satellite Image Classification](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification)** (RSI-CB256 subset, Kaggle): 5,631 images in 4 classes (cloudy, desert, green area, water), downloaded with `kagglehub`. The images belong to their original authors.

Notebook 2 builds on [satellite-image-classification](https://github.com/ilievanadezhda/satellite-image-classification).
