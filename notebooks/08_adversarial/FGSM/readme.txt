FGSM Attack on ResNet-18 with CIFAR-10
=======================================

This program trains a CIFAR-10 ResNet-18 and evaluates it against the Fast
Gradient Sign Method (FGSM) adversarial attack.

Requirements
------------

Install the required packages:

    pip install torch torchvision numpy matplotlib

Usage
-----

Train the model and test several attack strengths:

    python3 fgsm_cifar10.py --epochs 20 --epsilons 0 0.01 0.03 0.05

The CIFAR-10 dataset is downloaded automatically into the "data" directory.
After training, model weights are saved as "cifar10_resnet18.pt".

To reuse an existing checkpoint without training again:

    python3 fgsm_cifar10.py --skip-training --checkpoint cifar10_resnet18.pt

Use the same command if training completed but a later evaluation or
visualization step failed. It reruns the FGSM evaluation and creates
"fgsm_examples.png" without repeating training.

The program prints test accuracy for every epsilon value. A larger epsilon
produces a stronger attack. Successful adversarial examples are saved to
"fgsm_examples.png".

Useful options
--------------

    --batch-size 128       Training and evaluation batch size
    --learning-rate 0.1    Initial SGD learning rate
    --data-dir PATH        CIFAR-10 download location
    --checkpoint PATH      Model checkpoint location
    --output PATH          Adversarial example image location
    --workers 2            Number of data-loading workers

Display every available option with:

    python3 fgsm_cifar10.py --help

Jupyter notebook
----------------

An interactive version is available as "fgsm_cifar10.ipynb". Open it with
Jupyter and run the cells in order:

    jupyter notebook fgsm_cifar10.ipynb

The notebook automatically loads "cifar10_resnet18.pt" when it exists. If the
checkpoint is missing, it trains the model and saves a new checkpoint.
