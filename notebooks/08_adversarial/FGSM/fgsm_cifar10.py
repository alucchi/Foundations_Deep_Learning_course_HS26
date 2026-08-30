#!/usr/bin/env python3
"""Train a ResNet-18 on CIFAR-10 and evaluate an FGSM adversarial attack.

Example:
    python fgsm_cifar10.py --epochs 20 --epsilons 0 0.01 0.03 0.05
    python fgsm_cifar10.py --checkpoint cifar10_resnet18.pt --skip-training
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class NormalizedModel(nn.Module):
    """Keep inputs in [0, 1] while normalizing inside the differentiable model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR10_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


def make_model() -> nn.Module:
    """Create a ResNet-18 adapted to 32x32 CIFAR images."""
    model = models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return NormalizedModel(model)


def make_loaders(data_dir: Path, batch_size: int, workers: int) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.ToTensor()
    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    loader_args = dict(batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    return train_loader, test_loader


def train(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(1, epochs + 1):
        correct = total = 0
        loss_sum = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * labels.size(0)
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        print(
            f"Epoch {epoch:3d}/{epochs}: loss={loss_sum / total:.4f}, "
            f"train_accuracy={100.0 * correct / total:.2f}%"
        )


def fgsm_attack(images: torch.Tensor, epsilon: float, gradients: torch.Tensor) -> torch.Tensor:
    """Apply x_adv = clip(x + epsilon * sign(dL/dx), 0, 1)."""
    return torch.clamp(images + epsilon * gradients.sign(), min=0.0, max=1.0).detach()


def evaluate_fgsm(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epsilon: float,
    max_examples: int = 5,
) -> tuple[float, list[tuple[torch.Tensor, torch.Tensor, int, int]]]:
    """Return robust accuracy and a few successful adversarial examples."""
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    examples: list[tuple[torch.Tensor, torch.Tensor, int, int]] = []
    model.eval()

    # Gradients are required with respect to inputs, even though parameters are frozen.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad_(True)
        logits = model(images)
        loss = criterion(logits, labels)
        gradients = torch.autograd.grad(loss, images)[0]
        adversarial = fgsm_attack(images, epsilon, gradients)

        with torch.no_grad():
            predictions = model(adversarial).argmax(1)
        correct += predictions.eq(labels).sum().item()
        total += labels.size(0)

        if epsilon > 0 and len(examples) < max_examples:
            clean_predictions = logits.detach().argmax(1)
            successful = clean_predictions.eq(labels) & predictions.ne(labels)
            for index in successful.nonzero(as_tuple=False).flatten():
                examples.append(
                    (
                        images[index].detach().cpu(),
                        adversarial[index].cpu(),
                        labels[index].item(),
                        predictions[index].item(),
                    )
                )
                if len(examples) == max_examples:
                    break

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return correct / total, examples


def save_examples(
    examples: list[tuple[torch.Tensor, torch.Tensor, int, int]],
    epsilon: float,
    output: Path,
) -> None:
    if not examples:
        print("No successful attacks found to visualize.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping visualization.")
        return

    figure, axes = plt.subplots(2, len(examples), figsize=(2.5 * len(examples), 5))
    if len(examples) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for column, (clean, adversarial, true_label, predicted_label) in enumerate(examples):
        axes[0, column].imshow(clean.permute(1, 2, 0).numpy())
        axes[0, column].set_title(f"Clean: {CIFAR10_CLASSES[true_label]}")
        axes[1, column].imshow(adversarial.permute(1, 2, 0).numpy())
        axes[1, column].set_title(f"FGSM: {CIFAR10_CLASSES[predicted_label]}")
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    figure.suptitle(f"Successful FGSM attacks (epsilon={epsilon:g})")
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved examples to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("cifar10_resnet18.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.05])
    parser.add_argument("--skip-training", action="store_true", help="Load --checkpoint instead of training")
    parser.add_argument("--output", type=Path, default=Path("fgsm_examples.png"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(epsilon < 0 or epsilon > 1 for epsilon in args.epsilons):
        raise ValueError("Every epsilon must be in [0, 1]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, test_loader = make_loaders(args.data_dir, args.batch_size, args.workers)
    model = make_model().to(device)

    if args.skip_training:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        train(model, train_loader, device, args.epochs, args.learning_rate)
        torch.save(model.state_dict(), args.checkpoint)
        print(f"Saved checkpoint to {args.checkpoint}")

    all_examples = []
    for epsilon in args.epsilons:
        accuracy, examples = evaluate_fgsm(model, test_loader, device, epsilon)
        print(f"epsilon={epsilon:.4f}  test_accuracy={100.0 * accuracy:.2f}%")
        if examples:
            all_examples = examples
            visualization_epsilon = epsilon
    if all_examples:
        save_examples(all_examples, visualization_epsilon, args.output)


if __name__ == "__main__":
    main()
