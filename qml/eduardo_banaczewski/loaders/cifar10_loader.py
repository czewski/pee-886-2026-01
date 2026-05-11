from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from qml.eduardo_banaczewski.experiment import CifarExperimentConfig

__all__ = ["CifarLoaders", "build_cifar10_loaders"]


@dataclass(frozen=True)
class CifarLoaders:
    train_dataset_aug: datasets.CIFAR10
    train_dataset_eval: datasets.CIFAR10
    test_loader: DataLoader
    train_targets: torch.Tensor
    class_names: Tuple[str, ...]


def _seed_worker(worker_id: int) -> None:
    """Seed dataloader worker process for reproducible loading."""
    worker_seed = torch.initial_seed() % (2**32)
    torch.manual_seed(worker_seed + worker_id)


def build_cifar10_loaders(config: CifarExperimentConfig) -> CifarLoaders:
    """Build CIFAR-10 datasets and test loader for k-fold experiments."""
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset_full = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    val_dataset_full = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        download=False,
        transform=eval_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

    loader_kwargs: Dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
        "worker_init_fn": _seed_worker if config.num_workers > 0 else None,
    }
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return CifarLoaders(
        train_dataset_aug=train_dataset_full,
        train_dataset_eval=val_dataset_full,
        test_loader=test_loader,
        train_targets=torch.as_tensor(train_dataset_full.targets, dtype=torch.long),
        class_names=tuple(test_dataset.classes),
    )
