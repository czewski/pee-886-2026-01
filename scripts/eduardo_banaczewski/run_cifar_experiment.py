import argparse
from pathlib import Path

from qml.eduardo_banaczewski import (
    CifarExperimentConfig,
    CifarTrainer,
    build_cifar10_loaders,
    create_model,
    create_output_dir,
    set_global_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI parameters for CIFAR experiment execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="cnn_benchmark",
        choices=["cnn_benchmark", "qml_baseline", "qml_strong", "qml_data_reupload"],
    )
    parser.add_argument("--experiment-name", type=str, default="cifar_experiment")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default="./data/eduardo_banaczewski")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-q-layers", type=int, default=2)
    parser.add_argument("--non-deterministic", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build and execute the configured CIFAR experiment."""
    args = parse_args()
    config = CifarExperimentConfig(
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        n_folds=args.n_folds,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_qubits=args.n_qubits,
        n_q_layers=args.n_q_layers,
        deterministic=not args.non_deterministic,
    )

    set_global_seed(config.seed, deterministic=config.deterministic)
    output_dir = Path(create_output_dir(config))
    loaders = build_cifar10_loaders(config)
    trainer = CifarTrainer(
        model_builder=lambda: create_model(config, num_classes=len(loaders.class_names)),
        config=config,
        output_dir=output_dir,
        class_names=loaders.class_names,
    )
    result = trainer.run(
        train_dataset_aug=loaders.train_dataset_aug,
        train_dataset_eval=loaders.train_dataset_eval,
        train_targets=loaders.train_targets,
        test_loader=loaders.test_loader,
    )
    print(f"Done. Mean test acc: {result.summary['test_acc_mean']:.4f}")
    print(f"Artifacts saved to: {result.experiment_dir}")


if __name__ == "__main__":
    main()
