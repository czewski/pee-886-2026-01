import json
import os
import time
import copy
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class Trainer:
    def __init__(
        self,
        model,
        n_folds,
        epochs_per_fold,
        patience,
        batch_size,
        learning_rate,
        seed,
        num_workers,
        train_dataset,
        criterion,
        device,
        results_path,
    ):
        self.n_folds = n_folds
        self.epochs_per_fold = epochs_per_fold
        self.patience = patience
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.num_workers = num_workers
        self.train_dataset = train_dataset
        self.criterion = criterion
        self.device = device
        self.pin_memory = self.device.type == "cuda"
        self.results_path = results_path
        self.all_fold_metrics = {}
        self.model = model
        self._initial_model_state = copy.deepcopy(self.model.state_dict())
        self.postfix_update_interval = 20

    def fit(self):
        labels_cv = self.train_dataset.targets.numpy()
        indices = np.arange(len(labels_cv))
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels_cv)):
            self.model.load_state_dict(self._initial_model_state)
            train_sub, val_sub = (
                Subset(self.train_dataset, train_idx),
                Subset(self.train_dataset, val_idx),
            )
            train_generator = torch.Generator().manual_seed(self.seed + fold)
            cv_train_loader = DataLoader(
                train_sub,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
                generator=train_generator,
            )
            cv_val_loader = DataLoader(
                val_sub,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
            )

            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            early_stopper = EarlyStopping(patience=self.patience, min_delta=0.001)

            fold_history = {"train_loss": [], "val_loss": [], "val_acc": []}
            best_val_loss = float("inf")
            best_val_acc = 0.0

            epoch_pbar = tqdm(
                total=self.epochs_per_fold,
                desc=f"Fold {fold + 1}/{self.n_folds} Epochs",
                position=0,
                leave=False,
                dynamic_ncols=True,
                unit="epoch",
            )
            fold_train_start = time.time()

            for epoch in range(self.epochs_per_fold):
                self.model.train()
                tr_loss = 0.0
                total_batches = len(cv_train_loader) + len(cv_val_loader)
                batch_pbar = tqdm(
                    total=total_batches,
                    desc=f"Fold {fold + 1} Epoch {epoch + 1}",
                    position=1,
                    leave=False,
                    dynamic_ncols=True,
                    unit="batch",
                )
                for batch_idx, (imgs, lbls) in enumerate(cv_train_loader, start=1):
                    imgs = imgs.to(self.device, non_blocking=True)
                    lbls = lbls.to(self.device, non_blocking=True)
                    optimizer.zero_grad()
                    loss = self.criterion(self.model(imgs), lbls)
                    loss.backward()
                    optimizer.step()
                    tr_loss += loss.item()
                    batch_pbar.update(1)
                    if (
                        batch_idx % self.postfix_update_interval == 0
                        or batch_idx == len(cv_train_loader)
                    ):
                        batch_pbar.set_postfix(
                            {
                                "phase": "train",
                                "TrL": f"{(tr_loss / batch_idx):.3f}",
                            }
                        )

                self.model.eval()
                v_loss, correct = 0.0, 0
                val_seen = 0
                with torch.no_grad():
                    for batch_idx, (imgs, lbls) in enumerate(cv_val_loader, start=1):
                        imgs = imgs.to(self.device, non_blocking=True)
                        lbls = lbls.to(self.device, non_blocking=True)
                        outputs = self.model(imgs)
                        v_loss += self.criterion(outputs, lbls).item()
                        correct += (outputs.argmax(1) == lbls).sum().item()
                        val_seen += lbls.size(0)
                        batch_pbar.update(1)
                        if (
                            batch_idx % self.postfix_update_interval == 0
                            or batch_idx == len(cv_val_loader)
                        ):
                            batch_pbar.set_postfix(
                                {
                                    "phase": "val",
                                    "VaL": f"{(v_loss / batch_idx):.3f}",
                                    "Acc": f"{(100 * correct / val_seen):.2f}%",
                                }
                            )
                batch_pbar.close()

                avg_tr_loss = tr_loss / len(cv_train_loader)
                avg_v_loss = v_loss / len(cv_val_loader)
                v_acc = 100 * correct / len(val_sub)

                fold_history["train_loss"].append(avg_tr_loss)
                fold_history["val_loss"].append(avg_v_loss)
                fold_history["val_acc"].append(v_acc)

                epoch_pbar.set_postfix(
                    {
                        "TrL": f"{avg_tr_loss:.3f}",
                        "VaL": f"{avg_v_loss:.3f}",
                        "Acc": f"{v_acc:.2f}%",
                        "BestL": f"{best_val_loss:.3f}",
                        "BestA": f"{best_val_acc:.2f}%",
                    }
                )
                epoch_pbar.update(1)

                if avg_v_loss < best_val_loss:
                    best_val_loss = avg_v_loss
                    best_val_acc = v_acc
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.results_path, f"model_fold_{fold + 1}.pth"),
                    )
                    epoch_pbar.set_postfix(
                        {
                            "TrL": f"{avg_tr_loss:.4f}",
                            "VaL": f"{avg_v_loss:.4f}",
                            "Acc": f"{v_acc:.2f}%",
                            "BestL": f"{best_val_loss:.4f}",
                            "Best": f"{best_val_acc:.2f}%",
                            "*": "saved",
                        }
                    )

                early_stopper(avg_v_loss)
                if early_stopper.early_stop:
                    tqdm.write(
                        f"\nEarly stopping triggered at epoch {epoch + 1} for Fold {fold + 1}"
                    )
                    break

            epoch_pbar.close()
            fold_train_time_seconds = time.time() - fold_train_start
            fold_history["train_time_seconds"] = fold_train_time_seconds
            self.all_fold_metrics[f"fold_{fold + 1}"] = fold_history
            with open(
                os.path.join(self.results_path, f"metrics_fold_{fold + 1}.json"), "w"
            ) as f:
                json.dump(fold_history, f, indent=2)
        with open(os.path.join(self.results_path, "metrics.json"), "w") as f:
            json.dump({"folds": self.all_fold_metrics}, f, indent=2)

    def evaluate(self, test_loader, test_dataset):
        print(f"\n{'#' * 35}")
        print(" Test Set Evaluation ")
        print(f"{'#' * 35}")

        test_accs = []
        for fold in range(1, self.n_folds + 1):
            load_path = os.path.join(self.results_path, f"model_fold_{fold}.pth")
            self.model.load_state_dict(torch.load(load_path))
            self.model.eval()
            correct = 0
            with torch.no_grad():
                for imgs, lbls in test_loader:
                    imgs = imgs.to(self.device, non_blocking=True)
                    lbls = lbls.to(self.device, non_blocking=True)
                    correct += (self.model(imgs).argmax(1) == lbls).sum().item()

            t_acc = 100 * correct / len(test_dataset)
            test_accs.append(t_acc)
            print(f"Model Fold {fold} | Test Acc: {t_acc:.2f}%")

        evaluation_metrics = {
            "test_acc_per_fold": test_accs,
            "avg_test_acc": float(np.mean(test_accs)),
            "std_test_acc": float(np.std(test_accs)),
        }
        with open(os.path.join(self.results_path, "metrics_evaluation.json"), "w") as f:
            json.dump(evaluation_metrics, f, indent=2)
        with open(os.path.join(self.results_path, "metrics.json"), "w") as f:
            json.dump(
                {"folds": self.all_fold_metrics, "evaluation": evaluation_metrics},
                f,
                indent=2,
            )

        print(f"\nAvg Test Acc: {np.mean(test_accs):.2f}% ± {np.std(test_accs):.2f}%")
