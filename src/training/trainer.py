import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from src.training.loss    import get_loss_function
from src.training.metrics import get_accuracy
import logging
import csv
from configs.config import CONFIG


logger = logging.getLogger(__name__)
scheduler     = CONFIG["scheduler"]

class Trainer:
    def __init__(self, model, dataloaders: dict, config: dict):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f'Training on: {self.device}')

        self.model = model.to(self.device)

        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']

        self.epochs = config['epochs']
        self.save_dir = config['save_dir']

        self.criterion = get_loss_function(
            label_smoothing=config.get("label_smoothing", 0.1)
        )

        head_params = [p for n, p in self.model.model.named_parameters() if n.startswith('fc')]
        # Grab all parameters EXCEPT the final layer
        backbone_params = [p for n, p in self.model.model.named_parameters() if not n.startswith('fc')]

        self.optimizer = Adam([
            {'params': backbone_params, 'lr': config.get("lr", 1e-4) * 0.1},
            {'params': head_params,     'lr': config.get("lr", 1e-4)}
            ],
            weight_decay = config.get("weight_decay", 1e-4)
        )

        self.best_val_acc = 0.0
        os.makedirs(self.save_dir, exist_ok=True)

        self.warmup_scheduler = LinearLR(
            self.optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=3
        )

        self.main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.epochs - 3)
        )

        self.scheduler = SequentialLR(
            self.optimizer, 
            schedulers=[self.warmup_scheduler, self.main_scheduler], 
            milestones=[3]
        )



        self.patience = scheduler["patience"]
        self.patience_counter = 0

        self.history = {
            "train_loss" : [],
            "train_acc"  : [],
            "val_loss"   : [],
            "val_acc"    : []
        }

        self.csv_path = os.path.join(
            self.save_dir, "training_log.csv"
        )

        self._init_csv()

    def _init_csv(self):
        with open(self.csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "lr"
            ])
        print(f'log saved {self.csv_path}')
    
    def _log_to_csv(
        self,
        epoch        : int,
        train_metrics: dict,
        val_metrics  : dict,
        lr           : float
    ):
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                round(train_metrics["loss"], 4),
                round(train_metrics["acc"],  2),
                round(val_metrics["loss"],   4),
                round(val_metrics["acc"],    2),
                round(lr, 6)
            ])



    def _train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        total_acc  = 0.0
        batches    = len(self.train_loader)

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss    = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            acc         = get_accuracy(outputs, labels)
            total_loss += loss.item()
            total_acc  += acc

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}] "
                    f"Batch [{batch_idx+1}/{batches}] "
                    f"Loss: {loss.item():.4f} "
                    f"Acc: {acc:.2f}%")

        return {
            "loss" : total_loss / batches,
            "acc"  : total_acc  / batches
        }


    def _validate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_acc  = 0.0
        batches    = len(self.val_loader)

        with torch.no_grad():
            for images, labels in self.val_loader:
                images  = images.to(self.device)
                labels  = labels.to(self.device)
                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)

                total_loss += loss.item()
                total_acc  += get_accuracy(outputs, labels)

        return {
            "loss" : total_loss / batches,
            "acc"  : total_acc  / batches
        }



    def _save_checkpoint(self, epoch: int, val_acc: float):
        path = os.path.join(self.save_dir, "best_model.pth")
        torch.save({
            "epoch"     : epoch + 1,
            "model"     : self.model.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
            "val_acc"   : val_acc,
            "scheduler" : self.scheduler.state_dict(),
            "history"   : self.history
        }, path)

    def train(self):
        print("\n" + "=" * 50)
        print("  Starting Training")
        print("=" * 50 + "\n")

        for epoch in range(self.epochs):
            print(f"Epoch [{epoch+1:02d}/{self.epochs}]")

            train_metrics = self._train_one_epoch(epoch)
            val_metrics   = self._validate()
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            self._log_to_csv(
                epoch,
                train_metrics,
                val_metrics,
                current_lr
            )

            print(
                f"Train → Loss: {train_metrics['loss']:.4f}"
                f"  |Acc: {train_metrics['acc']:.2f}%\n"
                f"Val   → Loss: {val_metrics['loss']:.4f}"
                f"  |Acc: {val_metrics['acc']:.2f}%"
                f"  |LR    → {current_lr:.6f}"
            )

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["acc"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["acc"])   

            if val_metrics["acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["acc"]
                self._save_checkpoint(epoch, val_metrics["acc"])
                self.patience_counter = 0  
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.warning(f"Early stopping triggered at epoch {epoch+1}. Training stopped.")
                    break

            print("-" * 50)
