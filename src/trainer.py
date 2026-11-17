import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm

from src.dataset import MultiTaskDataset, DynamicPaddingCollator, clean_text_minimalist
from src.losses import JointMultiTaskLoss
from src.metrics import compute_metrics, print_evaluation_summary


def create_joint_stratified_groups(
    df: pd.DataFrame,
    sent_col: str = "Sentiment",
    tox_col: str = "Toxicity",
    min_count: int = 5,
) -> pd.Series:
    """
    Tạo nhóm phân tầng theo tổ hợp nhãn ghép (Sentiment, Toxicity).
    Các tổ hợp cực hiếm có số mẫu < min_count sẽ được gom vào nhóm 'rare'
    để tránh lỗi phân chia phân tầng.
    """
    joint_labels = df[sent_col].astype(str) + "_" + df[tox_col].astype(str)
    counts = joint_labels.value_counts()
    rare_classes = counts[counts < min_count].index.tolist()
    strat_labels = joint_labels.apply(lambda x: "rare" if x in rare_classes else x)
    return strat_labels


def build_optimizer_and_scheduler(
    model: nn.Module,
    total_training_steps: int,
    lr_backbone: float = 2e-5,
    lr_heads: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
):
    """
    Tốc độ học phân tầng (Differential Learning Rate) & Tối ưu hóa:
    - Backbone (Encoder): lr thấp để bảo tồn tri thức tiếng Việt tiền huấn luyện.
    - Heads + Pooling + Fusion: lr cao hơn để hội tụ nhanh.
    - Loại trừ bias và LayerNorm khỏi Weight Decay.
    - Bộ tối ưu AdamW + Cosine Warmup.
    """
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

    backbone_params = list(model.encoder.named_parameters())
    head_params = (
        list(model.pooling.named_parameters())
        + list(model.sentiment_head.named_parameters())
        + list(model.toxicity_head.named_parameters())
        + [("layer_weights", model.layer_weights)]
    )

    optimizer_grouped_parameters = [
        # 1. Backbone có weight decay
        {
            "params": [
                p for n, p in backbone_params if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "lr": lr_backbone,
            "weight_decay": weight_decay,
        },
        # 2. Backbone không weight decay (bias, layernorm)
        {
            "params": [
                p for n, p in backbone_params if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "lr": lr_backbone,
            "weight_decay": 0.0,
        },
        # 3. Heads & Pooling có weight decay
        {
            "params": [
                p for n, p in head_params if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "lr": lr_heads,
            "weight_decay": weight_decay,
        },
        # 4. Heads & Pooling không weight decay
        {
            "params": [
                p for n, p in head_params if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "lr": lr_heads,
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    warmup_steps = int(total_training_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    return optimizer, scheduler


class Trainer:
    """
    Trainer quản lý toàn bộ vòng lặp huấn luyện, đánh giá và Early Stopping đa mục tiêu.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: JointMultiTaskLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        output_dir: str = "checkpoints",
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        patience: int = 3,
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and device.type == "cuda"
        self.patience = patience

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        os.makedirs(output_dir, exist_ok=True)

        self.best_composite_score = -1.0
        self.best_metrics = {}
        self.no_improve_epochs = 0

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            sent_targets = batch["sentiment_labels"].to(self.device)
            tox_targets = batch["toxicity_labels"].to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(input_ids, attention_mask)
                loss, loss_sent, loss_tox = self.criterion(
                    outputs["sentiment_logits"],
                    outputs["toxicity_logits"],
                    sent_targets,
                    tox_targets,
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "s_loss": f"{loss_sent.item():.4f}",
                "t_loss": f"{loss_tox.item():.4f}",
            })

        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        sent_preds, sent_targets_list = [], []
        tox_preds, tox_targets_list = [], []
        tox_probs_list = []

        for batch in tqdm(val_loader, desc="[Validation]", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(input_ids, attention_mask)

            sent_logits = outputs["sentiment_logits"]
            tox_logits = outputs["toxicity_logits"]

            # Quyết định nhãn trực tiếp từ logits/xác suất đã hiệu chỉnh
            sent_pred = torch.argmax(sent_logits, dim=-1)
            tox_pred = torch.argmax(tox_logits, dim=-1)
            tox_prob = torch.softmax(tox_logits, dim=-1)

            sent_preds.extend(sent_pred.cpu().numpy())
            sent_targets_list.extend(batch["sentiment_labels"].numpy())
            tox_preds.extend(tox_pred.cpu().numpy())
            tox_targets_list.extend(batch["toxicity_labels"].numpy())
            tox_probs_list.append(tox_prob.cpu().numpy())

        sent_preds_arr = np.array(sent_preds)
        sent_targets_arr = np.array(sent_targets_list)
        tox_preds_arr = np.array(tox_preds)
        tox_targets_arr = np.array(tox_targets_list)
        tox_probs_arr = np.vstack(tox_probs_list)

        metrics = compute_metrics(
            sentiment_preds=sent_preds_arr,
            sentiment_targets=sent_targets_arr,
            toxicity_preds=tox_preds_arr,
            toxicity_targets=tox_targets_arr,
            toxicity_probs=tox_probs_arr,
        )
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        print(f"Starting training for {epochs} epochs on device: {self.device} (AMP: {self.use_amp})")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)

            composite_score = val_metrics["composite_score"]
            print(
                f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
                f"Val Composite F1: {composite_score:.4f} "
                f"(Sent: {val_metrics['sentiment_macro_f1']:.4f}, Tox: {val_metrics['toxicity_macro_f1']:.4f})"
            )

            # Composite Early Stopping: Lưu trạng thái mô hình tốt nhất
            if composite_score > self.best_composite_score:
                self.best_composite_score = composite_score
                self.best_metrics = val_metrics
                self.no_improve_epochs = 0
                best_model_path = os.path.join(self.output_dir, "best_model.pt")
                torch.save(self.model.state_dict(), best_model_path)
                print(f"  ★ Đạt kỷ lục mới! Đã lưu checkpoint tại {best_model_path}")
            else:
                self.no_improve_epochs += 1
                print(f"  Chưa cải thiện ({self.no_improve_epochs}/{self.patience})")
                if self.no_improve_epochs >= self.patience:
                    print(f"  [Early Stopping] Dừng sớm sau {epoch} epochs để chống overfitting.")
                    break

        print("\n" + "=" * 65)
        print("Huấn luyện hoàn tất! Kết quả tốt nhất trên Validation:")
        print_evaluation_summary(self.best_metrics, title="BEST VALIDATION METRICS")

        # Lưu best metrics ra file JSON
        with open(os.path.join(self.output_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(self.best_metrics, f, indent=2, ensure_ascii=False)
