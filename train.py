import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, set_seed

from src.dataset import (
    MultiTaskDataset,
    DynamicPaddingCollator,
    deduplicate_dataframe,
    TOXICITY_TO_ID,
)
from src.losses import JointMultiTaskLoss
from src.model import BaMiBERT
from src.trainer import build_optimizer_and_scheduler, Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình đa nhiệm BaMiBERT")

    # Dữ liệu
    parser.add_argument("--train_path", type=str, default="data/train/train.csv", help="Đường dẫn file train.csv")
    parser.add_argument("--valid_path", type=str, default="data/valid/valid.csv", help="Đường dẫn file valid.csv")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Thư mục lưu weights và metrics")
    parser.add_argument("--dedup", action="store_true", default=True, help="Khử trùng lặp bình luận trước khi huấn luyện")

    # Mô hình
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base-v2", help="Tên backbone pre-trained HuggingFace")
    parser.add_argument("--num_fusion_layers", type=int, default=4, help="Số tầng cuối lấy đặc trưng (Feature Fusion)")
    parser.add_argument("--num_dropout_samples", type=int, default=5, help="Số luồng Multi-Sample Dropout")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Tỷ lệ dropout")
    parser.add_argument("--max_length", type=int, default=256, help="Chiều dài tối đa của câu sau khi tokenize")

    # Huấn luyện
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước lô (batch size)")
    parser.add_argument("--epochs", type=int, default=6, help="Số epoch tối đa")
    parser.add_argument("--lr_backbone", type=float, default=2e-5, help="Learning rate cho Transformer Encoder")
    parser.add_argument("--lr_heads", type=float, default=1e-4, help="Learning rate cho Classification Heads & Pooling")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Hệ số suy giảm trọng số")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Tỷ lệ warmup")
    parser.add_argument("--patience", type=int, default=3, help="Số epoch không cải thiện trước khi dừng sớm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Loss parameters
    parser.add_argument("--tau", type=float, default=1.0, help="Hệ số hiệu chỉnh Logit Adjustment cho Toxicity")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Hệ số làm mịn nhãn cho Sentiment")
    parser.add_argument("--weight_sent", type=float, default=1.0, help="Trọng số loss Sentiment")
    parser.add_argument("--weight_tox", type=float, default=1.0, help="Trọng số loss Toxicity")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Kiểm tra thiết bị tính toán
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== KHỞI CHẠY HUẤN LUYỆN BAMIBERT ===")
    print(f"Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Backbone: {args.model_name}")

    # 1. Đọc dữ liệu
    print("\n[1/5] Đang đọc dữ liệu...")
    train_df = pd.read_csv(args.train_path)
    valid_df = pd.read_csv(args.valid_path)
    print(f"Train samples: {len(train_df)}, Valid samples: {len(valid_df)}")

    # Khử trùng lặp (Pre-split Deduplication)
    if args.dedup:
        print("Áp dụng khử trùng lặp (Pre-split Deduplication)...")
        train_df = deduplicate_dataframe(train_df, text_col="Comment")

    # 2. Tính toán phân phối lớp cho Logit-Adjustment (Toxicity)
    tox_counts = train_df["Toxicity"].astype(int).value_counts().to_dict()
    num_tox_classes = 4
    tox_freqs = [tox_counts.get(i, 1) for i in range(num_tox_classes)]
    print(f"Tần suất các lớp Toxicity (0->3): {tox_freqs}")

    # 3. Khởi tạo Tokenizer & DataLoader với Dynamic Padding
    print("\n[2/5] Khởi tạo Tokenizer & Dynamic Padding Collator...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    train_dataset = MultiTaskDataset(
        df=train_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    valid_dataset = MultiTaskDataset(
        df=valid_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    collator = DynamicPaddingCollator(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # 4. Xây dựng Kiến trúc BaMiBERT & Hàm Loss
    print("\n[3/5] Khởi tạo kiến trúc BaMiBERT & Loss functions...")
    model = BaMiBERT(
        model_name=args.model_name,
        num_sentiment_classes=3,
        num_toxicity_classes=4,
        num_fusion_layers=args.num_fusion_layers,
        num_dropout_samples=args.num_dropout_samples,
        dropout_rate=args.dropout_rate,
        pretrained=True,
    )

    criterion = JointMultiTaskLoss(
        toxicity_class_frequencies=tox_freqs,
        tau_toxicity=args.tau,
        epsilon_sentiment=args.label_smoothing,
        weight_sentiment=args.weight_sent,
        weight_toxicity=args.weight_tox,
    )

    # 5. Khởi tạo Optimizer & Scheduler (Differential Learning Rate)
    print("\n[4/5] Thiết lập Differential Learning Rate & Cosine Warmup...")
    total_training_steps = len(train_loader) * args.epochs
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        total_training_steps=total_training_steps,
        lr_backbone=args.lr_backbone,
        lr_heads=args.lr_heads,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
    )

    # 6. Bắt đầu quá trình Huấn luyện
    print("\n[5/5] Bắt đầu quá trình huấn luyện & kiểm định...")
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        max_grad_norm=1.0,
        use_amp=True,
        patience=args.patience,
    )

    trainer.fit(train_loader, valid_loader, epochs=args.epochs)


if __name__ == "__main__":
    main()
