import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.dataset import (
    MultiTaskDataset,
    DynamicPaddingCollator,
    ID_TO_SENTIMENT,
    ID_TO_TOXICITY,
    SENTIMENT_NAMES,
    TOXICITY_NAMES,
)
from src.model import BaMiBERT
from src.metrics import compute_metrics, print_evaluation_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình BaMiBERT & Phân tích lỗi")
    parser.add_argument("--data_path", type=str, default="data/challenge/challenge.csv", help="Đường dẫn file cần đánh giá")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Đường dẫn trọng số checkpoint tốt nhất")
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base-v2", help="Tên backbone pre-trained HuggingFace")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--max_length", type=int, default=256, help="Chiều dài tối đa câu")
    parser.add_argument("--output_file", type=str, default="evaluation_results.csv", help="File CSV lưu kết quả dự đoán chi tiết")
    return parser.parse_args()


@torch.no_grad()
def evaluate_and_analyze(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== ĐÁNH GIÁ MÔ HÌNH BAMIBERT ===")
    print(f"Thiết bị: {device}")
    print(f"Dữ liệu đánh giá: {args.data_path}")
    print(f"Checkpoint: {args.checkpoint}")

    # 1. Đọc dữ liệu
    df = pd.read_csv(args.data_path)
    print(f"Tổng số mẫu: {len(df)}")

    # 2. Tokenizer & DataLoader
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    dataset = MultiTaskDataset(df=df, tokenizer=tokenizer, max_length=args.max_length)
    collator = DynamicPaddingCollator(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # 3. Khởi tạo Model & Nạp weights
    model = BaMiBERT(model_name=args.model_name, pretrained=False)
    if os.path.exists(args.checkpoint):
        print(f"Đang nạp trọng số từ {args.checkpoint}...")
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"CẢNH BÁO: Không tìm thấy checkpoint tại {args.checkpoint}, sử dụng weights ngẫu nhiên.")

    model.to(device)
    model.eval()

    # 4. Suy luận
    sent_preds, tox_preds = [], []
    sent_probs_list, tox_probs_list = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask)
        sent_logits = outputs["sentiment_logits"]
        tox_logits = outputs["toxicity_logits"]

        s_prob = torch.softmax(sent_logits, dim=-1).cpu().numpy()
        t_prob = torch.softmax(tox_logits, dim=-1).cpu().numpy()

        s_pred = np.argmax(s_prob, axis=-1)
        t_pred = np.argmax(t_prob, axis=-1)

        sent_preds.extend(s_pred)
        tox_preds.extend(t_pred)
        sent_probs_list.append(s_prob)
        tox_probs_list.append(t_prob)

    sent_preds = np.array(sent_preds)
    tox_preds = np.array(tox_preds)
    tox_probs = np.vstack(tox_probs_list)

    # 5. Lưu kết quả chi tiết
    df["Pred_Sentiment_Raw"] = sent_preds
    df["Pred_Sentiment"] = [ID_TO_SENTIMENT[p] for p in sent_preds]
    df["Pred_Toxicity"] = [ID_TO_TOXICITY[p] for p in tox_preds]

    for c in range(4):
        df[f"Prob_Toxicity_{c}"] = tox_probs[:, c]

    df.to_csv(args.output_file, index=False, encoding="utf-8-sig")
    print(f"\nĐã xuất kết quả dự đoán chi tiết tại: {args.output_file}")

    # 6. Tính toán Metric nếu file có sẵn nhãn
    if "Sentiment" in df.columns and "Toxicity" in df.columns:
        sent_targets = np.array([{-1: 0, 0: 1, 1: 2}[int(x)] for x in df["Sentiment"]])
        tox_targets = np.array([int(x) for x in df["Toxicity"]])

        metrics = compute_metrics(
            sentiment_preds=sent_preds,
            sentiment_targets=sent_targets,
            toxicity_preds=tox_preds,
            toxicity_targets=tox_targets,
            toxicity_probs=tox_probs,
        )
        print_evaluation_summary(metrics, title="KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP DỮ LIỆU")

        # 7. Phân tích lỗi đặc thù (False Positive Analysis)
        print("\n=== PHÂN TÍCH LỖI ĐẶC THÙ (ERROR ANALYSIS) ===")
        # Phê bình lịch sự: Negative nhưng Non-toxic (Sentiment=-1, Toxicity=0)
        polite_criticism = df[(df["Sentiment"] == -1) & (df["Toxicity"] == 0)]
        polite_fp = polite_criticism[polite_criticism["Pred_Toxicity"] > 0]
        fp_rate = (len(polite_fp) / len(polite_criticism)) * 100 if len(polite_criticism) > 0 else 0
        print(f"1. Phê bình lịch sự (Negative nhưng Clean): {len(polite_criticism)} mẫu")
        print(f"   • Số mẫu bị nhận diện nhầm thành Toxic (False Positive): {len(polite_fp)} ({fp_rate:.2f}%)")

        if len(polite_fp) > 0:
            print("   • Ví dụ 2 mẫu False Positive điển hình:")
            for idx, (_, row) in enumerate(polite_fp.head(2).iterrows()):
                print(f"     [{idx+1}] Comment: \"{row['Comment']}\" -> Dự đoán Tox: {row['Pred_Toxicity']}")

        # Khen ngợi có dùng từ đệm/tục tĩu: Positive nhưng Toxic (Sentiment=1, Toxicity>0)
        praise_toxic = df[(df["Sentiment"] == 1) & (df["Toxicity"] > 0)]
        praise_detected = praise_toxic[praise_toxic["Pred_Toxicity"] > 0]
        det_rate = (len(praise_detected) / len(praise_toxic)) * 100 if len(praise_toxic) > 0 else 0
        print(f"\n2. Khen ngợi thân mật/tục tĩu (Positive nhưng Toxic): {len(praise_toxic)} mẫu")
        print(f"   • Số mẫu nhận diện đúng tính Toxic: {len(praise_detected)} ({det_rate:.2f}%)")


if __name__ == "__main__":
    args = parse_args()
    evaluate_and_analyze(args)
