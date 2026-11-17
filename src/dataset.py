import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from transformers import PreTrainedTokenizerBase
except ImportError:
    PreTrainedTokenizerBase = Any



# --- Minimalist Text Cleaning ---
ZERO_WIDTH_CHARS = re.compile(r'[\u200B\u200C\u200D\u200E\u200F\uFEFF]')
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
MULTI_SPACE_PATTERN = re.compile(r'[^\S\r\n]+')


def clean_text_minimalist(text: str) -> str:
    """
    Tiền xử lý tối giản (Minimalist Preprocessing) theo đặc tả:
    1. Chuyển Unicode dựng sẵn NFC.
    2. Loại bỏ BOM, ký tự trắng không độ rộng (zero-width) và ký tự điều khiển ẩn.
    3. Loại bỏ thẻ HTML và đường dẫn URL.
    4. Gộp khoảng trắng liên tiếp.
    5. BẢO TOÀN NGUYÊN TRẠNG: Không tách từ thủ công; giữ hoa/thường, dấu câu spam, emoji, teencode.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # 2. Xóa zero-width space & BOM
    text = ZERO_WIDTH_CHARS.sub("", text)

    # Lọc mã điều khiển ẩn (trừ xuống dòng và tab)
    text = "".join(
        ch for ch in text
        if not (unicodedata.category(ch).startswith("C") and ch not in ("\n", "\t", "\r"))
    )

    # 3. Làm sạch nhiễu phi ngôn ngữ: HTML tags & URLs
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)

    # 4. Gộp khoảng trắng thừa
    text = MULTI_SPACE_PATTERN.sub(" ", text)

    return text.strip()


# --- Nhãn & Mapping ---
# Sentiment trong data: -1 (Negative), 0 (Neutral), 1 (Positive)
# Chuyển về 0, 1, 2 để phù hợp với hàm Loss của PyTorch
SENTIMENT_TO_ID = {-1: 0, 0: 1, 1: 2}
ID_TO_SENTIMENT = {0: -1, 1: 0, 2: 1}
SENTIMENT_NAMES = {0: "Negative (-1)", 1: "Neutral (0)", 2: "Positive (1)"}

TOXICITY_TO_ID = {0: 0, 1: 1, 2: 2, 3: 3}
ID_TO_TOXICITY = {0: 0, 1: 1, 2: 2, 3: 3}
TOXICITY_NAMES = {
    0: "Clean (0)",
    1: "Toxic (1)",
    2: "Very Toxic (2)",
    3: "Extremely Toxic (3)",
}


def deduplicate_dataframe(df: pd.DataFrame, text_col: str = "Comment") -> pd.DataFrame:
    """
    Khử trùng lặp bình luận sau khi chuẩn hóa văn bản để triệt tiêu Data Leakage.
    """
    df = df.copy()
    cleaned = df[text_col].astype(str).apply(clean_text_minimalist)
    df["_cleaned_text"] = cleaned
    before = len(df)
    df = df.drop_duplicates(subset=["_cleaned_text"]).reset_index(drop=True)
    df = df.drop(columns=["_cleaned_text"])
    after = len(df)
    print(f"Deduplication: {before} -> {after} samples (removed {before - after} duplicates)")
    return df


class MultiTaskDataset(Dataset):
    """
    Dataset phục vụ bài toán đa nhiệm (Sentiment & Toxicity).
    Không thực hiện đệm tĩnh (fixed padding) tại đây, việc đệm động sẽ do Collator xử lý.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        text_col: str = "Comment",
        sentiment_col: Optional[str] = "Sentiment",
        toxicity_col: Optional[str] = "Toxicity",
        max_length: int = 256,
        is_test: bool = False,
    ):
        self.texts = df[text_col].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        self.has_labels = (
            sentiment_col in df.columns and toxicity_col in df.columns and not is_test
        )

        if self.has_labels:
            self.sentiment_labels = [
                SENTIMENT_TO_ID[val] for val in df[sentiment_col].astype(int)
            ]
            self.toxicity_labels = [
                TOXICITY_TO_ID[val] for val in df[toxicity_col].astype(int)
            ]
        else:
            self.sentiment_labels = None
            self.toxicity_labels = None

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw_text = self.texts[idx]
        cleaned_text = clean_text_minimalist(raw_text)

        # Tokenize không đệm tĩnh (truncation theo max_length)
        encoding = self.tokenizer(
            cleaned_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_token_type_ids=False,
        )

        item = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "raw_text": raw_text,
        }

        if self.has_labels:
            item["sentiment_label"] = self.sentiment_labels[idx]
            item["toxicity_label"] = self.toxicity_labels[idx]

        return item


class DynamicPaddingCollator:
    """
    Đệm động theo lô (Dynamic Padding):
    Chỉ đệm token <pad> theo độ dài thực tế của câu dài nhất TRONG TỪNG LÔ (batch).
    Giảm đáng kể khối lượng tính toán dư thừa và tiết kiệm bộ nhớ GPU.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: Optional[int] = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        features = [
            {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}
            for item in batch
        ]

        batch_encoding = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        output = {
            "input_ids": batch_encoding["input_ids"],
            "attention_mask": batch_encoding["attention_mask"],
        }

        if "sentiment_label" in batch[0]:
            output["sentiment_labels"] = torch.tensor(
                [item["sentiment_label"] for item in batch], dtype=torch.long
            )
            output["toxicity_labels"] = torch.tensor(
                [item["toxicity_label"] for item in batch], dtype=torch.long
            )

        if "raw_text" in batch[0]:
            output["raw_texts"] = [item["raw_text"] for item in batch]

        return output
