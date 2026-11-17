import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from transformers import AutoModel, AutoConfig


class MaskedAttentionPooling(nn.Module):
    """
    Gộp thông tin bằng cơ chế chú ý có mặt nạ (Masked Attention Pooling):
    Tính trọng số chú ý cho từng token, áp dụng mặt nạ che triệt để các vị trí <pad>
    trước khi chuẩn hóa Softmax, sau đó lấy tổng có trọng số để tạo vector biểu diễn câu.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (batch_size, seq_len, hidden_size)
        attention_mask: (batch_size, seq_len) - 1 cho token thật, 0 cho token <pad>
        Returns: (batch_size, hidden_size)
        """
        # (batch_size, seq_len, 1)
        scores = self.attention(hidden_states)

        # Gán giá trị cực âm cho các vị trí <pad> để sau softmax xác suất = 0
        mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        pooled = torch.sum(hidden_states * weights, dim=1)  # (batch_size, hidden_size)
        return pooled


class MultiSampleDropoutHead(nn.Module):
    """
    Nhánh suy luận chuyên biệt (Task-Specific Head) kết hợp Multi-Sample Dropout:
    1. Cụm biến đổi: Linear → GELU → LayerNorm
    2. Multi-Sample Dropout: Chiếu đặc trưng qua nhiều dropout mask độc lập song song,
       tính logits trên từng luồng và lấy trung bình cộng đầu ra.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_features: Optional[int] = None,
        num_samples: int = 5,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features

        # Cụm biến đổi đặc trưng chuyên biệt cho từng task
        self.transform = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.LayerNorm(hidden_features),
        )

        # Danh sách các mặt nạ Dropout độc lập
        self.num_samples = num_samples
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_rate) for _ in range(num_samples)
        ])
        self.classifier = nn.Linear(hidden_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.transform(x)  # (batch_size, hidden_features)

        # Nếu đang ở chế độ eval, chỉ cần forward 1 lần
        if not self.training or self.num_samples <= 1:
            return self.classifier(h)

        # Trong quá trình training: Lấy trung bình cộng từ nhiều luồng dropout
        logits_list = [self.classifier(drop(h)) for drop in self.dropouts]
        return torch.mean(torch.stack(logits_list, dim=0), dim=0)


class BaMiBERT(nn.Module):
    """
    Kiến trúc học đa nhiệm BaMiBERT cho Sentiment (3 classes) & Toxicity (4 classes):
    - Backbone: Transformer Encoder tiếng Việt (PhoBERT, ViBERT, XLM-R, v.v.)
    - Multi-Layer Feature Fusion: Kết hợp thông tin từ N tầng cuối của Encoder
    - Masked Attention Pooling: Gộp vector câu theo cơ chế Attention có mặt nạ pad
    - Task-Specific Heads: Nhánh Sentiment và nhánh Toxicity độc lập
    - Multi-Sample Dropout: Chống overfitting trên các lớp thiểu số
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        num_sentiment_classes: int = 3,
        num_toxicity_classes: int = 4,
        num_fusion_layers: int = 4,
        num_dropout_samples: int = 5,
        dropout_rate: float = 0.2,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_fusion_layers = num_fusion_layers

        if pretrained:
            config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
            self.encoder = AutoModel.from_pretrained(model_name, config=config)
        else:
            config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
            self.encoder = AutoModel.from_config(config)

        hidden_size = config.hidden_size

        # Trọng số có thể học (learnable weights) cho Multi-Layer Feature Fusion
        # Áp dụng Softmax trên các trọng số này để tổ hợp mượt mà N tầng cuối
        self.layer_weights = nn.Parameter(torch.zeros(num_fusion_layers))

        # Masked Attention Pooling
        self.pooling = MaskedAttentionPooling(hidden_size)

        # Nhánh phân loại chuyên biệt
        self.sentiment_head = MultiSampleDropoutHead(
            in_features=hidden_size,
            num_classes=num_sentiment_classes,
            num_samples=num_dropout_samples,
            dropout_rate=dropout_rate,
        )

        self.toxicity_head = MultiSampleDropoutHead(
            in_features=hidden_size,
            num_classes=num_toxicity_classes,
            num_samples=num_dropout_samples,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Trích xuất biểu diễn ẩn từ Encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # all_hidden_states: Tuple gồm (input_embeddings, layer_1, ..., layer_L)
        hidden_states = outputs.hidden_states

        # 1. Multi-Layer Feature Fusion: Lấy N tầng cuối cùng
        last_n_layers = hidden_states[-self.num_fusion_layers:]  # List các tensor (batch, seq, hidden)
        stacked_layers = torch.stack(last_n_layers, dim=0)      # (num_layers, batch, seq, hidden)

        # Chuẩn hóa trọng số tầng bằng Softmax
        normalized_weights = F.softmax(self.layer_weights, dim=0).view(
            self.num_fusion_layers, 1, 1, 1
        )
        fused_hidden = torch.sum(stacked_layers * normalized_weights, dim=0)  # (batch, seq, hidden)

        # 2. Masked Attention Pooling: Gộp vector đại diện câu
        sentence_vector = self.pooling(fused_hidden, attention_mask)  # (batch, hidden)

        # 3. Task-Specific Heads: Dự đoán đồng thời 2 nhiệm vụ
        sentiment_logits = self.sentiment_head(sentence_vector)
        toxicity_logits = self.toxicity_head(sentence_vector)

        return {
            "sentiment_logits": sentiment_logits,
            "toxicity_logits": toxicity_logits,
            "sentence_embedding": sentence_vector,
        }
