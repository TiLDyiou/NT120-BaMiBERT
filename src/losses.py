import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List


class LogitAdjustedCrossEntropyLoss(nn.Module):
    """
    Logit-Adjusted Cross-Entropy Loss (Menon et al., ICLR 2021).
    Kỹ thuật xử lý mất cân bằng lớp trầm trọng (Heavy Class Imbalance) cho nhánh Toxicity.

    Cơ chế:
        logits_adjusted = logits + tau * log(class_priors + 1e-12)
        loss = CrossEntropy(logits_adjusted, targets)

    Thay vì phạt gradient theo trọng số nghịch đảo (dễ gây mất ổn định huấn luyện),
    phương pháp này mở rộng margin phân định cho các lớp thiểu số trực tiếp trong không gian logit.
    """

    def __init__(
        self,
        class_frequencies: Union[List[int], torch.Tensor],
        tau: float = 1.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.tau = tau
        self.reduction = reduction

        if not isinstance(class_frequencies, torch.Tensor):
            class_frequencies = torch.tensor(class_frequencies, dtype=torch.float32)

        # Chuẩn hóa về phân phối xác suất tiên nghiệm (prior)
        total = class_frequencies.sum()
        priors = class_frequencies / total
        log_priors = torch.log(priors + 1e-12)

        # Đăng ký buffer để tự động chuyển device (CPU/GPU) theo model
        self.register_buffer("log_priors", log_priors)
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits shape: (batch_size, num_classes)
        adjusted_logits = logits + self.tau * self.log_priors.unsqueeze(0)
        return F.cross_entropy(
            adjusted_logits,
            targets,
            weight=self.weight,
            reduction=self.reduction,
        )


class LabelSmoothedCrossEntropyLoss(nn.Module):
    """
    Label-Smoothed Cross-Entropy Loss cho nhánh Sentiment.
    Mục đích: Giảm bớt sự tự tin cực đoan tại các vùng ranh giới cảm xúc mờ nhạt
    (đặc biệt giữa Neutral và Negative).
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.epsilon,
            reduction=self.reduction,
        )


class JointMultiTaskLoss(nn.Module):
    """
    Hàm mục tiêu tổng hợp (Joint Objective Function):
    Tổ hợp tuyến tính có kiểm soát giữa Sentiment Loss và Toxicity Loss.
        L_total = weight_sentiment * L_sentiment + weight_toxicity * L_toxicity
    """

    def __init__(
        self,
        toxicity_class_frequencies: Union[List[int], torch.Tensor],
        tau_toxicity: float = 1.0,
        epsilon_sentiment: float = 0.1,
        weight_sentiment: float = 1.0,
        weight_toxicity: float = 1.0,
    ):
        super().__init__()
        self.sentiment_criterion = LabelSmoothedCrossEntropyLoss(
            epsilon=epsilon_sentiment
        )
        self.toxicity_criterion = LogitAdjustedCrossEntropyLoss(
            class_frequencies=toxicity_class_frequencies,
            tau=tau_toxicity,
        )
        self.weight_sentiment = weight_sentiment
        self.weight_toxicity = weight_toxicity

    def forward(
        self,
        sentiment_logits: torch.Tensor,
        toxicity_logits: torch.Tensor,
        sentiment_targets: torch.Tensor,
        toxicity_targets: torch.Tensor,
    ) -> torch.Tensor:
        loss_sentiment = self.sentiment_criterion(sentiment_logits, sentiment_targets)
        loss_toxicity = self.toxicity_criterion(toxicity_logits, toxicity_targets)

        total_loss = (
            self.weight_sentiment * loss_sentiment
            + self.weight_toxicity * loss_toxicity
        )
        return total_loss, loss_sentiment, loss_toxicity
