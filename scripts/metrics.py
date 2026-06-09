from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ClassificationMetrics:
    loss: float
    accuracy: float
    top5_accuracy: float
    precision: float
    recall: float
    f1: float
    samples: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "top5_accuracy": self.top5_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "samples": self.samples,
        }


class ClassificationMetricTracker:
    def __init__(self, num_classes: int):
        if num_classes <= 0:
            raise ValueError("num_classes must be greater than 0.")

        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.total_loss = 0.0
        self.total_samples = 0
        self.total_correct = 0
        self.total_top5_correct = 0
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
        )

    def update(
        self,
        logits: torch.Tensor,
        one_hot_targets: torch.Tensor,
        loss: float | torch.Tensor,
    ) -> None:
        if logits.ndim != 2:
            raise ValueError(
                f"logits must have shape (batch, classes), got {tuple(logits.shape)}."
            )
        if one_hot_targets.shape != logits.shape:
            raise ValueError(
                "one_hot_targets must have the same shape as logits, "
                f"got {tuple(one_hot_targets.shape)} and {tuple(logits.shape)}."
            )
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} classes, got {logits.shape[1]}."
            )

        batch_size = logits.shape[0]
        targets = one_hot_targets.argmax(dim=1)
        predictions = logits.argmax(dim=1)
        top_k = min(5, self.num_classes)
        top_predictions = logits.topk(top_k, dim=1).indices

        loss_value = float(loss.detach().item()) if torch.is_tensor(loss) else float(loss)
        self.total_loss += loss_value * batch_size
        self.total_samples += batch_size
        self.total_correct += int((predictions == targets).sum().item())
        self.total_top5_correct += int(
            top_predictions.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
        )

        encoded_pairs = (
            targets.detach().to("cpu", dtype=torch.int64) * self.num_classes
            + predictions.detach().to("cpu", dtype=torch.int64)
        )
        self.confusion_matrix += torch.bincount(
            encoded_pairs,
            minlength=self.num_classes * self.num_classes,
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> ClassificationMetrics:
        if self.total_samples == 0:
            raise RuntimeError("No samples were added to the metric tracker.")

        confusion = self.confusion_matrix.to(torch.float64)
        true_positive = confusion.diag()
        predicted_count = confusion.sum(dim=0)
        target_count = confusion.sum(dim=1)

        precision_per_class = torch.where(
            predicted_count > 0,
            true_positive / predicted_count,
            torch.zeros_like(true_positive),
        )
        recall_per_class = torch.where(
            target_count > 0,
            true_positive / target_count,
            torch.zeros_like(true_positive),
        )
        f1_denominator = precision_per_class + recall_per_class
        f1_per_class = torch.where(
            f1_denominator > 0,
            2 * precision_per_class * recall_per_class / f1_denominator,
            torch.zeros_like(f1_denominator),
        )

        return ClassificationMetrics(
            loss=self.total_loss / self.total_samples,
            accuracy=self.total_correct / self.total_samples,
            top5_accuracy=self.total_top5_correct / self.total_samples,
            precision=float(precision_per_class.mean().item()),
            recall=float(recall_per_class.mean().item()),
            f1=float(f1_per_class.mean().item()),
            samples=self.total_samples,
        )
