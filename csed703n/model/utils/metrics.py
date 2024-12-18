from torchmetrics import MetricCollection
from torchmetrics.classification import AUROC, Accuracy, F1Score, Precision, Recall


def threshold_metrics(num_classes: int) -> MetricCollection:
    task = "binary" if num_classes == 2 else "multiclass"

    metrics = [
        Accuracy(task=task, num_classes=num_classes),
        Precision(task=task, num_classes=num_classes),
        Recall(task=task, num_classes=num_classes),
        F1Score(task=task, num_classes=num_classes),
    ]

    return MetricCollection(metrics)


def continuous_metrics(num_classes: int) -> MetricCollection:
    task = "binary" if num_classes == 2 else "multiclass"

    return MetricCollection([AUROC(task=task, num_classes=num_classes)])
