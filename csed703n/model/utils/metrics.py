from torchmetrics import MetricCollection
from torchmetrics.classification import AUROC, Accuracy, F1Score, Precision, Recall


def create_metrics(num_classes: int, do_auroc: bool = True) -> MetricCollection:
    task = "binary" if num_classes == 2 else "multiclass"

    metrics = [
        Accuracy(task=task, num_classes=num_classes),
        Precision(task=task, num_classes=num_classes),
        Recall(task=task, average="macro", num_classes=num_classes),
        F1Score(task=task, average="macro", num_classes=num_classes),
    ]

    if do_auroc:
        metrics.append(AUROC(task=task, num_classes=num_classes))

    return MetricCollection(metrics)
