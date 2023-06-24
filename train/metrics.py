import evaluate
import numpy as np
metric = evaluate.load("seqeval")

def compute_metrics(eval_preds, label_names):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    metrics = {
        "precision": all_metrics.get("overall_precision", 0.),
        "recall": all_metrics.get("overall_recall", 0.),
        "f1": all_metrics.get("overall_f1", 0.),
        "accuracy": all_metrics.get("overall_accuracy", 0.),
    }
    for key, values in all_metrics.items():
        if type(values) is dict:
            metrics.update({f"{key}_{k}": v for k, v in values.items()
                            if k in {"f1", "number"}
                           })
    return metrics