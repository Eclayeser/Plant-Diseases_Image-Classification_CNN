import sys
import torch
from torch import nn
from pathlib import Path
from leafapp.eval_functions import (
    calculate_metrics_per_loader,
    plot_training_curve,
    plot_confusion_matrix,
    plot_per_class_metrics,
)
from leafapp.data import get_loaders
from leafapp.checkpoints import load_checkpoint


def main(checkpoint_path: str|Path, data_type: str):
    # load checkpoint info
    model, _, history_metrics, config = load_checkpoint(checkpoint_path)

    # load data to evaluate against
    _, val_loader, test_loader, class_names = get_loaders(
            fraction=config["data"]["fraction"],
            image_size=config["data"]["image_size"],
            batch_size=config["data"]["batch_size"],
            seed=config["seed"],
        )
    
    accepted_data_types = {"validation": val_loader, "test": test_loader} 
    if data_type not in accepted_data_types.keys():
        raise ValueError(f"data_type should be one of {accepted_data_types.keys()}")
    target_data = accepted_data_types[data_type]

    # run model and calculate metrics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    overall_metrics = calculate_metrics_per_loader(
            model=model,
            loader=target_data,
            device=device,
            class_names=class_names,
            criterion=nn.CrossEntropyLoss(),
            metrics_list=[
                "accuracy",
                "top-3-accuracy",
                "top-5-accuracy",
                "precision",
                "f1",
                "recall",
                "confusion-matrix",
                "per-class-metrics"
            ]
        )

    print(
        f"Top-1 Accuracy: {overall_metrics['accuracy']}\n"
        f"Top-3 Accuracy: {overall_metrics['top-3-accuracy']}\n"
        f"Top-5 Accuracy: {overall_metrics['top-5-accuracy']}\n"
        f"Macro Precision: {overall_metrics['precision']}\n"
        f"Macro Recall: {overall_metrics['recall']}\n"
        f"Macro F1: {overall_metrics['f1']}\n"
        f"Per-Class Metrics:\n"
        f"{overall_metrics['per-class-metrics']}"
    )

    plot_training_curve(history_metrics)
    plot_confusion_matrix(overall_metrics["confusion-matrix"], class_names)
    
    
    
    # TODO: save final metrics locally(include associated model)


if __name__ == "__main__":
    try:
        checkpoint_path = Path(sys.argv[1])
        data_type = str(sys.argv[2])

    except:
        raise ValueError(
                            f"Failed to process required inputs. Check the following provided:\n"
                            f" - First input: checkpoint_path (str | Path)\n"
                            f" - Second input: data_type (str)\n"
                        )
    
    main(checkpoint_path, data_type)