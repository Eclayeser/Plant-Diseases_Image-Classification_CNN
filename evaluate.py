# Evaluates a selected model and produces performance metrics and error analysis.
import argparse
import torch
from torch import nn
from pathlib import Path
from datetime import datetime
from leafapp.eval_functions import (
    calculate_metrics_per_loader,
    plot_and_save_training_curve,
    plot_and_save_confusion_matrix,
    save_evaluation_report,
)
from leafapp.data import get_loaders
from leafapp.checkpoints import load_checkpoint
from leafapp.tracking import NAMES, RUN_TYPES, validate_issue_id, sanitise_name, initialise_task, close_task, log_final_metrics

REPORTS_DIR = Path("outputs") / "eval_reports"
DIAGRAMS_DIR = Path("outputs") / "diagrams"
ACCEPTED_DATA_TYPES = ("validation", "test")


def parse_args(argv=None):
    """Read the command-line arguments (--checkpoint, --split, --responsible, --run-type, --issue-id) and return them."""

    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on the validation or test set.")
    parser.add_argument("--checkpoint", required=True, help="Path to the .pt checkpoint to evaluate.")
    parser.add_argument("--split", choices=ACCEPTED_DATA_TYPES, default="validation", help="Which split to evaluate on (default: validation).")
    parser.add_argument("--responsible", required=True, choices=NAMES, help="Name who is responsible for / triggered the run.")
    parser.add_argument("--run-type", choices=RUN_TYPES, default="debugging", help="Describe the purpose of the run.")
    parser.add_argument("--issue-id", default="", help="Pin issue identifier in format 'SGP-XXX'", type=validate_issue_id)
    return parser.parse_args(argv)


def main(checkpoint_path: str|Path, data_type: str, responsible, run_type, issue_id):
    """
    Evaluate a saved checkpoint on the validation or test set.

    Computes metrics, saves the training-curve and confusion-matrix plots,
    and writes a text report to the eval reports directory.
    """
    if data_type not in ACCEPTED_DATA_TYPES:
        raise ValueError(f"data_type should be one of {ACCEPTED_DATA_TYPES}")

    # load checkpoint info
    model, _, history_metrics, config = load_checkpoint(checkpoint_path)
    
    checkpoint_name = Path(checkpoint_path).stem

    run_config = {
        "evaluation": {"checkpoint": str(checkpoint_path), "split": data_type},
        "training": config,
    }

    tags=[
        f"@run-type:{run_type}",
        f"@model:{config['model']['type']}",
        f"@responsible:{sanitise_name(responsible)}",
        "@task-type:evaluate.py",
        f"@split:{data_type}"
    ]

    if issue_id:
        tags.append(f"@issue:{issue_id}")

    task = initialise_task(
        config=run_config, 
        project_name="Plant-Disease-Image-Classification", 
        task_name=f"{config['model']['type']} - {checkpoint_name} - {data_type}", 
        tags=tags,
    )
    
    logger = task.get_logger()

    # load data to evaluate against
    _, val_loader, test_loader, class_names = get_loaders(
            fraction=config["data"]["fraction"],
            image_size=config["data"]["image_size"],
            batch_size=config["data"]["batch_size"],
            seed=config["seed"],
            normalize=config["data"].get("normalize", False),
        )
    loaders = {"validation": val_loader, "test": test_loader}
    target_data = loaders[data_type]

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

    log_final_metrics(logger, overall_metrics)

    # plot and save diagrams
    run_name = f"{checkpoint_name}_{data_type}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    curve_path = DIAGRAMS_DIR / f"{run_name}_training_curve.png"
    matrix_path = DIAGRAMS_DIR / f"{run_name}_confusion_matrix.png"

    plot_and_save_training_curve(history_metrics, curve_path)
    plot_and_save_confusion_matrix(overall_metrics["confusion-matrix"], class_names, matrix_path)

    # save written report
    report_path = REPORTS_DIR / f"{run_name}_report.txt"
    report = (
            f"---   Checkpoint: {checkpoint_path} | Data Set Chosen: {data_type.upper()}   ---\n"
            f"- Top-1 Accuracy: {overall_metrics['accuracy']}\n"
            f"- Top-3 Accuracy: {overall_metrics['top-3-accuracy']}\n"
            f"- Top-5 Accuracy: {overall_metrics['top-5-accuracy']}\n"
            f"- Macro Precision: {overall_metrics['precision']}\n"
            f"- Macro Recall: {overall_metrics['recall']}\n"
            f"- Macro F1: {overall_metrics['f1']}\n"
            f"\n"
            f" Per-Class Metrics:\n"
            f"{overall_metrics['per-class-metrics'][1]}"    # [1] is to_print version / [0] is a dict
            f"\n"
            f"==============================   Diagrams   ==============================\n"
            f" - Training Curve (Historical Metrics): {curve_path}\n"
            f" - Confusion Matrix: {matrix_path}"
        )

    save_evaluation_report(report, report_path)
    print(f"Saved evaluation report to {report_path}")

    close_task(task)


if __name__ == "__main__":
    args = parse_args()
    main(args.checkpoint, args.split, args.responsible, args.run_type, args.issue_id)