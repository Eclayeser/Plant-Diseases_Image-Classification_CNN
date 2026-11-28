import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
import numpy as np
import onnx
import onnxruntime

from leafapp.checkpoints import load_checkpoint
from leafapp.config import (
    MODEL_DIR,
    MODEL_CONFIG_FILE,
    MODEL_FILE,
    LABELS_FILE,
)
from leafapp.data import get_normalization_stats
from leafapp.tracking import (
    DEFAULT_ARTIFACT,
    get_task,
    download_checkpoint,
    upload_onnx_model_to_clearml,
    build_tags,
    close_task,
    initialise_task,
    validate_issue_id,
    NAMES,
    PROJECT_NAME,
    RUN_TYPES,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export a trained CNN to ONNX.")
    parser.add_argument(
        "--task-id",
        required=True,
        help=f"ClearML training task ID whose {DEFAULT_ARTIFACT} is exported.",
    )
    parser.add_argument(
        "--out",
        default=MODEL_DIR,
        help="Folder to write the ONNX model, labels and config to.",
    )

    parser.add_argument("--upload", action="store_true", help="Upload the exported files to the ClearML Models tab as one published model.")
    parser.add_argument("--responsible", choices=NAMES, help="Name who is responsible for / triggered the run. Required with --upload.")
    parser.add_argument("--run-type", choices=RUN_TYPES, default="debugging", help="Describe the purpose of the run.")
    parser.add_argument("--issue-id", default="", help="Pin issue identifier in format 'SGP-XXX'", type=validate_issue_id)

    args = parser.parse_args(argv)

    if args.upload and args.responsible is None:
        parser.error("--upload requires --responsible")

    return args


def build_export_model(model, normalize):
    """Return the model to export: normalisation is prepended, so the graph takes 0-1 RGB pixels."""
    stats = get_normalization_stats(normalize)
    if stats is not None:
        mean, std = stats
        model = nn.Sequential(transforms.Normalize(mean, std), model)

    model.eval()
    return model


def check_parity(torch_model, onnx_path, sample, atol=1e-4):
    """Raise ValueError if ONNX Runtime's output for sample differs from PyTorch's."""
    with torch.no_grad():
        expected = torch_model(sample).numpy()
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"image": sample.numpy()})[0]
    if not np.allclose(expected, actual, atol=atol):
        raise ValueError(f"ONNX output differs from PyTorch by {np.abs(expected - actual).max():.2e}")


def upload_export(training_task, config, files, responsible, run_type, issue_id, model_config):
    """Create an export task in ClearML, upload files as its published model and return the model ID."""
    name = f"{training_task.name} - {training_task.id[:6]} - onnx"
    tags = build_tags(
        run_type=run_type,
        model_type=config["model"]["type"],
        responsible=responsible,
        task_type="export_onnx.py",
        issue_id=issue_id
    )
    task, _ = initialise_task(
        config={
            "export": {"source_task_id": training_task.id, "source_artifact": DEFAULT_ARTIFACT},
            "training": config,
        },
        project_name=PROJECT_NAME,
        task_name=name,
        tags=tags,
        output_uri=True
    )
    model_id = upload_onnx_model_to_clearml(
        task=task,
        files=files,
        name=name,
        tags=tags,
        config=model_config,
        comment=f"ONNX export of {training_task.name} (task {training_task.id}, {DEFAULT_ARTIFACT})",
    )
    close_task(task)
    return model_id


def main(task_id, out_dir=MODEL_DIR, upload=False, responsible=None, run_type=None, issue_id=""):
    training_task = get_task(task_id)
    checkpoint_path = download_checkpoint(training_task, DEFAULT_ARTIFACT)

    model, class_names, _, config = load_checkpoint(checkpoint_path)
    image_size = config["data"]["image_size"]
    normalize = config["data"].get("normalize", False)

    export_model = build_export_model(model, normalize)
    sample = torch.rand(1, 3, image_size, image_size)

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    onnx_path = out_dir / MODEL_FILE

    torch.onnx.export(
        export_model,
        sample,
        str(onnx_path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(str(onnx_path))
    check_parity(export_model, onnx_path, sample)

    with open(out_dir / LABELS_FILE, "w") as f:
        json.dump(class_names, f, indent=2)

    model_config = {
        "image_size": image_size,
        "normalization_in_graph": normalize,
        "source_task_id": task_id,
        "source_artifact": DEFAULT_ARTIFACT,
    }
    with open(out_dir / MODEL_CONFIG_FILE, "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"ONNX export completed from task {task_id} ({DEFAULT_ARTIFACT}) to {out_dir}")

    if upload:
        files = [onnx_path, out_dir / LABELS_FILE, out_dir / MODEL_CONFIG_FILE]
        model_id = upload_export(
            training_task=training_task,
            config=config,
            files=files,
            responsible=responsible,
            run_type=run_type,
            issue_id=issue_id,
            model_config=model_config
        )
        print(f"Uploaded and published ClearML model {model_id}")


if __name__ == "__main__":
    args = parse_args()
    main(args.task_id, args.out, args.upload, args.responsible, args.run_type, args.issue_id)

    