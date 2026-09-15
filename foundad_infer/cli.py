from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .io import load_image, save_inference_result
from .model import AnomalyDetector, load_backbone
from .scoring import aggregate_image_score


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run anomaly inference on one or more images")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--category", required=True)
    parser.add_argument("--backbone-repo", required=True, type=Path)
    parser.add_argument("--backbone-weights", required=True, type=Path)
    parser.add_argument("--model-weights", default=Path("weights/model_weights.pt"), type=Path)
    parser.add_argument("--output", default=Path("outputs"), type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    device = torch.device(arguments.device)
    backbone = load_backbone(arguments.backbone_repo, arguments.backbone_weights, device)
    model = AnomalyDetector.from_checkpoint(
        backbone=backbone,
        checkpoint=arguments.model_weights,
    ).to(device).eval()

    for image_path in arguments.images:
        image = load_image(image_path, model.config.image_size).unsqueeze(0).to(device)
        result = model(image)
        score = aggregate_image_score(result["patch_scores"], arguments.category)[0]
        record = save_inference_result(
            arguments.output,
            image_path,
            arguments.category,
            float(score),
            result["anomaly_map"][0],
        )
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()

