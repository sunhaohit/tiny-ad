from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)


def load_image(path: str | Path, image_size: int = 512) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - _MEAN) / _STD


def save_inference_result(
    output_directory: str | Path,
    image_path: str | Path,
    category: str,
    image_score: float,
    anomaly_map: torch.Tensor,
) -> dict[str, str | float]:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    raw_path = output_path / f"{stem}_map.npy"
    preview_path = output_path / f"{stem}_map.png"
    record_path = output_path / f"{stem}_result.json"

    array = anomaly_map.detach().float().cpu().squeeze().numpy()
    np.save(raw_path, array)
    minimum = float(array.min())
    maximum = float(array.max())
    preview = (array - minimum) / max(maximum - minimum, 1e-12)
    Image.fromarray((preview * 255.0).astype(np.uint8), mode="L").save(preview_path)

    record: dict[str, str | float] = {
        "image": Path(image_path).name,
        "category": category,
        "image_score": float(image_score),
        "anomaly_map": raw_path.name,
        "preview": preview_path.name,
    }
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
