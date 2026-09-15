# FoundAD Inference

This repository contains only the forward-pass implementation and a sanitized
model checkpoint. It does not include optimization code, experiment runners,
evaluation utilities, datasets, logs, or local environment settings.

## Requirements

- Python 3.10 or newer
- PyTorch 2.1 or newer
- A local checkout of the official DINOv3 repository
- The official DINOv3 ViT-B/16 pretrained weights

Install the package in a virtual environment:

```bash
python -m pip install -e .
```

The DINOv3 backbone is not redistributed here. Obtain its repository and
pretrained weights from the official project and pass both paths explicitly.

## Run inference

```bash
foundad-infer image.png \
  --category class_name \
  --backbone-repo /path/to/dinov3 \
  --backbone-weights /path/to/backbone_weights.pth \
  --model-weights weights/model_weights.pt \
  --output outputs
```

Each image produces a JSON record, a NumPy anomaly map, and a normalized PNG
visualization. The JSON file contains the image-level anomaly score. The NumPy
file contains the unnormalized pixel-level map.

`--category` selects the fixed image-score calibration used by the released
checkpoint. It does not change the neural-network weights.

## Python API

```python
import torch

from foundad_infer.io import load_image
from foundad_infer.model import AnomalyDetector, load_backbone
from foundad_infer.scoring import aggregate_image_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = load_backbone(
    repository="/path/to/dinov3",
    weights="/path/to/backbone_weights.pth",
    device=device,
)
model = AnomalyDetector.from_checkpoint(
    backbone=backbone,
    checkpoint="weights/model_weights.pt",
).to(device).eval()

image = load_image("image.png").unsqueeze(0).to(device)
with torch.inference_mode():
    result = model(image)
score = aggregate_image_score(result["patch_scores"], category="class_name")
```

## Checkpoint contents

The checkpoint stores tensor state only. Run names, machine paths, user names,
optimizer state, schedules, iteration counters, and reported metrics have been
removed.

