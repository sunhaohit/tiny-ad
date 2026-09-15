# Tiny-AD Inference

This folder contains the paper-aligned forward pass and a sanitized checkpoint.
It contains no optimization code, experiment runners, evaluation framework,
dataset copy, logs, or local environment settings.

The inference path follows the paper description:

1. a frozen DINOv3 ViT-B/16 encoder extracts patch tokens from a 512 x 512 image;
2. the predictor reconstructs the tokens and their squared error forms a coarse map;
3. saliency-guided proposals select high-response image regions;
4. each 48 x 48 crop is refined at its original resolution;
5. when the checkpoint contains a spectral input stem, a locally normalized FFT
   map is supplied as the fourth crop channel;
6. sparse confidence gating fuses the local response into the coarse anomaly map.

Frequency processing is local to proposal crops. No whole-image frequency
filter or late global frequency branch is present.

## Requirements

- Python
- PyTorch
- NumPy
- Pillow
- A local checkout of the official DINOv3 repository
- The official DINOv3 ViT-B/16 pretrained weights

Install the package in a virtual environment:

```bash
python -m pip install -r requirements.txt
```

The DINOv3 backbone is not redistributed here. Obtain its repository and
pretrained weights from the official project and pass both paths explicitly.

## Run inference

```bash
python -m tinyad_infer image.png \
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

from tinyad_infer.io import load_image
from tinyad_infer.model import AnomalyDetector, load_backbone
from tinyad_infer.scoring import aggregate_image_score

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
removed. The loader infers the available local inputs from tensor shapes and
never activates a missing branch with randomly initialized parameters.
