"""Inference-only Tiny-AD package."""

from .model import AnomalyDetector, DetectorConfig, load_backbone
from .scoring import aggregate_image_score

__all__ = [
    "AnomalyDetector",
    "DetectorConfig",
    "aggregate_image_score",
    "load_backbone",
]
