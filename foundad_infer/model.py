from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .components import (
    FactorizedLocalRefinement,
    FeaturePredictor,
    FrequencyContext,
    SaliencyProposals,
    SpectralMapper,
)


@dataclass(frozen=True)
class DetectorConfig:
    image_size: int = 512
    feature_layer: int = 3
    proposal_count: int = 8
    patch_size: int = 48
    refinement_weight: float = 0.55
    background_quantile: float = 0.45
    confidence_quantile: float = 0.95
    minimum_confidence: float = 0.02
    gate_offset: float = 0.40
    gate_sharpness: float = 4.0
    gate_power: float = 1.10
    frequency_weight: float = 0.20


def _safe_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("The model checkpoint must contain a tensor dictionary")
    allowed = {"predictor", "local_refiner", "frequency_context", "frequency_branch"}
    unexpected = set(checkpoint) - allowed
    if unexpected:
        raise ValueError(f"Unexpected checkpoint sections: {sorted(unexpected)}")
    if not {"predictor", "local_refiner"}.issubset(checkpoint):
        raise ValueError("The model checkpoint is missing required tensor sections")
    return checkpoint


def load_backbone(
    repository: str | Path,
    weights: str | Path,
    device: torch.device | str,
) -> nn.Module:
    repository_path = Path(repository).expanduser().resolve()
    weights_path = Path(weights).expanduser().resolve()
    if not repository_path.is_dir():
        raise FileNotFoundError(f"Backbone repository not found: {repository_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Backbone weights not found: {weights_path}")
    repository_text = str(repository_path)
    if repository_text not in sys.path:
        sys.path.insert(0, repository_text)
    module = importlib.import_module("dinov3.hub.backbones")
    backbone = module.dinov3_vitb16(pretrained=True, weights=str(weights_path))
    backbone.requires_grad_(False)
    return backbone.to(device).eval()


class AnomalyDetector(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        local_input_channels: int = 3,
        factorized_local_state: bool = False,
        local_has_rgb_state: bool = True,
        local_has_spectral_state: bool = False,
        has_frequency_state: bool = False,
        config: DetectorConfig | None = None,
    ) -> None:
        super().__init__()
        if local_input_channels not in {3, 4}:
            raise ValueError("Local refinement expects RGB or RGB plus one spectral channel")
        self.config = config or DetectorConfig()
        self.backbone = backbone
        self.predictor = FeaturePredictor(position_embedding=False)
        self.proposal_selector = SaliencyProposals(self.config.proposal_count)
        self.local_spectrum = SpectralMapper()
        self.factorized_local_state = bool(factorized_local_state)
        if self.factorized_local_state:
            self.local_refiner = FactorizedLocalRefinement(
                use_rgb=local_has_rgb_state,
                use_spectral=local_has_spectral_state,
            )
        else:
            self.local_refiner = nn.Sequential(
                nn.Conv2d(local_input_channels, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 1, kernel_size=1),
            )
        self.frequency_context = FrequencyContext()
        self.local_uses_frequency = (
            local_has_spectral_state if self.factorized_local_state else local_input_channels == 4
        )
        self.frequency_context_enabled = bool(has_frequency_state)

    @classmethod
    def from_checkpoint(
        cls,
        backbone: nn.Module,
        checkpoint: str | Path,
        config: DetectorConfig | None = None,
    ) -> "AnomalyDetector":
        state = _safe_checkpoint(checkpoint)
        local_state = state["local_refiner"]
        first_weight = local_state.get("0.weight")
        factorized = first_weight is None
        if factorized:
            has_rgb = "rgb_stem.weight" in local_state
            has_spectral = "frequency_stem.weight" in local_state
            if not has_rgb and not has_spectral:
                raise ValueError("Local refinement state is malformed")
            local_channels = 4 if has_spectral else 3
        else:
            if not isinstance(first_weight, torch.Tensor) or first_weight.ndim != 4:
                raise ValueError("Local refinement state is malformed")
            has_rgb = True
            has_spectral = int(first_weight.shape[1]) == 4
            local_channels = int(first_weight.shape[1])
        frequency_state = state.get("frequency_context", state.get("frequency_branch"))
        model = cls(
            backbone=backbone,
            local_input_channels=local_channels,
            factorized_local_state=factorized,
            local_has_rgb_state=has_rgb,
            local_has_spectral_state=has_spectral,
            has_frequency_state=frequency_state is not None,
            config=config,
        )
        model.predictor.load_state_dict(state["predictor"], strict=True)
        model.local_refiner.load_state_dict(state["local_refiner"], strict=True)
        if model.frequency_context_enabled:
            model.frequency_context.load_state_dict(frequency_state, strict=True)
        return model.eval()

    def _features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.get_intermediate_layers(
            images,
            n=self.config.feature_layer,
            return_class_token=False,
        )[0]
        return features

    @staticmethod
    def _square_grid(token_count: int) -> int:
        side = int(math.sqrt(token_count))
        if side * side != token_count:
            raise ValueError(f"Expected a square token grid, received {token_count} tokens")
        return side

    def _coarse(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        features = self._features(images)
        prediction = self.predictor(features)
        patch_scores = (features - prediction).square().mean(dim=2)
        side = self._square_grid(patch_scores.shape[1])
        anomaly_map = F.interpolate(
            patch_scores.reshape(images.shape[0], 1, side, side),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return patch_scores, anomaly_map, side

    def _paste(
        self,
        maps: torch.Tensor,
        proposals: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch, count = maps.shape[:2]
        canvas = maps.new_zeros((batch, 1, height, width))
        half = self.config.patch_size // 2
        for batch_index in range(batch):
            for proposal_index in range(count):
                y_value, x_value = proposals[batch_index, proposal_index]
                y_min = max(0, int(y_value) - half)
                y_max = min(height, int(y_value) + half)
                x_min = max(0, int(x_value) - half)
                x_max = min(width, int(x_value) + half)
                if y_max <= y_min or x_max <= x_min:
                    continue
                patch_map = F.interpolate(
                    maps[batch_index, proposal_index].unsqueeze(0),
                    size=(y_max - y_min, x_max - x_min),
                    mode="bilinear",
                    align_corners=False,
                )
                region = canvas[batch_index : batch_index + 1, :, y_min:y_max, x_min:x_max]
                canvas[batch_index : batch_index + 1, :, y_min:y_max, x_min:x_max] = torch.maximum(
                    region, patch_map
                )
        return canvas

    def _local_refinement(
        self,
        images: torch.Tensor,
        coarse_map: torch.Tensor,
        patch_scores: torch.Tensor,
        side: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = images.shape
        proposals, proposal_scores = self.proposal_selector(coarse_map)
        patches = self.proposal_selector.crop(images, proposals, self.config.patch_size)
        flat_patches = patches.reshape(-1, *patches.shape[2:])
        spectral_maps = self.local_spectrum(flat_patches) if self.local_uses_frequency else None
        if self.factorized_local_state:
            logits = self.local_refiner(flat_patches, spectral_maps)
        else:
            local_inputs = (
                torch.cat([flat_patches, spectral_maps], dim=1)
                if spectral_maps is not None
                else flat_patches
            )
            logits = self.local_refiner(local_inputs)
        refined = torch.sigmoid(logits).reshape(
            batch,
            proposals.shape[1],
            1,
            self.config.patch_size,
            self.config.patch_size,
        )

        coarse_flat = coarse_map.flatten(2)
        coarse_mean = coarse_flat.mean(dim=2)
        coarse_std = coarse_flat.std(dim=2).clamp_min(1e-8)
        proposal_gate = torch.sigmoid(
            self.config.gate_sharpness
            * ((proposal_scores - coarse_mean) / coarse_std - self.config.gate_offset)
        )

        refined_flat = refined.flatten(2)
        background = torch.quantile(
            refined_flat, self.config.background_quantile, dim=2, keepdim=True
        )
        sparse = (refined_flat - background).clamp_min(0.0)
        confidence = torch.quantile(
            sparse, self.config.confidence_quantile, dim=2
        )
        local_gate = torch.sigmoid(
            self.config.gate_sharpness * (confidence - self.config.minimum_confidence)
        )
        gate = (proposal_gate * local_gate).clamp(0.0, 1.0)
        if self.config.gate_power != 1.0:
            gate = gate.pow(self.config.gate_power)
        gated = sparse.reshape_as(refined) * gate.reshape(batch, -1, 1, 1, 1)
        canvas = self._paste(gated, proposals, height, width)
        local_scores = F.interpolate(
            canvas, size=(side, side), mode="bilinear", align_corners=False
        ).reshape(batch, -1)
        score_range = (
            patch_scores.max(dim=1, keepdim=True).values
            - patch_scores.min(dim=1, keepdim=True).values
        ).clamp_min(1e-8)
        return patch_scores + self.config.refinement_weight * local_scores * score_range, proposals

    def _frequency_refinement(
        self,
        images: torch.Tensor,
        patch_scores: torch.Tensor,
        side: int,
    ) -> torch.Tensor:
        if not self.frequency_context_enabled:
            return patch_scores
        logits, _ = self.frequency_context(images)
        delta = self.frequency_context.normalized_delta(logits)
        delta = F.interpolate(
            delta, size=(side, side), mode="bilinear", align_corners=False
        ).flatten(1)
        score_range = (
            patch_scores.max(dim=1, keepdim=True).values
            - patch_scores.min(dim=1, keepdim=True).values
        ).clamp_min(1e-8)
        return patch_scores + self.config.frequency_weight * delta * score_range

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected normalized images with shape [batch, 3, height, width]")
        patch_scores, coarse_map, side = self._coarse(images)
        patch_scores, proposals = self._local_refinement(
            images, coarse_map, patch_scores, side
        )
        patch_scores = self._frequency_refinement(images, patch_scores, side)
        anomaly_map = F.interpolate(
            patch_scores.reshape(images.shape[0], 1, side, side),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return {
            "patch_scores": patch_scores,
            "anomaly_map": anomaly_map,
            "proposals": proposals,
        }
