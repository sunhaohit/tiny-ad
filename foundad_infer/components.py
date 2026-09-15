from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        self.num_heads = int(heads)
        self.scale = (dimension // self.num_heads) ** -0.5
        self.qkv = nn.Linear(dimension, dimension * 3, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dimension, dimension, bias=True)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = values.shape
        qkv = self.qkv(values).reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0] * self.scale, qkv[1], qkv[2]
        weights = (query @ key.transpose(-2, -1)).softmax(dim=-1)
        weights = self.attn_drop(weights)
        output = (weights @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(output))


class FeedForward(nn.Module):
    def __init__(self, dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dimension, hidden_dimension, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dimension, dimension, bias=True)
        self.drop = nn.Dropout(0.0)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.drop(self.act(self.fc1(values)))
        return self.drop(self.fc2(values))


class PredictorBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, expansion: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension, eps=1e-6)
        self.attn = Attention(dimension, heads)
        self.norm2 = nn.LayerNorm(dimension, eps=1e-6)
        self.mlp = FeedForward(dimension, int(dimension * expansion))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values + self.attn(self.norm1(values))
        return values + self.mlp(self.norm2(values))


class FeaturePredictor(nn.Module):
    """Transformer predictor with checkpoint-compatible parameter names."""

    def __init__(
        self,
        input_dimension: int = 768,
        predictor_dimension: int = 384,
        depth: int = 6,
        heads: int = 12,
        token_count: int = 1024,
        position_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.predictor_embed = nn.Linear(input_dimension, predictor_dimension, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dimension))
        self.if_pe = bool(position_embedding)
        if self.if_pe:
            self.predictor_pos_embed = nn.Parameter(
                torch.zeros(1, token_count, predictor_dimension),
                requires_grad=False,
            )
        else:
            self.register_parameter("predictor_pos_embed", None)
        self.predictor_blocks = nn.ModuleList(
            [PredictorBlock(predictor_dimension, heads) for _ in range(depth)]
        )
        self.predictor_norm = nn.LayerNorm(predictor_dimension, eps=1e-6)
        self.predictor_proj = nn.Linear(predictor_dimension, input_dimension, bias=True)
        self.feat_normed = False

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.predictor_embed(values)
        if self.predictor_pos_embed is not None:
            if values.shape[1] != self.predictor_pos_embed.shape[1]:
                raise ValueError("The checkpoint position embedding does not match the token grid")
            values = values + self.predictor_pos_embed
        residual = values.clone()
        for block in self.predictor_blocks:
            values = block(values) + residual
        values = self.predictor_norm(values)
        return self.predictor_proj(values)


class SaliencyProposals(nn.Module):
    def __init__(self, count: int = 8, minimum_distance: float = 3.0) -> None:
        super().__init__()
        self.count = int(count)
        self.minimum_distance = float(minimum_distance)

    def forward(self, anomaly_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = anomaly_map.shape
        candidate_count = min(self.count * 3, height * width)
        scores, indices = torch.topk(anomaly_map.flatten(1), candidate_count, dim=1)
        coordinates = torch.stack(
            [(indices // width).float(), (indices % width).float()], dim=-1
        )
        proposals = anomaly_map.new_zeros((batch, self.count, 2))
        selected_scores = anomaly_map.new_zeros((batch, self.count))

        for batch_index in range(batch):
            selected: list[tuple[float, float]] = []
            for candidate_index in torch.argsort(scores[batch_index], descending=True):
                y_value, x_value = coordinates[batch_index, candidate_index]
                point = (float(y_value), float(x_value))
                if all(
                    math.hypot(point[0] - other[0], point[1] - other[1])
                    >= self.minimum_distance
                    for other in selected
                ):
                    selected.append(point)
                if len(selected) == self.count:
                    break

            if not selected:
                selected.append((0.0, 0.0))
            while len(selected) < self.count:
                selected.append(selected[-1])
            for proposal_index, (y_value, x_value) in enumerate(selected[: self.count]):
                proposals[batch_index, proposal_index] = anomaly_map.new_tensor(
                    [y_value, x_value]
                )
                selected_scores[batch_index, proposal_index] = anomaly_map[
                    batch_index, 0, int(y_value), int(x_value)
                ]
        return proposals, selected_scores

    @staticmethod
    def crop(
        images: torch.Tensor,
        proposals: torch.Tensor,
        patch_size: int,
    ) -> torch.Tensor:
        batch, _, height, width = images.shape
        half = patch_size // 2
        all_patches: list[torch.Tensor] = []
        for batch_index in range(batch):
            image_patches: list[torch.Tensor] = []
            for y_value, x_value in proposals[batch_index]:
                y_min = max(0, int(y_value) - half)
                y_max = min(height, int(y_value) + half)
                x_min = max(0, int(x_value) - half)
                x_max = min(width, int(x_value) + half)
                patch = images[batch_index, :, y_min:y_max, x_min:x_max]
                patch = F.interpolate(
                    patch.unsqueeze(0),
                    size=(patch_size, patch_size),
                    mode="bilinear",
                    align_corners=False,
                )
                image_patches.append(patch)
            all_patches.append(torch.cat(image_patches, dim=0))
        return torch.stack(all_patches, dim=0)


class SpectralMapper(nn.Module):
    """Deterministic high-pass Fourier map used by local and global paths."""

    def __init__(
        self,
        low_radius: float = 0.08,
        low_quantile: float = 0.50,
        high_quantile: float = 0.96,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.low_radius = max(float(low_radius), 0.0)
        self.low_quantile = min(max(float(low_quantile), 0.0), 0.95)
        self.high_quantile = min(max(float(high_quantile), self.low_quantile + 1e-3), 0.999)
        self.gamma = max(float(gamma), 1e-6)

    @staticmethod
    def _radial_mask(height: int, width: int, radius: float, device: torch.device) -> torch.Tensor:
        y_frequency = torch.fft.fftshift(torch.fft.fftfreq(height, device=device))
        x_frequency = torch.fft.fftshift(torch.fft.fftfreq(width, device=device))
        y_grid, x_grid = torch.meshgrid(y_frequency, x_frequency, indexing="ij")
        return (
            torch.sqrt(x_grid.square() + y_grid.square()) >= radius
        ).float().reshape(1, 1, height, width)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=images.device.type, enabled=False):
            gray = images.float().mean(dim=1, keepdim=True)
            gray = gray - gray.mean(dim=(-2, -1), keepdim=True)
            spectrum = torch.fft.fftshift(
                torch.fft.fft2(gray, dim=(-2, -1), norm="ortho"), dim=(-2, -1)
            )
            mask = self._radial_mask(
                gray.shape[-2], gray.shape[-1], self.low_radius, gray.device
            )
            residual = torch.fft.ifft2(
                torch.fft.ifftshift(spectrum * mask, dim=(-2, -1)),
                dim=(-2, -1),
                norm="ortho",
            ).real.abs()
            flat = residual.flatten(1)
            low = torch.quantile(flat, self.low_quantile, dim=1).reshape(-1, 1, 1, 1)
            high = torch.quantile(flat, self.high_quantile, dim=1).reshape(-1, 1, 1, 1)
            spread = high - low
            quantile_map = ((residual - low) / spread.clamp_min(1e-6)).clamp(0.0, 1.0)
            peak_map = residual / residual.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            output = torch.where(spread > 1e-6, quantile_map, peak_map)
            if self.gamma != 1.0:
                output = output.clamp_min(1e-6).pow(self.gamma)
        return output.to(dtype=images.dtype)


class FactorizedLocalRefinement(nn.Module):
    """Local refinement with independent RGB and spectral input stems."""

    def __init__(
        self,
        use_rgb: bool = True,
        use_spectral: bool = True,
        hidden_channels: int = 256,
        bottleneck_channels: int = 128,
    ) -> None:
        super().__init__()
        if not use_rgb and not use_spectral:
            raise ValueError("At least one local input must be enabled")
        self.rgb_stem = (
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1, bias=False)
            if use_rgb
            else None
        )
        self.frequency_stem = (
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1, bias=False)
            if use_spectral
            else None
        )
        self.stem_bias = nn.Parameter(torch.zeros(hidden_channels))
        self.body = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, 1, kernel_size=1),
        )

    def forward(
        self,
        rgb_patches: torch.Tensor,
        spectral_maps: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden = self.rgb_stem(rgb_patches) if self.rgb_stem is not None else None
        if self.frequency_stem is not None:
            if spectral_maps is None:
                raise ValueError("The spectral input required by this checkpoint is missing")
            spectral_hidden = self.frequency_stem(spectral_maps)
            hidden = spectral_hidden if hidden is None else hidden + spectral_hidden
        if hidden is None:
            raise RuntimeError("No local input path is available")
        hidden = hidden + self.stem_bias.reshape(1, -1, 1, 1)
        return self.body(hidden)


class FrequencyContext(nn.Module):
    def __init__(self, hidden_channels: int = 16) -> None:
        super().__init__()
        self.mapper = SpectralMapper(high_quantile=0.98)
        self.head = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spectral_map = self.mapper(images)
        return self.head(spectral_map), spectral_map

    @staticmethod
    def normalized_delta(logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits.float())
        center = torch.quantile(probabilities.flatten(1), 0.50, dim=1).reshape(-1, 1, 1, 1)
        delta = (probabilities - center).clamp_min(0.0)
        scale = torch.quantile(delta.flatten(1), 0.95, dim=1).reshape(-1, 1, 1, 1)
        return (delta / scale.clamp_min(1e-6)).clamp(0.0, 1.0)
