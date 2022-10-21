from typing import List, Tuple, Union

import torch
from torch import nn
from torchvision import ops


try:
    from math import prod
except ImportError:

    def prod(seq):
        acc = seq[0]
        for s in seq[1:]:
            acc *= s
        return acc


__all__ = [
    "MHARoIPool",
    "_ensure_tensor_rois",
    "get_encoding_1d",
    "get_encoding_2d",
    "prod",
]


@torch.jit.script
def get_encoding_1d(
    encoding_dim: int,
    size: int,
    canonical_size: float = -1.0,
    start_offset: float = 0.0,
) -> torch.Tensor:
    inv_freq = 1 / (10000 ** (torch.arange(0, encoding_dim, 2) / encoding_dim))
    if canonical_size < 0:
        canonical_size = size - 1.0  # equivalent to arange(size)
    vals = torch.linspace(start_offset, canonical_size, size).view(
        -1, 1
    ) * inv_freq.view(1, -1)
    enc = torch.cat([vals.sin(), vals.cos()], dim=-1)
    return enc


@torch.jit.script
def get_encoding_2d(
    encoding_dim: int,
    size: Tuple[int, int],
    canonical_size: Tuple[float, float] = (-1.0, -1.0),
    start_offset: Tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    per_enc_dim = encoding_dim // 2
    x_enc = get_encoding_1d(per_enc_dim, size[0], canonical_size[0], start_offset[0])
    y_enc = get_encoding_1d(per_enc_dim, size[1], canonical_size[1], start_offset[1])
    x_enc = x_enc.unsqueeze(0).expand(size[1], -1, -1)
    y_enc = y_enc.unsqueeze(1).expand(-1, size[0], -1)
    enc = torch.cat([x_enc, y_enc], dim=-1)
    return enc.permute(2, 0, 1)


def _ensure_tensor_rois(rois: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(rois, (list, tuple)) and rois[0].size(1) == 4:
        rois = torch.cat(
            [
                torch.cat(
                    [torch.full((r.size(0), 1), i, dtype=r.dtype, device=r.device), r],
                    dim=1,
                )
                for i, r in enumerate(rois)
            ],
            dim=0,
        )
    elif not (isinstance(rois, torch.Tensor) and rois.size(1) == 5):
        raise ValueError(
            "rois must be a list of tensors of shape (N, 4) or a tensor of shape (N, 5)"
        )
    return rois


class MHARoIPool(nn.Module):
    def __init__(
        self,
        output_size: Tuple[int, int],
        spatial_scale: float,
        num_features: int,
        num_heads: int = 1,
        include_position_in_value: bool = False,
    ) -> None:
        super(MHARoIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.num_features = num_features
        self.include_position_in_value = include_position_in_value
        self._num_queries = prod(output_size)
        self._content_queries = nn.Embedding(self._num_queries, num_features)
        self._position_queries = nn.parameter.Parameter(
            get_encoding_2d(num_features, output_size).flatten(1).T.contiguous()
        )
        self.canonical_size = [float(s) for s in output_size]
        self.mha = nn.MultiheadAttention(num_features, num_heads)

    def get_queries(self) -> torch.Tensor:
        return self._content_queries.weight + self._position_queries

    def forward(
        self,
        features: torch.Tensor,
        rois: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        with torch.no_grad():
            rois = _ensure_tensor_rois(rois)
            scaled_rois = rois.clone()
            scaled_rois[:, 1:] *= self.spatial_scale

        queries = self.get_queries()
        outs = []
        for roi in scaled_rois:
            width, height = (roi[-2:] - roi[-4:-2]).round().int()
            roi_features = ops.roi_align(
                features, roi.unsqueeze(0), (height, width), 1.0, -1, True
            )  # easiest way to extract offset features
            pos_enc = get_encoding_2d(
                self.num_features, (width, height), self.canonical_size
            ).to(roi_features)
            roi_features_with_pos = (
                (roi_features + pos_enc).flatten(2).permute(2, 0, 1).squeeze(1)
            )
            roi_features = roi_features.flatten(2).permute(2, 0, 1).squeeze(1)
            if self.include_position_in_value:
                attended_features, _ = self.mha(
                    queries, roi_features_with_pos, roi_features_with_pos
                )
            else:
                attended_features, _ = self.mha(
                    queries, roi_features_with_pos, roi_features
                )
            outs.append(attended_features.T)
        return torch.stack(outs, dim=0).view(len(outs), -1, *self.output_size)
