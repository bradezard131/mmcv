from math import prod
from typing import List, Tuple, Union

from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import ops

from mha_roi_pool import _ensure_tensor_rois, get_encoding_2d


__all__ = ["GpsaRoIPool"]


class GpsaRoIPool(nn.Module):
    def __init__(
        self,
        output_size: Tuple[int, int],
        spatial_scale: float,
        num_features: int,
        num_heads: int = 1,
        position_resolution_factor: int = 2,
    ) -> None:
        super(GpsaRoIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.num_features = num_features
        self.num_heads = num_heads
        self.position_resolution_factor = position_resolution_factor

        self.w_k = nn.Conv2d(num_features, num_features, 1)
        self.w_v = nn.Conv2d(num_features, num_features, 1)
        self.queries = nn.Embedding(prod(output_size), num_features)
        self._balance = nn.parameter.Parameter(torch.tensor(3.0))
        self.position_weights = nn.parameter.Parameter(
            torch.empty(
                1,
                prod(output_size),
                position_resolution_factor * output_size[0],
                position_resolution_factor * output_size[1],
            )
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.queries.reset_parameters()
        small_pos_enc = get_encoding_2d(
            self.num_features,
            self.output_size,
            [os - 0.5 for os in self.output_size],
            [0.5, 0.5],
        )
        big_pos_enc = get_encoding_2d(
            self.num_features,
            [os * self.position_resolution_factor for os in self.output_size],
            self.output_size,
        )
        energy = torch.einsum(
            "c y x, c h w -> y x h w", small_pos_enc, big_pos_enc
        ).flatten(0, 1)
        self.position_weights.data.copy_(energy.unsqueeze(0))
        self._balance.data.fill_(3.0)

    @property
    def balance(self) -> torch.Tensor:
        return F.hardsigmoid(self._balance)

    def get_position_attention(self, roi_map: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            self.position_weights.data, roi_map.shape[-2:], mode="bilinear"
        )

    def forward(
        self,
        features: torch.Tensor,
        rois: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        with torch.no_grad():
            rois = _ensure_tensor_rois(rois)
            scaled_rois = rois.clone()
            scaled_rois[:, 1:] *= self.spatial_scale

        k = self.w_k(features)
        v = self.w_v(features)
        outs = []
        for roi in scaled_rois:
            width, height = (roi[-2:] - roi[-4:-2]).round().int()
            roi_keys = ops.roi_align(
                k, roi.unsqueeze(0), (height, width), 1.0, -1, True
            )
            roi_vals = ops.roi_align(
                v, roi.unsqueeze(0), (height, width), 1.0, -1, True
            )
            cont_attn = (
                torch.einsum(
                    "n h c l, q h c -> n h q l",
                    rearrange(roi_keys, "n (h c) y x -> n h c (y x)", h=self.num_heads),
                    rearrange(
                        self.queries.weight, "q (h c) -> q h c", h=self.num_heads
                    ),
                )
                .mul_(self.num_features**-0.5)
                .softmax(-1)
                .mean(1)
            )
            pos_attn = self.get_position_attention(roi_keys).flatten(2).softmax(-1)
            attn = self.balance * pos_attn + (1 - self.balance) * cont_attn
            outs.append(
                rearrange(
                    torch.einsum("n q l, n c l -> n c q", attn, roi_vals.flatten(2)),
                    "n c (y x) -> n c y x",
                    y=self.output_size[0],
                    x=self.output_size[1],
                )
            )
        return torch.cat(outs, 0)
