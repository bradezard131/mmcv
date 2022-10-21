from typing import List, Tuple, Union

from einops import rearrange
from mmcv.ops import DeformRoIPool
import torch
from torch import nn

from mha_roi_pool import _ensure_tensor_rois, prod


__all__ = ["DeformAttnRoIPool"]


class DeformAttnRoIPool(nn.Module):
    def __init__(
        self,
        output_size: Tuple[int, int],
        spatial_scale: float,
        num_features: int,
        num_points: int = 4,
        num_heads: int = 1,
    ) -> None:
        super(DeformAttnRoIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.num_features = num_features
        self.num_heads = num_heads
        self.num_points = num_points
        self.per_point_offsets = nn.parameter.Parameter(
            torch.randn(num_points, *output_size, 2) * 0.001
        )
        self.queries = nn.Embedding(prod(output_size), num_features)
        self.deform_pool = DeformRoIPool(output_size, spatial_scale, 1)
        self.proj = nn.Conv2d(num_features, num_features, 1)
        self.mha = nn.MultiheadAttention(num_features, num_heads)

    def forward(
        self,
        features: torch.Tensor,
        rois: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        rois = _ensure_tensor_rois(rois)

        all_point_samples = torch.stack(
            [
                self.deform_pool(
                    features,
                    rois,
                    point_offset.unsqueeze(0).expand(rois.size(0), -1, -1, -1),
                )
                for point_offset in self.per_point_offsets
            ],
            dim=0,
        )
        samples = self.proj(all_point_samples.flatten(0, 1)).view(
            all_point_samples.shape
        )

        # samples = self.proj(
        #     torch.stack(
        #         [
        #             self.deform_pool(
        #                 features,
        #                 rois,
        #                 point_offset.unsqueeze(0).expand(rois.size(0), -1, -1, -1),
        #             )
        #             for point_offset in self.per_point_offsets
        #         ],
        #         dim=0,
        #     )  # (N, num_points, C, H, W)
        # )
        p, n, c, h, w = samples.size()
        samples = rearrange(samples, "p n c h w -> p (n h w) c")
        results, _ = self.mha(
            self.queries.weight.unsqueeze(0).repeat(1, n, 1), samples, samples
        )
        return rearrange(results, "p (n h w) c -> n (p c) h w", n=n, h=h, w=w)  # p = 1
