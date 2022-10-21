from typing import List, Tuple, Union

import torch
from torch import nn
from torchvision import ops

from .mha_roi_pool import _ensure_tensor_rois, prod


class Gpsa2RoIPool(nn.Module):
    def __init__(
        self,
        output_size: Tuple[int, int],
        spatial_scale: float,
        num_features: int,
        num_heads: int = 1,
    ) -> None:
        super(Gpsa2RoIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.num_features = num_features
        self.num_heads = num_heads

        self._num_queries = prod(output_size)
        self._content_queries = nn.Embedding(self._num_queries, num_features)
        self.mha = nn.MultiheadAttention(num_features, num_heads)
        self.position_pool = ops.RoIAlign(output_size, spatial_scale, -1, True)
        self._balance = nn.parameter.Parameter(torch.tensor(3.0))

    @property
    def balance(self) -> torch.Tensor:
        return F.hardsigmoid(self._balance)

    def forward(
        self,
        features: torch.Tensor,
        rois: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        with torch.no_grad():
            rois = _ensure_tensor_rois(rois)
            scaled_rois = rois.clone()
            scaled_rois[:, 1:] *= self.spatial_scale

        balance = self.balance
        position_features = self.position_pool(features, rois) * balance
        inv_balance = 1 - self.balance
        for roi, position_feature in zip(scaled_rois, position_features):
            width, height = (roi[-2:] - roi[-4:-2]).round().int()
            roi_content_features = (
                ops.roi_align(
                    features, roi.unsqueeze(0), (height, width), 1.0, -1, True
                )
                .flatten(2)
                .permute(2, 0, 1)
            )
            attended_features, _ = self.mha(
                self._content_queries.weight.unsqueeze(1),
                roi_content_features,
                roi_content_features,
            )
            position_feature.add_(
                attended_features.squeeze(1).T.reshape(position_feature.shape)
                * inv_balance
            )
        return position_features
