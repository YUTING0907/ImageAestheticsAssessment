from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankLoss(nn.Module):

    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        y_pred: Tuple[torch.Tensor, torch.Tensor],
        y_true: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:

        device, dtype = y_pred[0].device, y_pred[0].dtype

        target = torch.ones_like(y_true[0]).to(device).to(dtype)

        # Set indices where y_true1 < y_true2 to -1
        target[y_true[0] < y_true[1]] = -1.0

        return F.margin_ranking_loss(
            y_pred[0],
            y_pred[1],
            target,
            margin=self.margin
        )


class RegRankLoss(nn.Module):

    def __init__(self, margin: float):
        super().__init__()
        self.reg_loss = nn.MSELoss(reduction="mean")
        self.rank_loss = RankLoss(margin)

    def forward(
        self,
        y_pred: Tuple[torch.Tensor, torch.Tensor],
        y_true: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        loss_reg = (
            self.reg_loss(y_pred[0], y_true[0]) +
            self.reg_loss(y_pred[1], y_true[1])
        ) / 2.0

        loss_rank = self.rank_loss(y_pred, y_true)
        loss = loss_reg + loss_rank
        return loss, loss_reg, loss_rank


class EDMLoss(nn.Module):
    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self, y_true, y_pred):
        #assert y_true.shape == y_pred.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(y_true, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(y_pred, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
        return samplewise_emd.mean()
