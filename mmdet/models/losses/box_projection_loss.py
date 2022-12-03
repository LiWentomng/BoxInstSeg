import torch
import torch.nn as nn
from ..builder import LOSSES

@LOSSES.register_module()
class BoxProjectionLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(BoxProjectionLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, mask_scores, box_bitmask):

        projection_loss = self.compute_project_term(mask_scores, box_bitmask)
        loss_project_loss = self.loss_weight * projection_loss

        return loss_project_loss

    def compute_project_term(self, mask_scores, gt_bitmasks):
        """
        box projection function
        """
        mask_losses_y = self.dice_coefficient(
            mask_scores.max(dim=2, keepdim=True)[0],
            gt_bitmasks.max(dim=2, keepdim=True)[0]
        )
        mask_losses_x = self.dice_coefficient(
            mask_scores.max(dim=3, keepdim=True)[0],
            gt_bitmasks.max(dim=3, keepdim=True)[0]
        )
        return (mask_losses_x + mask_losses_y)


    def dice_coefficient(self, x, target):

        eps = 1e-5
        n_inst = x.size(0)
        x = x.reshape(n_inst, -1)
        target = target.reshape(n_inst, -1)
        intersection = (x * target).sum(dim=1)
        union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
        loss = 1. - (2 * intersection / union)
        return loss

