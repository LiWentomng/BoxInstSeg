import torch
import torch.nn as nn
from ..builder import LOSSES

@LOSSES.register_module()
class LevelsetLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(LevelsetLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, mask_logits, targets, pixel_num):
        region_levelset_term = region_levelset()
        region_levelset_loss = region_levelset_term(mask_logits, targets) / pixel_num
        loss_levelst = self.loss_weight * region_levelset_loss

        return loss_levelst


class region_levelset(nn.Module):
    '''
    The mian of region leveset function.
    '''

    def __init__(self):
        super(region_levelset, self).__init__()

    def forward(self, mask_score, lst_target):
        '''
        mask_score: predcited mask scores        tensor:(N,C,W,H)
        lst_target:  input target for levelset   tensor:(N,C,W,H)
        '''
        mask_score_shape = mask_score.shape
        lst_target_shape = lst_target.shape
        level_set_loss = 0.0

        for i in range(lst_target_shape[1]):

            lst_target_ = torch.unsqueeze(lst_target[:, i], 1)
            lst_target_ = lst_target_.expand(lst_target_shape[0], mask_score_shape[1], lst_target_shape[2], lst_target_shape[3])
            ave_similarity = torch.sum(lst_target_ * mask_score, (2, 3)) / (torch.sum(mask_score, (2, 3))).clamp(min=0.00001)
            ave_similarity = ave_similarity.view(lst_target_shape[0], mask_score_shape[1], 1, 1)

            region_level = lst_target_ - ave_similarity.expand(lst_target_shape[0], mask_score_shape[1],
                                                                               lst_target_shape[2], lst_target_shape[3])

            region_level_loss = region_level * region_level * mask_score
            level_set_loss += torch.sum(region_level_loss, dim=(1,2,3))

        return level_set_loss / lst_target_shape[1]


class length_regularization(nn.Module):

    '''
    calcaulate the length by the gradient for regularization.
    '''

    def __init__(self):
        super(length_regularization, self).__init__()

    def forward(self, mask_score):
        gradient_H = torch.abs(mask_score[:, :, 1:, :] - mask_score[:, :, :-1, :])
        gradient_W = torch.abs(mask_score[:, :, :, 1:] - mask_score[:, :, :, :-1])
        curve_length = torch.sum(gradient_H, dim=(1, 2, 3)) + torch.sum(gradient_W, dim=(1, 2, 3))
        return curve_length

