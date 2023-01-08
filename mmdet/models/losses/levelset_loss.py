import torch
import torch.nn as nn
import torch.nn.functional as F
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



def LCM(imgs, pred_phis, box_targets):

    lcm = LocalConsistencyModule(num_iter=10, dilations=[2]).to(pred_phis.device)
    refine_phis = lcm(imgs, pred_phis)
    local_consist = (torch.abs(refine_phis - pred_phis) * box_targets).sum()
    local_regions = box_targets.sum().clamp(min=1)
    return local_consist / local_regions


class LocalConsistencyModule(nn.Module):
    """
    Local Consistency Module (LCM) for Level set phi prediction.
    """

    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = self.get_kernel()
        self.register_buffer('kernel', kernel)
        self.alpha = 0.3

    def get_kernel(self):

        kernel = torch.zeros(8, 1, 3, 3)
        kernel[0, 0, 0, 0] = 1
        kernel[1, 0, 0, 1] = 1
        kernel[2, 0, 0, 2] = 1
        kernel[3, 0, 1, 0] = 1
        kernel[4, 0, 1, 2] = 1
        kernel[5, 0, 2, 0] = 1
        kernel[6, 0, 2, 1] = 1
        kernel[7, 0, 2, 2] = 1

        return kernel

    def get_dilated_neighbors(self, x):
        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)
        return torch.cat(x_aff, dim=2)

    def forward(self, imgs, pred_phis):

        _imgs = self.get_dilated_neighbors(imgs)
        _imgs_rep = imgs.unsqueeze(2).repeat(1, 1, _imgs.shape[2], 1, 1)

        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=2, keepdim=True)

        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.alpha) ** 2
        aff = aff.mean(dim=1, keepdim=True)
        aff = F.softmax(aff, dim=2)

        for _ in range(self.num_iter):
            _pred_phis = self.get_dilated_neighbors(pred_phis)
            refine_phis = (_pred_phis * aff).sum(2)

        return refine_phis

