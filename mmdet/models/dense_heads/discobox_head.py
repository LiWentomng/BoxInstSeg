import mmcv
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import multi_apply, InstanceData, mask_matrix_nms
from mmcv.ops.roi_align import RoIAlign
from mmcv import tensor2imgs
from mmcv.runner import BaseModule
from mmcv.runner.fp16_utils import force_fp32
from ..builder import build_loss, HEADS
from torch.cuda.amp import autocast
from mmcv.cnn import ConvModule
import numpy as np

def relu_and_l2_norm_feat(feat, dim=1):
    feat = F.relu(feat, inplace=True)
    feat_norm = ((feat ** 2).sum(dim=dim, keepdim=True) + 1e-6) ** 0.5
    feat = feat / (feat_norm + 1e-6)
    return feat


class ObjectFactory:

    @staticmethod
    def create_one(mask, feature, box, img, category):
        if img is not None:
            img_size = img.shape[2]
        else:
            img_size = 0
        object_elements = ObjectElements(size=1,
                                         img_size=img_size,
                                         feat_size=feature.shape[2],
                                         mask_size=mask.shape[1],
                                         n_channel=feature.shape[1],
                                         device=mask.device,
                                         category=category)
        object_elements.mask[...] = mask
        object_elements.feature[...] = feature
        object_elements.feature[...] = relu_and_l2_norm_feat(object_elements.feature[0:1])
        object_elements.box[...] = box
        if img is not None:
            object_elements.img[...] = img
        return object_elements

    @staticmethod
    def create_queue_by_one(len_queue, category, idx, feature, mask, box, img=None, device='cpu'):
        if img is not None:
            img_size = img.shape[2]
        else:
            img_size = 0
        if category == 1:
            device = mask.device
        object_elements = ObjectElements(size=len_queue,
                                         img_size=img_size,
                                         feat_size=feature.shape[2],
                                         mask_size=mask.shape[1],
                                         n_channel=feature.shape[1],
                                         device=device,
                                         category=category)
        object_elements.mask[0:1] = mask[idx:idx + 1]
        object_elements.feature[0:1] = feature[idx:idx + 1]
        object_elements.box[0:1] = box[idx:idx + 1]
        if img is not None:
            object_elements.img[0:1] = img[idx:idx + 1]
        return object_elements


class ObjectElements:

    def __init__(self, size=100, img_size=56, feat_size=28, mask_size=56, n_channel=256, device='cpu', category=None):
        self.mask = torch.zeros(size, mask_size, mask_size).to(device).to(device)
        self.feature = torch.zeros(size, n_channel, feat_size, feat_size).to(device)
        self.box = torch.zeros(size, 4).to(device)
        self.img = torch.zeros(size, 3, img_size, img_size).to(device)
        self.category = int(category)
        self.ptr = 0

    def get_box_area(self):
        box = self.box
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def get_category(self):
        return self.category

    def get_feature(self):
        return self.feature

    def get_mask(self):
        return self.mask

    def get_ratio(self):
        box = self.box
        return (box[:, 2] - box[:, 0]) / (box[:, 3] - box[:, 1] + 1e-5)

    def get_img(self):
        return self.img

    def get_box(self):
        return self.box

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        if isinstance(idx, slice) or torch.is_tensor(idx) or isinstance(idx, list):
            if torch.is_tensor(idx):
                idx = idx.to(self.mask).long()  # self.mask might be in cpu
            if self.img is not None:
                img = self.img[idx]
            else:
                img = None
            mask = self.mask[idx]
            feature = self.feature[idx]
            box = self.box[idx]
            category = self.category
        elif isinstance(idx, int):
            if self.img is not None:
                img = self.img[idx:idx + 1]
            else:
                img = None
            mask = self.mask[idx:idx + 1]
            feature = self.feature[idx:idx + 1]
            box = self.box[idx:idx + 1]
            category = self.category
        else:
            raise NotImplementedError("type: {}".format(type(idx)))
        return dict(img=img, mask=mask, feature=feature, box=box, category=category)


class ObjectQueues:

    def __init__(self, num_class, len_queue, fg_iou_thresh, bg_iou_thresh, ratio_range, appear_thresh,
                 max_retrieval_objs):
        self.num_class = num_class
        self.queues = [None for i in range(self.num_class)]
        self.len_queue = len_queue
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.appear_thresh = appear_thresh
        self.ratio_range = ratio_range
        self.max_retrieval_objs = max_retrieval_objs

    def append(self, class_idx, idx, feature, mask, box, img=None, device='cpu'):
        with torch.no_grad():
            if self.queues[class_idx] is None:
                self.queues[class_idx] = \
                    ObjectFactory.create_queue_by_one(
                        len_queue=self.len_queue,
                        category=class_idx,
                        idx=idx,
                        feature=feature,
                        mask=mask,
                        box=box,
                        img=img,
                        device=device
                    )
                create_new_gpu_bank = True
                self.queues[class_idx].ptr += 1
                self.queues[class_idx].ptr = self.queues[class_idx].ptr % self.len_queue
            else:
                ptr = self.queues[class_idx].ptr
                self.queues[class_idx].feature[ptr:ptr + 1] = feature[idx:idx + 1]
                self.queues[class_idx].mask[ptr:ptr + 1] = mask[idx:idx + 1]
                self.queues[class_idx].box[ptr:ptr + 1] = box[idx:idx + 1]
                if img is not None:
                    self.queues[class_idx].img[ptr:ptr + 1] = img[idx:idx + 1]
                self.queues[class_idx].ptr = (ptr + 1) % self.len_queue
                create_new_gpu_bank = False
            return create_new_gpu_bank

    def cal_fg_iou(self, qobjs, kobjs):
        # return the min value of
        # foreground IoU and background IoU
        maskA, maskB = qobjs.get_mask(), kobjs.get_mask()
        maskB = maskB.to(maskA)  # might be in cpu
        fiou = (maskA * maskB).sum([1, 2]) / ((maskA + maskB) >= 1).float().sum([1, 2])
        return fiou

    def cal_bg_iou(self, qobjs, kobjs):
        maskA, maskB = qobjs.get_mask(), kobjs.get_mask()
        maskB = maskB.to(maskA)
        biou = ((1 - maskA) * (1 - maskB)).sum([1, 2]) / ((2 - maskA - maskB) >= 1).float().sum([1, 2])
        return biou

    def cal_appear_identity_sim(self, qobjs, kobjs):
        f0 = qobjs.get_feature()
        f1 = kobjs.get_feature()
        f1 = f1.to(f0)  # might be in cpu
        mask0 = qobjs.get_mask()
        mask1 = kobjs.get_mask()
        mask1 = mask1.to(mask0)  # might be in cpu
        mask0 = F.interpolate(mask0.unsqueeze(1), (f0.shape[2], f0.shape[3]), mode='bilinear',
                              align_corners=False).squeeze(1)
        mask1 = F.interpolate(mask1.unsqueeze(1), (f1.shape[2], f1.shape[3]), mode='bilinear',
                              align_corners=False).squeeze(1)
        sim = (f0 * f1 * mask0.unsqueeze(1) * mask1.unsqueeze(1)).sum([1, 2, 3]) / ((mask0 * mask1).sum([1, 2]) + 1e-6)
        return sim

    def cal_shape_ratio(self, qobj, kobjs):
        ratio0 = qobj.get_ratio().unsqueeze(1)
        ratio1 = kobjs.get_ratio().unsqueeze(0)
        ratio1 = ratio1.to(ratio0)  # might be in cpu
        return ratio0 / ratio1

    def get_similar_obj(self, qobj: ObjectElements):
        with torch.no_grad():
            category = qobj.get_category()

            if self.queues[category] is not None:
                kobjs = self.queues[qobj.category]
                fg_ious = self.cal_fg_iou(qobj, kobjs)
                bg_ious = self.cal_bg_iou(qobj, kobjs)
                appear_sim = self.cal_appear_identity_sim(qobj, kobjs)
                ratio = self.cal_shape_ratio(qobj, kobjs).squeeze(0)
                seg_masking = ((fg_ious > self.fg_iou_thresh).float() * (bg_ious > self.bg_iou_thresh).float()).to(fg_ious)
                sim_masking = (appear_sim > self.appear_thresh).float().to(fg_ious)
                ratio_masking = ((ratio >= self.ratio_range[0]).float() * (ratio <= self.ratio_range[1]).float()).to(
                    fg_ious)
                masking = torch.where((seg_masking * sim_masking * ratio_masking).bool())[0][
                          :self.max_retrieval_objs].long()
                ret_objs = kobjs[masking]
                return ret_objs
            else:
                return None
            # ObjectElements(torch.zeros([0, qmask.shape[1], qmask.shape[2]]).to(device), torch.zeros([0, qfeature.shape[1], qfeature.shape[2], qfeature.shape[3]]))


class SemanticCorrSolver:

    def __init__(self, exp, eps, gaussian_filter_size, low_score, num_iter, num_smooth_iter, dist_kernel):
        self.exp = exp
        self.eps = eps
        self.gaussian_filter_size = gaussian_filter_size
        self.low_score = low_score
        self.hsfilter = self.generate_gaussian_filter(gaussian_filter_size)
        self.num_iter = num_iter
        self.num_smooth_iter = num_smooth_iter
        self.count = None
        self.pairwise = None
        self.dist_kernel = dist_kernel
        self.ncells = 8192

    def generate_gaussian_filter(self, size=3):
        r"""Returns 2-dimensional gaussian filter"""
        dim = [size, size]

        siz = torch.LongTensor(dim)
        sig_sq = (siz.float() / 2 / 2.354).pow(2)
        siz2 = (siz - 1) / 2

        x_axis = torch.arange(-siz2[0], siz2[0] + 1).unsqueeze(0).expand(dim).float()
        y_axis = torch.arange(-siz2[1], siz2[1] + 1).unsqueeze(1).expand(dim).float()

        gaussian = torch.exp(-(x_axis.pow(2) / 2 / sig_sq[0] + y_axis.pow(2) / 2 / sig_sq[1]))
        gaussian = gaussian / gaussian.sum()

        return gaussian

    def perform_sinkhorn(self, a, b, M, reg, stopThr=1e-3, numItermax=100):
        # init data
        dim_a = a.shape[1]
        dim_b = b.shape[1]

        batch_size = b.shape[0]

        u = torch.ones((batch_size, dim_a), requires_grad=False).cuda() / dim_a
        v = torch.ones((batch_size, dim_b), requires_grad=False).cuda() / dim_b
        K = torch.exp(-M / reg)

        Kp = (1 / a).unsqueeze(2) * K
        cpt = 0
        err = 1
        KtransposeU = (K * u.unsqueeze(2)).sum(dim=1)  # has shape K.shape[1]

        while err > stopThr and cpt < numItermax:
            KtransposeU[...] = (K * u.unsqueeze(2)).sum(dim=1)  # has shape K.shape[1]
            v[...] = b / KtransposeU
            u[...] = 1. / (Kp * v.unsqueeze(1)).sum(dim=2)
            cpt = cpt + 1

        T = u.unsqueeze(2) * K * v.unsqueeze(1)
        # del u, K, v
        return T

    def appearance_similarityOT(self, m0, m1, sim):
        r"""Semantic Appearance Similarity"""

        pow_sim = torch.pow(torch.clamp(sim, min=0.3, max=0.7), 1.0)
        cost = 1 - pow_sim

        b, n1, n2 = sim.shape[0], sim.shape[1], sim.shape[2]
        m0, m1 = torch.clamp(m0, min=self.low_score, max=1 - self.low_score), torch.clamp(m1, min=self.low_score,
                                                                                          max=1 - self.low_score)
        mu = m0 / m0.sum(1, keepdim=True)
        nu = m1 / m1.sum(1, keepdim=True)
        with torch.no_grad():
            epsilon = self.eps
            cnt = 0
            while epsilon < 5:
                PI = self.perform_sinkhorn(mu, nu, cost, epsilon)
                if not torch.isnan(PI).any():
                    if cnt > 0:
                        print(cnt)
                    break
                else:
                    epsilon *= 2.0
                    cnt += 1
                    print(cnt, epsilon)

        if torch.isnan(PI).any():
            from IPython import embed
            embed()

        PI = n1 * PI  # re-scale PI
        exp = self.exp
        PI = torch.pow(torch.clamp(PI, min=0), exp)

        return PI

    def build_hspace(self, src_imsize, trg_imsize, ncells):
        r"""Build Hough space where voting is done"""
        hs_width = src_imsize[0] + trg_imsize[0]
        hs_height = src_imsize[1] + trg_imsize[1]
        hs_cellsize = math.sqrt((hs_width * hs_height) / ncells)
        nbins_x = int(hs_width / hs_cellsize) + 1
        nbins_y = int(hs_height / hs_cellsize) + 1

        return nbins_x, nbins_y, hs_cellsize

    def receptive_fields(self, rfsz, feat_size):
        r"""Returns a set of receptive fields (N, 4)"""
        width = feat_size[3]
        height = feat_size[2]

        feat_ids = torch.tensor(list(range(width))).repeat(1, height).t().repeat(1, 2).to(rfsz.device)
        feat_ids[:, 0] = torch.tensor(list(range(height))).unsqueeze(1).repeat(1, width).view(-1).to(rfsz.device)

        box = torch.zeros(feat_ids.size()[0], 4).to(rfsz.device)
        box[:, 0] = feat_ids[:, 1] - rfsz // 2
        box[:, 1] = feat_ids[:, 0] - rfsz // 2
        box[:, 2] = feat_ids[:, 1] + rfsz // 2
        box[:, 3] = feat_ids[:, 0] + rfsz // 2
        box = box.unsqueeze(0)

        return box

    def pass_message(self, T, shape):
        T = T.view(T.shape[0], shape[0], shape[1], shape[0], shape[1])
        pairwise = torch.zeros_like(T).to(T)
        count = torch.zeros_like(T).to(T)
        dxs, dys = [-1, 0, 1], [-1, 0, 1]
        for dx in dxs:
            for dy in dys:
                count[:, max(0, dy): min(shape[0] + dy, shape[0]), max(0, dx): min(shape[1] + dx, shape[1]),
                max(0, dy): min(shape[0] + dy, shape[0]), max(0, dx): min(shape[1] + dx, shape[1])] += 1
                pairwise[:, max(0, dy): min(shape[0] + dy, shape[0]), max(0, dx): min(shape[1] + dx, shape[1]),
                max(0, dy): min(shape[0] + dy, shape[0]), max(0, dx): min(shape[1] + dx, shape[1])] += \
                    T[:, max(0, -dy): min(shape[0] - dy, shape[0]), max(0, -dx): min(shape[1] - dx, shape[1]),
                    max(0, -dy): min(shape[0] - dy, shape[0]), max(0, -dx): min(shape[1] - dx, shape[1])]

        T[...] = pairwise / count
        T = T.view(T.shape[0], shape[0] * shape[1], shape[0] * shape[1])
        # del pairwise, count

        return T

    def solve(self, qobjs, kobjs, f0):
        r"""Regularized Hough matching"""
        # Unpack hyperpixels
        m0 = qobjs.mask.float()
        f0 = f0.float()
        f1 = kobjs['feature'].to(m0).float()
        m1 = kobjs['mask'].to(m0).float()
       
        fg_mask = m0.reshape(m0.shape[0], -1, 1) * m1.reshape(m1.shape[0], 1, -1)
        bg_mask = (1 - m0).reshape(m0.shape[0], -1, 1) * (1 - m1).reshape(m1.shape[0], 1, -1)
        
        m0 = F.interpolate(m0.unsqueeze(1), (f0.shape[2], f0.shape[3]), mode='bilinear', align_corners=False).squeeze(1)
        m1 = F.interpolate(m1.unsqueeze(1), (f1.shape[2], f1.shape[3]), mode='bilinear', align_corners=False).squeeze(1)
        shape = f0.shape[2], f0.shape[3]

        m0 = m0.reshape(m0.shape[0], -1)
        m1 = m1.reshape(m1.shape[0], -1)
        f0 = f0.reshape(f0.shape[0], f0.shape[1], -1).transpose(2, 1)
        f1 = f1.reshape(f1.shape[0], f1.shape[1], -1)

        f0_norm = torch.norm(f0, p=2, dim=2, keepdim=True) + 1e-4
        f1_norm = torch.norm(f1, p=2, dim=1, keepdim=True) + 1e-4
        with autocast(enabled=False):
            Cu = torch.matmul((f0 / f0_norm), (f1 / f1_norm))

        eye = torch.eye(shape[0] * shape[1]).to(f0).reshape(1, -1, shape[0], shape[1])
        dist_mask = F.max_pool2d(eye, kernel_size=self.dist_kernel, stride=1, padding=self.dist_kernel//2).reshape(1, shape[0] * shape[1],
                                                                                    shape[0] * shape[1]).transpose(2, 1)
        with torch.no_grad():
            C = Cu.clone() * dist_mask

        for i in range(self.num_iter):
            pairwise_votes = C.clone()
            for _ in range(self.num_smooth_iter):
                pairwise_votes = self.pass_message(pairwise_votes, (shape[0], shape[1]))
                pairwise_votes = pairwise_votes / (pairwise_votes.sum(2, keepdim=True) + 1e-4)

            max_val, _ = pairwise_votes.max(2, keepdim=True)

            C = Cu + pairwise_votes
            C = C / (C.sum(2, keepdim=True) + 1e-4)

        return Cu, C, fg_mask, bg_mask


@HEADS.register_module()
class DiscoBoxMaskFeatHead(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 start_level,
                 end_level,
                 num_classes,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=[dict(type='Normal', layer='Conv2d', std=0.01)]):
        super(DiscoBoxMaskFeatHead, self).__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self.convs_all_levels = nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()
            if i == 0:
                one_conv = ConvModule(
                    self.in_channels,
                    self.out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False)
                convs_per_level.add_module('conv' + str(i), one_conv)
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    chn = self.in_channels + 2 if i == 3 else self.in_channels
                    one_conv = ConvModule(
                        chn,
                        self.out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        inplace=False)
                    convs_per_level.add_module('conv' + str(j), one_conv)
                    one_upsample = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), one_upsample)
                    continue

                one_conv = ConvModule(
                    self.out_channels,
                    self.out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False)
                convs_per_level.add_module('conv' + str(j), one_conv)
                one_upsample = nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False)
                convs_per_level.add_module('upsample' + str(j), one_upsample)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = nn.Sequential(
            ConvModule(
                self.out_channels,
                self.num_classes,
                1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg),
        )

    @autocast()
    def forward(self, inputs):
        assert len(inputs) == (self.end_level - self.start_level + 1)
        inputs = [input.float() for input in inputs]

        feature_add_all_level = self.convs_all_levels[0](inputs[0])
        for i in range(1, len(inputs)):
            input_p = inputs[i]
            if i == 3:
                input_feat = input_p
                x_range = torch.linspace(-1, 1, input_feat.shape[-1], device=input_feat.device)
                y_range = torch.linspace(-1, 1, input_feat.shape[-2], device=input_feat.device)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([input_feat.shape[0], 1, -1, -1])
                x = x.expand([input_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                input_p = torch.cat([input_p, coord_feat], 1)

            feature_add_all_level += self.convs_all_levels[i](input_p)

        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred

def center_of_mass(bitmasks):
    _, h, w = bitmasks.size()
    ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
    xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

    m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
    m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
    m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
    center_x = m10 / m00
    center_y = m01 / m00
    return center_x, center_y

def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep


def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1).float()
    target = target.contiguous().view(target.size()[0], -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1-d

def mil_loss(loss_func, input, _, target):
    row_labels = target.max(1)[0]
    column_labels = target.max(2)[0]

    row_input = input.max(1)[0]
    column_input = input.max(2)[0]

    loss = loss_func(column_input, column_labels) +\
           loss_func(row_input, row_labels)

    return loss


def vis_seg(img_tensor, cur_mask, img_norm_cfg, save_dir='work_dirs/corr_vis', data_id=0):
    img = tensor2imgs(img_tensor, **img_norm_cfg)[0]

    h, w = img.shape[:2]

    cur_mask = cur_mask.cpu().numpy()
    cur_mask = mmcv.imresize(cur_mask, (w, h))
    cur_mask = (cur_mask > 0.5)
    cur_mask = cur_mask.astype(np.int32)

    seg_show = img.copy()
    color_mask = np.random.randint(
        0, 256, (1, 1, 3), dtype=np.uint8)
    cur_mask_bool = cur_mask.astype(np.bool)
    seg_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5

    mmcv.imwrite(seg_show, '{}/_{}.jpg'.format(save_dir, data_id))



class MeanField(nn.Module):

    # feature map (RGB)
    # B = #num of object
    # shape of [N 3 H W]
    def __init__(self, feature_map, kernel_size=3, require_grad=False, theta0=0.5, theta1=30, theta2=10, alpha0=3,
                 iter=20, base=0.45, gamma=0.01):
        super(MeanField, self).__init__()
        self.require_grad = require_grad
        self.kernel_size = kernel_size
        with torch.no_grad():
            self.unfold = torch.nn.Unfold(kernel_size, stride=1, padding=kernel_size//2)
            feature_map = feature_map + 10
            unfold_feature_map = self.unfold(feature_map).view(feature_map.size(0), feature_map.size(1), kernel_size**2, -1)
            self.feature_map = feature_map
            self.theta0 = theta0
            self.theta1 = theta1
            self.theta2 = theta2
            self.alpha0 = alpha0
            self.gamma = gamma
            self.base = base
            self.spatial = torch.tensor((np.arange(kernel_size**2)//kernel_size - kernel_size//2) ** 2 +\
                                        (np.arange(kernel_size**2) % kernel_size - kernel_size//2) ** 2).to(feature_map.device).float()

            self.kernel = alpha0 * torch.exp((-(unfold_feature_map - feature_map.view(feature_map.size(0), feature_map.size(1), 1, -1)) ** 2).sum(1) / (2 * self.theta0 ** 2) + (-(self.spatial.view(1, -1, 1) / (2 * self.theta1 ** 2))))
            self.kernel = self.kernel.unsqueeze(1)

            self.iter = iter

    # input x
    # shape of [N H W]
    def forward(self, x, targets, inter_img_mask=None):
        with torch.no_grad():
            x = x * targets
            x = (x > 0.5).float() * (1 - self.base*2) + self.base
            U = torch.cat([1-x, x], 1)
            U = U.view(-1, 1, U.size(2), U.size(3))
            if inter_img_mask is not None:
                inter_img_mask.reshape(-1, 1, inter_img_mask.shape[2], inter_img_mask.shape[3])
            ret = U
            for _ in range(self.iter):
                nret = self.simple_forward(ret, targets, inter_img_mask)
                ret = nret
            ret = ret.view(-1, 2, ret.size(2), ret.size(3))
            ret = ret[:,1:]
            ret = (ret > 0.5).float()
            count = ret.reshape(ret.shape[0], -1).sum(1)
            valid = (count >= ret.shape[2] * ret.shape[3] * 0.05) * (count <= ret.shape[2] * ret.shape[3] * 0.95)
            valid = valid.float()
        return ret, valid

    def simple_forward(self, x, targets, inter_img_mask):
        h, w = x.size(2), x.size(3)
        unfold_x = self.unfold(-torch.log(x)).view(x.size(0)//2, 2, self.kernel_size**2, -1)
        aggre = (unfold_x * self.kernel).sum(2)
        aggre = aggre.view(-1, 1, h, w)
        f = torch.exp(-aggre)
        f = f.view(-1, 2, h, w)
        if inter_img_mask is not None:
            f += inter_img_mask * self.gamma
        f[:, 1:] *= targets
        f = f + 1e-6
        f = f / f.sum(1, keepdim=True)
        f = (f > 0.5).float() * (1 - self.base*2) + self.base
        f = f.view(-1, 1, h, w)

        return f



@HEADS.register_module()
class DiscoBoxSOLOv2Head(BaseModule):

    def __init__(self,
                 num_classes,
                 in_channels,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 sigma=0.2,
                 num_grids=None,
                 ins_out_channels=64,
                 loss_ins=None,
                 loss_ts=None,
                 loss_cate=None,
                 loss_corr=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 use_dcn_in_tower=False,
                 type_dcn=None,
                 init_cfg=dict(
                    type='Normal', 
                    layer='Conv2d', 
                    std=0.01,
                    override=dict(
                        type='Normal',
                        name='solo_cate',
                        std=0.01,
                        bias_prob=0.01))):
        super(DiscoBoxSOLOv2Head, self).__init__(init_cfg=init_cfg)
        self.fp16_enabled = False
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes
        self.ins_out_channels = ins_out_channels
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.stacked_convs = stacked_convs
        self.kernel_out_channels = self.ins_out_channels * 1 * 1
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.scale_mids = torch.tensor(np.array(scale_ranges))
        self.scale_mids = (self.scale_mids[:, 0] * self.scale_mids[:, 1]) ** 0.5
        self.loss_cate = build_loss(loss_cate)
        self.ins_loss_weight = loss_ins['loss_weight']
        self.ins_loss_type = loss_ins['type']
        self.ts_loss_weight = loss_ts['loss_weight']
        self.alpha0 = loss_ts['alpha0']
        self.theta0 = loss_ts['theta0']
        self.theta1 = loss_ts['theta1']
        self.theta2 = loss_ts['theta2']
        self.mkernel = loss_ts['kernel']
        self.crf_base = loss_ts['base']
        self.crf_max_iter = loss_ts['max_iter']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_dcn_in_tower = use_dcn_in_tower
        self.type_dcn = type_dcn
        self._init_layers()

        if loss_corr is not None:
            self.semantic_corr_solver = SemanticCorrSolver(loss_corr['corr_exp'],
                                                           loss_corr['corr_eps'],
                                                           loss_corr['gaussian_filter_size'],
                                                           loss_corr['low_score'],
                                                           loss_corr['corr_num_iter'],
                                                           loss_corr['corr_num_smooth_iter'],
                                                           dist_kernel=loss_corr['dist_kernel'])

            obj_bank = loss_corr['obj_bank']
            self.vis_cnt = 0
            self.corr_loss_weight = loss_corr['loss_weight']
            self.object_queues = ObjectQueues(num_class=num_classes, len_queue=obj_bank['len_object_queues'],
                                              fg_iou_thresh=obj_bank['fg_iou_thresh'],
                                              bg_iou_thresh=obj_bank['bg_iou_thresh'],
                                              ratio_range=obj_bank['ratio_range'],
                                              appear_thresh=obj_bank['appear_thresh'],
                                              max_retrieval_objs=obj_bank['max_retrieval_objs'])

            self.img_norm_cfg = obj_bank['img_norm_cfg']
            self.feat_roi_align = RoIAlign((obj_bank['feat_height'], obj_bank['feat_width']))
            self.mask_roi_align = RoIAlign((obj_bank['mask_height'], obj_bank['mask_width']))
            self.img_roi_align = RoIAlign((obj_bank['img_height'], obj_bank['img_width']))
            self.corr_feat_height, self.corr_feat_width = obj_bank['feat_height'], obj_bank['feat_width']
            self.corr_mask_height, self.corr_mask_width = obj_bank['mask_height'], obj_bank['mask_width']
            self.objbank_min_size = obj_bank['min_size']
            self.save_corr_img = loss_corr['save_corr_img']
            self.qobj = None
            self.num_created_gpu_bank = 0
            self.num_gpu_bank = obj_bank['num_gpu_bank']
            self.color_panel = np.array([ (i * 32, j * 32, k * 32) for i in range(8)
                                 for j in range(8) for k in range(8) ])
            np.random.shuffle(self.color_panel)

            self.loss_corr = nn.CrossEntropyLoss()

        # for debug
        self.cnt = 0

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.cate_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            if self.use_dcn_in_tower:
                cfg_conv = dict(type=self.type_dcn)
            else:
                cfg_conv = self.conv_cfg

            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.kernel_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

        self.solo_cate = nn.Conv2d(
            self.seg_feat_channels, self.cate_out_channels, 3, padding=1)

        self.solo_kernel = nn.Conv2d(
            self.seg_feat_channels, self.kernel_out_channels, 3, padding=1)

    def forward(self, feats, eval=False):
        feats = [feat.float() for feat in feats]
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)
        cate_pred, kernel_pred = multi_apply(self.forward_single, new_feats,
                                                       list(range(len(self.seg_num_grids))),
                                                       eval=eval, upsampled_size=upsampled_size)
        return cate_pred, kernel_pred

    def split_feats(self, feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear'))

    @autocast()
    def forward_single(self, x, idx, eval=False, upsampled_size=None):
        ins_kernel_feat = x
        # ins branch
        # concat coord
        x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
        y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)
        
        # kernel branch
        kernel_feat = ins_kernel_feat
        seg_num_grid = self.seg_num_grids[idx]
        kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear')

        cate_feat = kernel_feat[:, :-2, :, :]

        kernel_feat = kernel_feat.contiguous()
        for i, kernel_layer in enumerate(self.kernel_convs):
            kernel_feat = kernel_layer(kernel_feat)
        kernel_pred = self.solo_kernel(kernel_feat)

        # cate branch
        cate_feat = cate_feat.contiguous()
        for i, cate_layer in enumerate(self.cate_convs):
            cate_feat = cate_layer(cate_feat)
        cate_pred = self.solo_cate(cate_feat)

        if eval:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return cate_pred, kernel_pred

    def superres_T(self, T):
        # bilinear interpolation
        T_bchw = T.reshape(-1, self.corr_feat_height * self.corr_feat_width, self.corr_feat_height, self.corr_feat_width)
        T_bchw = F.interpolate(T_bchw, (self.corr_mask_height, self.corr_mask_width),
                               mode='bilinear', align_corners=False)
        T_b1hwc = T_bchw.reshape(-1, 1, self.corr_feat_height, self.corr_feat_width,
                                 self.corr_mask_height * self.corr_mask_width)
        T_b1hwc = F.interpolate(T_b1hwc,
                                (self.corr_mask_height, self.corr_mask_width, self.corr_mask_height * self.corr_mask_width),
                                mode='trilinear', align_corners=False)
        T_superres = T_b1hwc.reshape(-1, self.corr_mask_height * self.corr_mask_width,
                                     self.corr_mask_height * self.corr_mask_width) * \
                     (1.0 * self.corr_feat_height * self.corr_feat_width / self.corr_mask_height / self.corr_mask_width)

        return T_superres

    def vis_corr(self, img_a_tensor, img_b_tensor, T, a_mask, b_mask, **img_norm_cfg):

        img_a = tensor2imgs(img_a_tensor, **img_norm_cfg)[0]
        img_b = tensor2imgs(img_b_tensor, **img_norm_cfg)[0]


        img_ab = np.zeros((img_a.shape[0], img_a.shape[1] + img_b.shape[1], 3), np.uint8)
        img_ab[:, :img_a.shape[1]] = img_a
        img_ab[:, img_a.shape[1]:] = img_b

        img_size = img_a.shape[0]

        assignment = T.argmax(1)
        size = int(assignment.shape[0] ** 0.5)
        assignment = assignment.reshape(size, size)

        a_mask = F.interpolate(a_mask.unsqueeze(0), (size, size), mode='bilinear', align_corners=False).squeeze()
        b_mask = F.interpolate(b_mask.unsqueeze(0), (size, size), mode='bilinear', align_corners=False).squeeze()

        scale = img_size / size

        for i in range(0,size,2):
            for j in range(0,size,2):
                x = assignment[i, j] % size
                y = assignment[i, j] // size
                if a_mask[i,j] > 0.5 and b_mask[x,y] > 0.5:
                    cv2.line(img_ab, (int(i * scale), int(j * scale)), (int((x + size) * scale), int(y * scale)),
                             color=tuple(self.color_panel[i*size + j].tolist()), thickness=5)

        self.vis_cnt += 1

        cv2.imwrite('corr_vis/{}.jpg'.format(self.vis_cnt), img_ab)

    def corr_loss(self,
                  cate_preds,
                  s_kernel_preds_raw,
                  t_kernel_preds_raw,
                  s_ins_pred,
                  t_ins_pred,
                  gt_bbox_list,
                  gt_label_list,
                  gt_mask_list,
                  mean_fields,
                  img_metas,
                  cfg,
                  img=None,
                  gt_bboxes_ignore=None,
                  use_loss_ts=False,
                  use_ind_teacher=False,
                  s_feat=None,
                  t_feat=None):

        mask_feat_size = s_ins_pred.size()[-2:]
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = multi_apply(
            self.best_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            mask_feat_size=mask_feat_size)
        # ins
        ins_labels = [torch.cat([ins_labels_level_img
                                 for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]

        kernel_label_list = [torch.cat([
            cate_label_list[batch_idx][level_idx].reshape(-1)[grid_order_list[batch_idx][level_idx]]
            for batch_idx in range(len(grid_order_list))], 0)
            for level_idx in range(len(grid_order_list[0]))]

        s_kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                         for kernel_preds_level_img, grid_orders_level_img in
                         zip(kernel_preds_level, grid_orders_level)]
                        for kernel_preds_level, grid_orders_level in zip(s_kernel_preds_raw, zip(*grid_order_list))]

        if use_ind_teacher:
            t_kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                               for kernel_preds_level_img, grid_orders_level_img in
                               zip(kernel_preds_level, grid_orders_level)]
                              for kernel_preds_level, grid_orders_level in zip(t_kernel_preds_raw, zip(*grid_order_list))]
        else:
            t_kernel_preds = s_kernel_preds


        # generate masks
        s_ins_pred_list = []
        t_ins_pred_list = []
        color_feats = F.interpolate(img, (s_ins_pred.shape[2], s_ins_pred.shape[3]), mode='bilinear',
                                    align_corners=True)


        img_ind_list = []
        # This code segmentation is for weakly supervised instance segmentation
        # if no independent teacher, t_kenerl_preds is assigned to be s_kernel_preds
        for b_s_kernel_pred, b_t_kernel_pred in zip(s_kernel_preds, t_kernel_preds):
            b_s_mask_pred = []
            b_t_mask_pred = []
            b_img_inds = []
            for idx, (s_kernel_pred, t_kernel_pred) in enumerate(zip(b_s_kernel_pred, b_t_kernel_pred)):

                if s_kernel_pred.size()[-1] == 0:
                    continue
                s_cur_ins_pred = s_ins_pred[idx, ...]
                H, W = s_cur_ins_pred.shape[-2:]
                N, I = s_kernel_pred.shape
                s_cur_ins_pred = s_cur_ins_pred.unsqueeze(0)
                s_kernel_pred = s_kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                s_cur_ins_pred = F.conv2d(s_cur_ins_pred, s_kernel_pred, stride=1).view(-1, H, W)
                b_s_mask_pred.append(s_cur_ins_pred)

                if use_ind_teacher:
                    t_cur_ins_pred = t_ins_pred[idx, ...]
                    t_cur_ins_pred = t_cur_ins_pred.unsqueeze(0)
                    t_kernel_pred = t_kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                    t_cur_ins_pred = F.conv2d(t_cur_ins_pred, t_kernel_pred, stride=1).view(-1, H, W)
                    b_t_mask_pred.append(t_cur_ins_pred)

                b_img_inds.append(torch.ones(s_cur_ins_pred.shape[0]) * idx)
            if len(b_s_mask_pred) == 0:
                b_s_mask_pred = None
                if use_ind_teacher:
                    b_t_mask_pred = None
                b_img_inds = None
            else:
                b_s_mask_pred = torch.cat(b_s_mask_pred, 0)
                if use_ind_teacher:
                    b_t_mask_pred = torch.cat(b_t_mask_pred, 0)
                b_img_inds = torch.cat(b_img_inds, 0)
            s_ins_pred_list.append(b_s_mask_pred)
            # if no independent teacher, t_ins_pred_list is assigned to be s_ins_pred_list
            if use_ind_teacher:
                t_ins_pred_list.append(b_t_mask_pred)
            else:
                t_ins_pred_list = s_ins_pred_list
            img_ind_list.append(b_img_inds)

        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()
        corr_loss = torch.tensor(0).to(s_ins_pred).float()
        num_ins = 0
        loss_ts = []

        for s_input, t_input, img_inds, target, kernel_labels in \
                zip(s_ins_pred_list, t_ins_pred_list, img_ind_list, ins_labels, kernel_label_list):
            if s_input is None:
                continue
            s_input = torch.sigmoid(s_input)
            if use_ind_teacher:
                t_input = torch.sigmoid(t_input)
            else:
                t_input = s_input

            # remove all-zero target
            mask = torch.tensor([t.sum() for t in target]).to(s_input).bool()
            if mask.sum() == 0:
                continue
            # keep non-zero target
            s_input, t_input, img_inds, target = s_input[mask], t_input[mask], img_inds[mask], target[mask]

            pos_inds = [torch.where(t) for t in target]
            min_y, max_y, min_x, max_x = \
                torch.tensor([ids[0].min() for ids in pos_inds]), torch.tensor(
                    [ids[0].max() for ids in pos_inds]) + 1, \
                torch.tensor([ids[1].min() for ids in pos_inds]), torch.tensor(
                    [ids[1].max() for ids in pos_inds]) + 1
            boxes = torch.cat([min_x.unsqueeze(1), min_y.unsqueeze(1), max_x.unsqueeze(1), max_y.unsqueeze(1)],
                              1).to(s_input)

            roi_s_feat = relu_and_l2_norm_feat(self.feat_roi_align(
                s_feat, torch.cat([img_inds.to(s_feat).unsqueeze(1), boxes], 1)))
            with torch.no_grad():
                roi_t_feat = relu_and_l2_norm_feat(self.feat_roi_align(
                    t_feat.detach(), torch.cat([img_inds.to(s_feat).unsqueeze(1), boxes], 1)))
            if self.save_corr_img:
                roi_img = self.img_roi_align(img, torch.cat(
                    [img_inds.to(t_input).unsqueeze(1), boxes*4], 1)).clone().detach()

            with torch.no_grad():
                roi_s_mask = self.mask_roi_align(s_input.unsqueeze(1).detach(), torch.cat(
                    [torch.arange(target.shape[0]).to(t_input).unsqueeze(1), boxes], 1)).squeeze(1).detach()
                roi_t_mask = self.mask_roi_align(t_input.unsqueeze(1).detach(), torch.cat(
                    [torch.arange(target.shape[0]).to(t_input).unsqueeze(1), boxes], 1)).squeeze(1).detach()
                iiu = torch.zeros(t_input.shape[0] * 2, *t_input.shape[1:]).to(t_input)
                iiu_mask = torch.zeros(t_input.shape[0] * 2).to(t_input)
                queue_area_mask = ((max_x - min_x) > self.objbank_min_size) * (
                        (max_y - min_y) > self.objbank_min_size)

            for idx in torch.arange(len(queue_area_mask)):
                if self.qobj is None:
                    self.qobj = ObjectFactory.create_one(mask=roi_s_mask[idx:idx + 1].detach(),
                                                         feature=roi_s_feat[idx:idx + 1].detach(),
                                                         box=boxes[idx:idx + 1].detach(),
                                                         category=kernel_labels[idx],
                                                         img=roi_img[idx:idx + 1] if self.save_corr_img else None)
                else:
                    self.qobj.mask[...] = roi_s_mask[idx:idx + 1].detach()
                    self.qobj.feature[...] = roi_s_feat[idx:idx + 1].detach()
                    self.qobj.box[...] = boxes[idx:idx + 1].detach()
                    self.qobj.category = int(kernel_labels[idx])
                    if self.save_corr_img:
                        self.qobj.img[...] = roi_img[idx:idx + 1]

                kobjs = self.object_queues.get_similar_obj(self.qobj)
                if kobjs is not None and kobjs['mask'].shape[0] >= 5:
                    Cu, T, fg_mask, bg_mask = self.semantic_corr_solver.solve(self.qobj, kobjs, roi_s_feat[idx:idx + 1])

                    if self.save_corr_img:
                        self.vis_corr(self.qobj.img, kobjs['img'][0:1], T[0], self.qobj.mask, kobjs['mask'][0:1], **self.img_norm_cfg)
                    nce_loss = nn.CrossEntropyLoss()
                    assignment = T.argmax(2).reshape(-1)
                    Cu = Cu.float()
                    Cu = F.softmax(Cu, 2).reshape(-1, Cu.shape[2])
                    corr_loss += nce_loss(Cu, assignment)
                    num_ins += 1

                    with torch.no_grad():
                        T = T * Cu.reshape(T.shape)

                    T = T / (T.sum(2, keepdim=True) + 1e-5)

                    T_superres = self.superres_T(T)

                    fg_ci = torch.matmul(T_superres * (fg_mask > 0.5).float(), torch.clamp(kobjs['mask'], min=0.1, max=0.9).reshape(T_superres.shape[0], T_superres.shape[2], 1).to(Cu)).mean(0).reshape(roi_s_mask.shape[1:])
                    bg_ci = torch.matmul(T_superres * (bg_mask > 0.5).float(), torch.clamp(1-kobjs['mask'], min=0.1, max=0.9).reshape(T_superres.shape[0], T_superres.shape[2], 1).to(Cu)).mean(0).reshape(roi_s_mask.shape[1:])

                    fg_ci = F.interpolate(fg_ci.reshape(1, 1, fg_ci.shape[0], fg_ci.shape[1]),
                                          (int(boxes[idx][3] - boxes[idx][1]), int(boxes[idx][2] - boxes[idx][0])),
                                          mode='bilinear', align_corners=False).squeeze()
                    bg_ci = F.interpolate(bg_ci.reshape(1, 1, bg_ci.shape[0], bg_ci.shape[1]),
                                          (int(boxes[idx][3] - boxes[idx][1]), int(boxes[idx][2] - boxes[idx][0])),
                                          mode='bilinear', align_corners=False).squeeze()
                    iiu[idx*2, int(boxes[idx, 1]):int(boxes[idx, 3]),
                                int(boxes[idx, 0]):int(boxes[idx, 2])] = bg_ci
                    iiu[idx*2+1, int(boxes[idx, 1]):int(boxes[idx, 3]),
                                int(boxes[idx, 0]):int(boxes[idx, 2])] = fg_ci

                    if self.save_corr_img:
                        self.cnt += 1
                        vis_seg(self.qobj.img, ci, self.img_norm_cfg, save_dir='work_dirs/corr_vis', data_id=self.cnt)
                        self.cnt += 1
                        vis_seg(self.qobj.img, roi_s_mask[idx], self.img_norm_cfg, save_dir='work_dirs/corr_vis', data_id=self.cnt)
                if queue_area_mask[idx]:
                    if self.num_created_gpu_bank < self.num_gpu_bank:
                        device = mask.device
                    else:
                        device = 'cpu'
                    created_gpu_bank = self.object_queues.append(int(kernel_labels[idx]),
                                                                 idx,
                                                                 roi_t_feat,
                                                                 roi_t_mask,
                                                                 boxes.detach(),
                                                                 roi_img if self.save_corr_img else None,
                                                                 device=device)
                    self.num_created_gpu_bank += created_gpu_bank

            iiu = iiu.reshape(iiu.shape[0] // 2, 2, iiu.shape[1], iiu.shape[2])
            for img_idx in range(len(mean_fields)):
                obj_inds = (img_inds == img_idx)
                enlarged_target = F.max_pool2d(target.float().unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1).byte()
                if obj_inds.sum() > 0:
                    pseudo_label, valid = mean_fields[int(img_idx)](
                        (t_input[obj_inds].unsqueeze(1) + s_input[obj_inds].unsqueeze(1)) / 2,
                        target[obj_inds].unsqueeze(1), iiu[obj_inds])
                    cropped_s_input = s_input[obj_inds] * enlarged_target[obj_inds]
                    cropped_s_input = cropped_s_input * mean_fields[int(img_idx)].gamma + cropped_s_input.detach() * (1 - mean_fields[int(img_idx)].gamma)
                    loss_ts.append(dice_loss(cropped_s_input, pseudo_label))

        return corr_loss / (num_ins + 1e-4), loss_ts


    @autocast()
    def loss(self,
                 cate_preds,
                 s_kernel_preds_raw,
                 t_kernel_preds_raw,
                 s_ins_pred,
                 t_ins_pred,
                 gt_bbox_list,
                 gt_label_list,
                 gt_mask_list,
                 img_metas,
                 cfg,
                 img=None,
                 gt_bboxes_ignore=None,
                 use_loss_ts=False,
                 use_ind_teacher=False,
                 use_corr=False,
                 s_feat=None,
                 t_feat=None):

        mask_feat_size = s_ins_pred.size()[-2:]
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = multi_apply(
            self.solov2_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            mask_feat_size=mask_feat_size)

        if s_feat is not None:
            s_feat = s_feat[0]
        if t_feat is not None:
            t_feat = t_feat[0]

        # ins
        ins_labels = [torch.cat([ins_labels_level_img
                                 for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]

        s_kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                         for kernel_preds_level_img, grid_orders_level_img in
                         zip(kernel_preds_level, grid_orders_level)]
                        for kernel_preds_level, grid_orders_level in zip(s_kernel_preds_raw, zip(*grid_order_list))]

        if use_ind_teacher:
            t_kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                               for kernel_preds_level_img, grid_orders_level_img in
                               zip(kernel_preds_level, grid_orders_level)]
                              for kernel_preds_level, grid_orders_level in zip(t_kernel_preds_raw, zip(*grid_order_list))]
        else:
            t_kernel_preds = s_kernel_preds

        kernel_label_list = [torch.cat([
            cate_label_list[batch_idx][level_idx].reshape(-1)[grid_order_list[batch_idx][level_idx]]
            for batch_idx in range(len(grid_order_list))], 0)
            for level_idx in range(len(grid_order_list[0]))]

        # generate masks
        s_ins_pred_list = []
        t_ins_pred_list = []
        color_feats = F.interpolate(img, (s_ins_pred.shape[2], s_ins_pred.shape[3]), mode='bilinear', align_corners=True)

        img_ind_list = []
        # This code segmentation is for weakly supervised instance segmentation
        # if no independent teacher, t_kenerl_preds is assigned to be s_kernel_preds
        for b_s_kernel_pred, b_t_kernel_pred in zip(s_kernel_preds, t_kernel_preds):
            b_s_mask_pred = []
            b_t_mask_pred = []
            b_img_inds = []
            for idx, (s_kernel_pred, t_kernel_pred) in enumerate(zip(b_s_kernel_pred, b_t_kernel_pred)):

                if s_kernel_pred.size()[-1] == 0:
                    continue
                s_cur_ins_pred = s_ins_pred[idx, ...]
                H, W = s_cur_ins_pred.shape[-2:]
                N, I = s_kernel_pred.shape
                s_cur_ins_pred = s_cur_ins_pred.unsqueeze(0)
                s_kernel_pred = s_kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                s_cur_ins_pred = F.conv2d(s_cur_ins_pred, s_kernel_pred, stride=1).view(-1, H, W)
                b_s_mask_pred.append(s_cur_ins_pred)

                if use_ind_teacher:
                    t_cur_ins_pred = t_ins_pred[idx, ...]
                    t_cur_ins_pred = t_cur_ins_pred.unsqueeze(0)
                    t_kernel_pred = t_kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                    t_cur_ins_pred = F.conv2d(t_cur_ins_pred, t_kernel_pred, stride=1).view(-1, H, W)
                    b_t_mask_pred.append(t_cur_ins_pred)

                b_img_inds.append(torch.ones(s_cur_ins_pred.shape[0]) * idx)
            if len(b_s_mask_pred) == 0:
                b_s_mask_pred = None
                if use_ind_teacher:
                    b_t_mask_pred = None
                b_img_inds = None
            else:
                b_s_mask_pred = torch.cat(b_s_mask_pred, 0)
                if use_ind_teacher:
                    b_t_mask_pred = torch.cat(b_t_mask_pred, 0)
                b_img_inds = torch.cat(b_img_inds, 0)
            s_ins_pred_list.append(b_s_mask_pred)
            # if no independent teacher, t_ins_pred_list is assigned to be s_ins_pred_list
            if use_ind_teacher:
                t_ins_pred_list.append(b_t_mask_pred)
            else:
                t_ins_pred_list = s_ins_pred_list
            img_ind_list.append(b_img_inds)

        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()


        #weakly supervised semantic correspondence

        # dice loss
        loss_ins = []
        loss_ts = []
        loss_corr = []

        # Mean Field Init
        mean_fields = [MeanField(color_feat.unsqueeze(0), alpha0=self.alpha0,
                                 theta0=self.theta0, theta1=self.theta1, theta2=self.theta2,
                                 iter=self.crf_max_iter, kernel_size=self.mkernel, base=self.crf_base) \
                       for color_feat in color_feats]

        for s_input, t_input, img_inds, target, kernel_labels in \
                zip(s_ins_pred_list, t_ins_pred_list, img_ind_list, ins_labels, kernel_label_list):
            if s_input is None:
                continue
            s_input = torch.sigmoid(s_input)
            if use_ind_teacher:
                t_input = torch.sigmoid(t_input)
            else:
                t_input = s_input

            # remove all-zero target
            mask = torch.tensor([t.sum() for t in target]).to(s_input).bool()
            if mask.sum() == 0:
                continue
            # keep non-zero target
            s_input, t_input, img_inds, target = s_input[mask], t_input[mask], img_inds[mask], target[mask]

            # unary loss
            loss_ins.append(mil_loss(dice_loss, s_input, s_input, target))

            # pairwise loss
            # crf
            if use_loss_ts:
                enlarged_target = F.max_pool2d(target.float().unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1).byte()
                for img_idx in range(len(mean_fields)):
                    obj_inds = (img_inds == img_idx)
                    if obj_inds.sum() > 0:
                        pseudo_label, valid = mean_fields[int(img_idx)](
                            (t_input[obj_inds].unsqueeze(1) + s_input[obj_inds].unsqueeze(1))/2, target[obj_inds].unsqueeze(1))
                        loss_ts.append(dice_loss(s_input[obj_inds] * enlarged_target[obj_inds], pseudo_label))

        if len(loss_ins) > 0:
            loss_ins = torch.cat(loss_ins).mean()
            loss_ins = loss_ins * self.ins_loss_weight
        else:
            loss_ins = torch.zeros(1).to(color_feats)


        # corr loss
        if use_loss_ts and use_corr:
            loss_corr, corr_loss_ts = self.corr_loss(
                cate_preds,
                s_kernel_preds_raw,
                t_kernel_preds_raw,
                s_ins_pred,
                t_ins_pred,
                gt_bbox_list,
                gt_label_list,
                gt_mask_list,
                mean_fields,
                img_metas,
                cfg,
                img=img,
                gt_bboxes_ignore=gt_bboxes_ignore,
                use_loss_ts=use_loss_ts,
                use_ind_teacher=use_ind_teacher,
                s_feat=s_feat,
                t_feat=t_feat)
            loss_corr = loss_corr * self.corr_loss_weight
            corr_loss_ts = torch.cat(corr_loss_ts).mean()
        else:
            loss_corr = torch.tensor(0).to(loss_ins)
            corr_loss_ts = torch.tensor(0).to(loss_ins)

        if use_loss_ts and len(loss_ts) > 0:
            loss_ts = torch.cat(loss_ts).mean()
            loss_ts = (loss_ts + corr_loss_ts) * self.ts_loss_weight
        else:
            loss_ts = torch.zeros(1).mean().to(loss_ins)

        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)
        flatten_cate_preds = flatten_cate_preds.float()

        loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_labels, avg_factor=num_ins + 1)
        return dict(
            loss_ins=loss_ins,
            loss_ts=loss_ts,
            loss_cate=loss_cate,
            loss_corr=loss_corr)

    def best_target_single(self,
                           gt_bboxes_raw,
                           gt_labels_raw,
                           gt_masks_raw,
                           mask_feat_size):


        device = gt_labels_raw[0].device

        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1])).unsqueeze(1)

        scale_mids = self.scale_mids.to(device).unsqueeze(0)
        scale_diffs = scale_mids / (gt_areas + 1e-6)
        scale_diffs[scale_diffs < 1] = 1 / (scale_diffs[scale_diffs < 1] + 1e-6)
        scale_ids = scale_diffs.argmin(1)

        ins_label_list = []
        ins_ind_label_list = []
        cate_label_list = []
        grid_order_list = []
        for level_ids, (stride, num_grid) in enumerate(zip(self.strides, self.seg_num_grids)):

            hit_indices = (level_ids == scale_ids).nonzero().flatten()
            num_ins = len(hit_indices)

            ins_label = []
            grid_order = []
            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device) + self.num_classes
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            if num_ins == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # mass center
            gt_masks_pt = torch.from_numpy(gt_masks.to_ndarray()).to(device=device)
            center_ws, center_hs = center_of_mass(gt_masks_pt)
            valid_mask_flags = gt_masks_pt.sum(dim=-1).sum(dim=-1) > 0

            output_stride = 4
            for seg_mask, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_labels, half_hs, half_ws, center_hs, center_ws, valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                cate_label[coord_h, coord_w] = gt_label
                seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = torch.from_numpy(seg_mask).to(device=device)
                label = int(coord_h * num_grid + coord_w)
                cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                            device=device)
                cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                ins_label.append(cur_ins_label)
                ins_ind_label[label] = True
                grid_order.append(label)
            if len(ins_label) == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            else:
                ins_label = torch.stack(ins_label, 0)
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

    def solov2_target_single(self,
                           gt_bboxes_raw,
                           gt_labels_raw,
                           gt_masks_raw,
                           mask_feat_size):

        device = gt_labels_raw[0].device

        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        for (lower_bound, upper_bound), stride, num_grid \
                in zip(self.scale_ranges, self.strides, self.seg_num_grids):

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            num_ins = len(hit_indices)

            ins_label = []
            grid_order = []
            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device) + self.num_classes
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            if num_ins == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # mass center
            gt_masks_pt = torch.from_numpy(gt_masks.to_ndarray()).to(device=device)
            center_ws, center_hs = center_of_mass(gt_masks_pt)
            valid_mask_flags = gt_masks_pt.sum(dim=-1).sum(dim=-1) > 0

            output_stride = 4
            for seg_mask, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_labels, half_hs, half_ws, center_hs, center_ws, valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)

                cate_label[top:(down+1), left:(right+1)] = gt_label
                seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = torch.from_numpy(seg_mask).to(device=device)
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)

                        cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                    device=device)
                        cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_label.append(cur_ins_label)
                        ins_ind_label[label] = True
                        grid_order.append(label)
            if len(ins_label) == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            else:
                ins_label = torch.stack(ins_label, 0)
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

    def get_seg(self, cate_preds, kernel_preds, seg_pred, img_metas, cfg, rescale=None, img=None):
        num_levels = len(cate_preds)
        featmap_size = seg_pred.size()[-2:]

        result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.cate_out_channels).detach() for i in range(num_levels)
            ]
            seg_pred_list = seg_pred[img_id, ...].unsqueeze(0)
            kernel_pred_list = [
                kernel_preds[i][img_id].permute(1, 2, 0).view(-1, self.kernel_out_channels).detach()
                                for i in range(num_levels)
            ]

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            kernel_pred_list = torch.cat(kernel_pred_list, dim=0)

            result = self.get_seg_single(
                cate_pred_list, 
                seg_pred_list, 
                kernel_pred_list,
                featmap_size, 
                img_meta=img_metas[img_id],
                cfg=cfg)
            result_list.append(result)
        return result_list

    @autocast()
    def get_seg_single(self,
                       cate_preds,
                       seg_preds,
                       kernel_preds,
                       featmap_size,
                       img_meta,
                       cfg):

        def empty_results(results, cls_scores):
            results.scores = cls_scores.new_ones(0)
            results.masks = cls_scores.new_zeros(0, *results.ori_shape[:2])
            results.labels = cls_scores.new_ones(0)
            return results

        assert len(cate_preds) == len(kernel_preds)
        results = InstanceData(img_meta)

        img_shape = results.img_shape
        ori_shape = results.ori_shape

        # overall info.
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.
        inds = (cate_preds > cfg.score_thr)
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return empty_results(results, cate_scores)

        # cate_labels & kernel_preds
        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(size_trans[-1])

        n_stage = len(self.seg_num_grids)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_-1]:size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]

        # mask encoding.
        I, N = kernel_preds.shape
        kernel_preds = kernel_preds.view(I, N, 1, 1)
        seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()

        # mask.
        seg_masks = seg_preds > cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return empty_results(results, cate_scores)

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # mask scoring.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # Matrix NMS
        scores, labels, _, keep_inds  = mask_matrix_nms(
            seg_masks, 
            cate_labels, 
            cate_scores,
            filter_thr=cfg.filter_thr,
            nms_pre=cfg.nms_pre,
            max_num=cfg.max_per_img,
            kernel=cfg.kernel, 
            sigma=cfg.sigma, 
            mask_area=sum_masks)

        seg_preds = seg_preds[keep_inds]

        if seg_preds.shape[0] == 0:
            return empty_results(results, cate_scores)

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]

        seg_masks = F.interpolate(seg_preds,
                               size=ori_shape[:2],
                               mode='bilinear').squeeze(0)

        masks = seg_masks > cfg.mask_thr ##

        results.masks = masks
        results.labels = labels
        results.scores = scores
        
        return results
