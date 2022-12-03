import torch
from torch import nn
import torch.distributed as dist

from ..functions.mst import mst
from ..functions.bfs import bfs
from ..functions.refine import refine

class MinimumSpanningTree(nn.Module):
    def __init__(self, distance_func):
        super(MinimumSpanningTree, self).__init__()
        self.distance_func = distance_func

    @staticmethod
    def _build_matrix_index(fm):
        batch, height, width = (fm.shape[0], *fm.shape[2:])
        row = torch.arange(width, dtype=torch.int32, device=fm.device).unsqueeze(0)
        col = torch.arange(height, dtype=torch.int32, device=fm.device).unsqueeze(1)
        raw_index = row + col * width
        row_index = torch.stack([raw_index[:-1, :], raw_index[1:, :]], 2)
        col_index = torch.stack([raw_index[:, :-1], raw_index[:, 1:]], 2)
        index = torch.cat([row_index.reshape(1, -1, 2),
                           col_index.reshape(1, -1, 2)], 1)
        index = index.expand(batch, -1, -1)
        return index

    def _build_feature_weight(self, fm):
        batch = fm.shape[0]
        weight_row = self.distance_func(fm[:, :, :-1, :], fm[:, :, 1:, :])
        weight_col = self.distance_func(fm[:, :, :, :-1], fm[:, :, :, 1:])
        weight_row = weight_row.reshape([batch, -1])
        weight_col = weight_col.reshape([batch, -1])
        weight = torch.cat([weight_row, weight_col], dim=1) + 1
        return weight

    def _build_label_weight(self, fm):
        batch = fm.shape[0]
        weight_row = self.distance_func(fm[:, :, :-1, :], fm[:, :, 1:, :])
        weight_col = self.distance_func(fm[:, :, :, :-1], fm[:, :, :, 1:])
        weight_row = weight_row.reshape([batch, -1])
        weight_col = weight_col.reshape([batch, -1])
        diff_weight = torch.cat([weight_row, weight_col], dim=1)

        weight_row = (fm[:, :, :-1, :] + fm[:, :, 1:, :]).sum(1)
        weight_col = (fm[:, :, :, :-1] + fm[:, :, :, 1:]).sum(1)
        weight_row = weight_row.reshape([batch, -1])
        weight_col = weight_col.reshape([batch, -1])
        labeled_weight = torch.cat([weight_row, weight_col], dim=1)

        weight = diff_weight * labeled_weight
        return weight

    def forward(self, guide_in, label=None):
        with torch.no_grad():
            index = self._build_matrix_index(guide_in)
            weight = self._build_feature_weight(guide_in)
            if label is not None:
                label_weight = self._build_label_weight(label)
                label_idx = (label_weight > 0)
                weight[label_idx] = torch.sigmoid(weight[label_idx])
            tree = mst(index, weight, guide_in.shape[2] * guide_in.shape[3])
        return tree


class TreeFilter2D(nn.Module):
    def __init__(self, groups=1, sigma=0.02, distance_func=None, enable_log=False):
        super(TreeFilter2D, self).__init__()
        self.groups = groups
        self.enable_log = enable_log
        if distance_func is None:
            self.distance_func = self.norm2_distance
        else:
            self.distance_func = distance_func

        self.sigma = sigma

    @staticmethod
    def norm2_distance(fm_ref, fm_tar):
        diff = fm_ref - fm_tar
        weight = (diff * diff).sum(dim=1)
        return weight

    @staticmethod
    def batch_index_opr(data, index):
        with torch.no_grad():
            channel = data.shape[1]
            index = index.unsqueeze(1).expand(-1, channel, -1).long()
        data = torch.gather(data, 2, index)
        return data

    def build_edge_weight(self, fm, sorted_index, sorted_parent, low_tree):
        batch   = fm.shape[0]
        channel = fm.shape[1]
        vertex  = fm.shape[2] * fm.shape[3]

        fm = fm.reshape([batch, channel, -1])
        fm_source = self.batch_index_opr(fm, sorted_index)
        fm_target = self.batch_index_opr(fm_source, sorted_parent)
        fm_source = fm_source.reshape([-1, channel // self.groups, vertex])
        fm_target = fm_target.reshape([-1, channel // self.groups, vertex])

        edge_weight = self.distance_func(fm_source, fm_target)
        if low_tree:
            edge_weight = torch.exp(-edge_weight / self.sigma)
        else:
            edge_weight = torch.exp(-edge_weight)

        return edge_weight

    def split_group(self, feature_in, *tree_orders):
        feature_in = feature_in.reshape(feature_in.shape[0] * self.groups,
                                        feature_in.shape[1] // self.groups,
                                        -1)
        returns = [feature_in.contiguous()]
        for order in tree_orders:
            order = order.unsqueeze(1).expand(order.shape[0], self.groups, *order.shape[1:])
            order = order.reshape(-1, *order.shape[2:])
            returns.append(order.contiguous())
        return tuple(returns)

    def print_info(self, edge_weight):
        edge_weight = edge_weight.clone()
        info = torch.stack([edge_weight.mean(), edge_weight.std(), edge_weight.max(), edge_weight.min()])
        if self.training and dist.is_initialized():
            dist.all_reduce(info / dist.get_world_size())
            info_str = (float(x) for x in info)
            if dist.get_rank() == 0:
                print('Mean:{0:.4f}, Std:{1:.4f}, Max:{2:.4f}, Min:{3:.4f}'.format(*info_str))
        else:
            info_str = [float(x) for x in info]
            print('Mean:{0:.4f}, Std:{1:.4f}, Max:{2:.4f}, Min:{3:.4f}'.format(*info_str))

    def forward(self, feature_in, embed_in, tree, low_tree=True):
        ori_shape = feature_in.shape
        sorted_index, sorted_parent, sorted_child = bfs(tree, 4)

        edge_weight = self.build_edge_weight(embed_in, sorted_index, sorted_parent, low_tree)

        self.enable_log = False
        with torch.no_grad():
            if self.enable_log:
                self.print_info(edge_weight)

        feature_in, sorted_index, sorted_parent, sorted_child = \
            self.split_group(feature_in, sorted_index, sorted_parent, sorted_child)

        feature_out = refine(feature_in, edge_weight, sorted_index,
                             sorted_parent, sorted_child, low_tree)

        feature_out = feature_out.reshape(ori_shape)
        return feature_out