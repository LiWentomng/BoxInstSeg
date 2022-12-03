import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

import tree_filter_cuda as _C

class _BFS(Function):
    @staticmethod
    def forward(ctx, edge_index, max_adj_per_vertex):
        sorted_index, sorted_parent, sorted_child =\
                _C.bfs_forward(edge_index, max_adj_per_vertex)
        return sorted_index, sorted_parent, sorted_child

bfs = _BFS.apply

