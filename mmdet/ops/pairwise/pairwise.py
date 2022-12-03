import torch
from torch.autograd import Function
from .pairwise_ext import pairwise_nlog_forward, pairwise_nlog_backward


class _pairwise_nlog(Function):
    @staticmethod
    def forward(ctx, logits, pairwise_size, pairwise_dilation):
        logits = logits.contiguous()
        pairwise = pairwise_nlog_forward(
            pairwise_size, pairwise_dilation, logits)
        ctx.pairwise_size = pairwise_size
        ctx.pairwise_dilation = pairwise_dilation
        ctx.save_for_backward(logits, pairwise)
        return pairwise

    @staticmethod
    def backward(ctx, g_pairwise):
        g_pairwise = g_pairwise.contiguous()
        logits, pairwise = ctx.saved_tensors
        g_logits = pairwise_nlog_backward(
            ctx.pairwise_size, ctx.pairwise_dilation, logits, pairwise, g_pairwise)
        return g_logits, None, None


pairwise_nlog = _pairwise_nlog.apply
