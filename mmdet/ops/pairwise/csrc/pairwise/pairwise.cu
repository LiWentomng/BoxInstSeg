#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/types.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace pairwise_kernel {
long const threadsPerBlock = 512;
long const maxGridDim = 50000;

__device__ __forceinline__ double _exp(double x) {
  return exp(x);
}

__device__ __forceinline__ float _exp(float x) {
  return expf(x);
}

__device__ __forceinline__ double _logsig(double x) {
  if (x < -40.0)
    return x;
  return -log(1.0 + exp(-x));
}

__device__ __forceinline__ float _logsig(float x) {
  if (x < -20.0f)
    return x;
  return -logf(1.0f + expf(-x));
}
template <class T>
__device__ void pairwise_nlog_forward(const T x, const T y, bool is_y_pad,
                                      T* pair) {
  T ls_px = _logsig(x);
  T ls_mx = _logsig(-x);
  T ls_py = is_y_pad ? 0 : _logsig(y);
  T ls_my = is_y_pad ? 0 : _logsig(-y);
  T ls_eq1 = ls_px + ls_py;
  T ls_eq0 = ls_mx + ls_my;
  T ls_eq_max = ls_eq1 > ls_eq0 ? ls_eq1 : ls_eq0;
  T ls_eq_diff = ls_eq1 > ls_eq0 ? ls_eq1 - ls_eq0 : ls_eq0 - ls_eq1;
  *pair = _logsig(ls_eq_diff) - ls_eq_max;
}

template <class T>
__device__ void pairwise_nlog_backward(const T x, const T y, bool is_y_pad,
                                       const T pair, T* g_x, T* g_y,
                                       const T g_pair) {
  T ls_px = _logsig(x);
  T ls_mx = _logsig(-x);
  T ls_py = is_y_pad ? 0 : _logsig(y);
  T ls_my = is_y_pad ? 0 : _logsig(-y);
  T _g_x = -(_exp(ls_py) - _exp(ls_my)) * _exp(ls_px + ls_mx + pair) * g_pair;
  T _g_y = -(_exp(ls_px) - _exp(ls_mx)) * _exp(ls_py + ls_my + pair) * g_pair;
  if (g_x)
    atomicAdd(g_x, _g_x);
  if (g_y)
    atomicAdd(g_y, _g_y);
}

template <class T>
__global__ void pairwise_nlog_forward_kernel(
    const int pairwise_size, const int pairwise_dilation,
    const torch::PackedTensorAccessor32<T, 4> logits,
    torch::PackedTensorAccessor32<T, 4> pairwise) {
  const int B = logits.size(0);
  const int H = logits.size(2);
  const int W = logits.size(3);

  const int R = pairwise_size / 2 * pairwise_dilation;
  for (int _idx = blockIdx.x * blockDim.x + threadIdx.x; _idx < B * H * W;
       _idx += gridDim.x * blockDim.x) {
    int _tmp = _idx;
    int x = _tmp % W;
    _tmp /= W;
    int y = _tmp % H;
    _tmp /= H;
    int b = _tmp;
    int count = 0;
    for (int dy = -R; dy <= R; dy += pairwise_dilation) {
      for (int dx = -R; dx <= R; dx += pairwise_dilation) {
        if (dx == 0 && dy == 0)
          continue;
        int x2 = x + dx;
        int y2 = y + dy;
        T here = logits[b][0][y][x];
        T there = 0;
        bool valid_there = (x2 >= 0 && x2 < W && y2 >= 0 && y2 < H);
        if (valid_there)
          there = logits[b][0][y2][x2];
        pairwise_nlog_forward(here, there, !valid_there,
                              &pairwise[b][count][y][x]);
        count++;
      }
    }
  }
}

template <class T>
__global__ void pairwise_nlog_backward_kernel(
    const int pairwise_size, const int pairwise_dilation,
    torch::PackedTensorAccessor32<T, 4> logits,
    torch::PackedTensorAccessor32<T, 4> pairwise,
    torch::PackedTensorAccessor32<T, 4> g_logits,
    torch::PackedTensorAccessor32<T, 4> g_pairwise) {
  const int B = logits.size(0);
  const int H = logits.size(2);
  const int W = logits.size(3);

  const int R = pairwise_size / 2 * pairwise_dilation;
  for (int _idx = blockIdx.x * blockDim.x + threadIdx.x; _idx < B * H * W;
       _idx += gridDim.x * blockDim.x) {
    int _tmp = _idx;
    int x = _tmp % W;
    _tmp /= W;
    int y = _tmp % H;
    _tmp /= H;
    int b = _tmp;
    int count = 0;
    for (int dy = -R; dy <= R; dy += pairwise_dilation) {
      for (int dx = -R; dx <= R; dx += pairwise_dilation) {
        if (dx == 0 && dy == 0)
          continue;
        int x2 = x + dx;
        int y2 = y + dy;
        T here = logits[b][0][y][x];
        T* g_here = &g_logits[b][0][y][x];
        T there = 0;
        T* g_there = nullptr;
        bool valid_there = (x2 >= 0 && x2 < W && y2 >= 0 && y2 < H);
        if (valid_there) {
          there = logits[b][0][y2][x2];
          g_there = &g_logits[b][0][y2][x2];
        }
        pairwise_nlog_backward(here, there, !valid_there,
                               pairwise[b][count][y][x], g_here, g_there,
                               g_pairwise[b][count][y][x]);
        count++;
      }
    }
  }
}

}  // namespace pairwise_kernel

namespace pairwise {
torch::Tensor pairwiseNLogForwardCUDALauncher(const int pairwise_size,
                                              const int pairwise_dilation,
                                              torch::Tensor& logits) {
  CHECK_INPUT(logits);
  int pair_count = pairwise_size * pairwise_size - 1;
  auto pairwise =
      torch::empty({logits.size(0), pair_count, logits.size(2), logits.size(3)},
                   logits.options());
  AT_DISPATCH_FLOATING_TYPES(
      logits.scalar_type(), "pairwise_nlog_forward_kernel", ([&]() {
        pairwise_kernel::pairwise_nlog_forward_kernel<scalar_t>
            <<<std::min(at::cuda::ATenCeilDiv(logits.numel() / logits.size(1),
                                              pairwise_kernel::threadsPerBlock),
                        pairwise_kernel::maxGridDim),
               pairwise_kernel::threadsPerBlock>>>(
                pairwise_size, pairwise_dilation,
                logits.packed_accessor32<scalar_t, 4>(),
                pairwise.packed_accessor32<scalar_t, 4>());
      }));
  AT_CUDA_CHECK(cudaGetLastError());
  return pairwise;
}

torch::Tensor pairwiseNLogBackwardCUDALauncher(const int pairwise_size,
                                               const int pairwise_dilation,
                                               torch::Tensor& logits,
                                               torch::Tensor& pairwise,
                                               torch::Tensor& g_pairwise) {

  CHECK_INPUT(logits);
  CHECK_INPUT(pairwise);
  CHECK_INPUT(g_pairwise);
  auto g_logits = torch::zeros_like(logits);
  AT_DISPATCH_FLOATING_TYPES(
      g_pairwise.scalar_type(), "pairwise_nlog_backward_kernel", ([&]() {
        pairwise_kernel::pairwise_nlog_backward_kernel<scalar_t>
            <<<std::min(at::cuda::ATenCeilDiv(logits.numel() / logits.size(1),
                                              pairwise_kernel::threadsPerBlock),
                        pairwise_kernel::maxGridDim),
               pairwise_kernel::threadsPerBlock>>>(
                pairwise_size, pairwise_dilation,
                logits.packed_accessor32<scalar_t, 4>(),
                pairwise.packed_accessor32<scalar_t, 4>(),
                g_logits.packed_accessor32<scalar_t, 4>(),
                g_pairwise.packed_accessor32<scalar_t, 4>());
      }));
  AT_CUDA_CHECK(cudaGetLastError());
  return g_logits;
}
}  // namespace pairwise
