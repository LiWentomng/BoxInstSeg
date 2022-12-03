#include <torch/extension.h>

namespace pairwise {

torch::Tensor pairwiseNLogForwardCUDALauncher(const int pairwise_size,
                                              const int pairwise_dilation,
                                              torch::Tensor& logits);

torch::Tensor pairwiseNLogBackwardCUDALauncher(const int pairwise_size,
                                               const int pairwise_dilation,
                                               torch::Tensor& logits,
                                               torch::Tensor& pairwise,
                                               torch::Tensor& g_pairwise);

torch::Tensor pairwise_nlog_forward(const int pairwise_size,
                                    const int pairwise_dilation,
                                    torch::Tensor& logits) {
  return pairwiseNLogForwardCUDALauncher(pairwise_size, pairwise_dilation,
                                         logits);
}

torch::Tensor pairwise_nlog_backward(const int pairwise_size,
                                     const int pairwise_dilation,
                                     torch::Tensor& logits,
                                     torch::Tensor& pairwise,
                                     torch::Tensor& g_pairwise) {
  return pairwiseNLogBackwardCUDALauncher(pairwise_size, pairwise_dilation,
                                          logits, pairwise, g_pairwise);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pairwise_nlog_forward", &pairwise_nlog_forward,
        "pairwise_nlog_forward");
  m.def("pairwise_nlog_backward", &pairwise_nlog_backward,
        "pairwise_nlog_backward");
}

}  // namespace pairwise
