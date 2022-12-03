#pragma once
#include <torch/extension.h>
#include "refine/refine.hpp"
#include "mst/mst.hpp"
#include "bfs/bfs.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mst_forward", &mst_forward, "mst forward");
    m.def("bfs_forward", &bfs_forward, "bfs forward");
    m.def("refine_forward", &refine_forward, "refine forward");
    m.def("refine_backward_feature", &refine_backward_feature, "refine backward wrt feature");
    m.def("refine_backward_weight", &refine_backward_weight, "refine backward wrt weight");
}

