// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <Python.h>
#include "pybind11/functional.h"
#include "pybind11/stl.h"

#include "paddle/fluid/distributed/collective/marlin/moe_ops.h"
#include "paddle/fluid/pybind/moe_wna16_marlin_api.h"
#include "paddle/utils/pybind.h"

namespace py = pybind11;

namespace paddle::pybind {

void BindMoeWna16MarlinApi(pybind11::module *m) {
m->def(
    "moe_wna16_marlin_gemm",
    &moe_wna16_marlin_gemm_api,
    return_value_policy::move,
    py::arg("a"),
    py::arg("c_or_none") = std::nullopt,
    py::arg("b_q_weight"),
    py::arg("b_scales"),
    py::arg("global_scale_or_none") = std::nullopt,
    py::arg("b_zeros_or_none") = std::nullopt,
    py::arg("g_idx_or_none") = std::nullopt,
    py::arg("perm_or_none") = std::nullopt,
    py::arg("workspace"),
    py::arg("sorted_token_ids"),
    py::arg("expert_ids"),
    py::arg("num_tokens_past_padded"),
    py::arg("topk_weights"),
    py::arg("moe_block_size"),
    py::arg("top_k"),
    py::arg("mul_topk_weights"),
    py::arg("is_ep"),
    py::arg("b_q_type_id"),
    py::arg("size_m"),
    py::arg("size_n"),
    py::arg("size_k"),
    py::arg("is_k_full"),
    py::arg("use_atomic_add"),
    py::arg("use_fp32_reduce"),
    py::arg("is_zp_float"));

}

}  // namespace paddle::pybind
