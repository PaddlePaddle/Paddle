/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_MKLDNN

#include <unordered_set>

namespace paddle {
namespace platform {

// NOTE(jiahongyu): Below ops have specific PADDLE_WITH_MKLDNN hard codes within
// the function GetExpectedKernelType, so we need to handle them through
// mkldnn_white_list and solve them one-by-one in the future.
// TODO(jiahongyu): Delete mkldnn_white_list and fully support
// PADDLE_WITH_MKLDNN of GetExpectedKernelType.
static const std::unordered_set<std::string> mkldnn_white_list = {
    // NOTE(jiahongyu): Below ops use mem_desc function, which is encoded by
    // PADDLE_WITH_MKLDNN in DenseTensor. The hardcodes within
    // GetExpectedKernelType of these ops cannot be deleted now.
    "pad2d",
    "pad3d",
    "slice",
    "slice_grad",
    "split",
    // NOTE(jiahongyu): squeeze MKLDNN kernel are disabled
    // (https://github.com/PaddlePaddle/Paddle/pull/35781). If these MKLDNN
    // kernels and codes are deleted in the future, attributes `use_mkldnn`
    // should be removed from function declaration
    "squeeze",
    "squeeze_grad",
    "squeeze2",
    "squeeze2_grad",
    // NOTE(jiahongyu): reshape and flatten have attribute use_mkldnn and they
    // are registered in paddle, but they didn't change the ExpectedKernelType
    // of tensor. Actually, mkldnn kernel of squeeze, reshape, and flatten
    // should never be called.
    "reshape",
    "reshape_grad",
    "reshape2",
    "reshape2_grad",
    "flatten",
    "flatten_grad",
    "flatten2",
    "flatten2_grad"};

/**
 * NOTE: [ Why using phi_mkldnn_white_list? ]
 * This list records MKLDNN kernels that are supported in phi. The process of
 * fully supporting MKLDNN kernels are divided into two steps:
 * 1. Enlarge phi_mkldnn_white_list: larger phi_mkldnn_white_list means phi
 * supports more mkldnn kernels.
 * 2. Delete phi_mkldnn_white_list: after phi supports all mkldnn kernels, this
 * list can be deleted safely because MKLDNN kernel selection information has
 * already contained in phi::DenseTensorMeta.
 */
static const std::unordered_set<std::string> phi_mkldnn_white_list = {
    "acosh",      "addmm", "asinh", "atanh",    "ceil",  "cos",
    "cosh",       "elu",   "exp",   "expm1",    "gelu",  "hard_swish",
    "leaky_relu", "log",   "log10", "log1p",    "log2",  "log_softmax",
    "logsigmoid", "prelu", "relu",  "relu6",    "rsqrt", "sigmoid",
    "silu",       "sin",   "sinh",  "softsign", "sqrt",  "tan",
    "tanh_shrink"};

inline bool in_mkldnn_white_list(const std::string& op_name) {
  return mkldnn_white_list.find(op_name) != mkldnn_white_list.end();
}

inline bool in_phi_mkldnn_white_list(const std::string& op_name) {
  return phi_mkldnn_white_list.find(op_name) != phi_mkldnn_white_list.end();
}

}  // namespace platform
}  // namespace paddle
#endif
