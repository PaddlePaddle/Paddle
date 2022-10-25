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
    "cast",
    "transfer_dtype",
    "layer_norm",
    "pad2d",
    "pad3d",
    "pool2d",
    "pool2d_grad",
    "slice",
    "slice_grad",
    "split",
    "sum",
    "sgd",
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
    "flatten2_grad",
    // NOTE(jiahongyu): After fixing GetExpectedKernelType in ReduceOp, reduce
    // series hard code can be deleted together.
    "reduce_max",
    "reduce_mean",
    "reduce_mean_grad",
    "reduce_min",
    "reduce_sum",
    "reduce_sum_grad",
    // NOTE(jiahongyu): Below ops register kernel with customized_type_value, we
    // need to analysis and solve them one-by-one.
    "prior_box"};

inline bool in_mkldnn_white_list(const std::string& op_name) {
  return mkldnn_white_list.find(op_name) != mkldnn_white_list.end();
}

}  // namespace platform
}  // namespace paddle
#endif
