/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

static const std::unordered_set<std::string> mkldnn_white_list = {
    "cast",
    "transfer_dtype",
    "conv2d_transpose",
    "depthwise_conv2d_transpose",
    "conv3d_transpose",
    "layer_norm",
    "mul",
    "pad2d",
    "pad3d",
    "pool2d",
    "pool2d_grad",
    "pool2d_double_grad",
    "pool3d",
    "pool3d_grad",
    "slice",
    "slice_grad",
    "softmax",
    "softmax_grad",
    "split",
    "sum",
    "transpose2_grad",
    "sgd",
    // NOTE(jiahy0825): squeeze MKLDNN kernel are disabled
    // (https://github.com/PaddlePaddle/Paddle/pull/35781). If these MKLDNN
    // kernels and codes are deleted in the future, attributes `use_mkldnn`
    // should be removed from function declaration
    "squeeze",
    "squeeze_grad",
    "squeeze2",
    "squeeze2_grad",
    // NOTE(jiahy0825): After fixing GetExpectedKernelType in ReduceOp, reduce
    // series hard code can be deleted together.
    "frobenius_norm",
    "reduce_amax",
    "reduce_amin",
    "reduce_max",
    "reduce_mean",
    "reduce_min",
    "reduce_prod",
    "reduce_sum",
    "frobenius_norm_grad",
    "reduce_amax_grad",
    "reduce_amin_grad",
    "reduce_max_grad",
    "reduce_mean_grad",
    "reduce_min_grad",
    "reduce_prod_grad",
    "reduce_sum_grad",
    // NOTE(jiahy0825): Below ops register kernel with customized_type_value, we
    // need to analysis and solve them one-by-one.
    "conv2d",
    "conv2d_grad",
    "depthwise_conv2d",
    "depthwise_conv2d_grad",
    "conv3d",
    "conv3d_grad",
    "prior_box",
    "fc",
    "mul",
    "mul_grad",
    "transpose2"};

inline bool in_mkldnn_white_list(const std::string& op_name) {
  return mkldnn_white_list.find(op_name) != mkldnn_white_list.end();
}

}  // namespace platform
}  // namespace paddle
#endif
