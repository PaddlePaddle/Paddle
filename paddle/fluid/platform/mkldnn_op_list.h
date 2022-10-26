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
    // kernels and codes are deleted in the future, attributes "use_mkldnn"
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
    // NOTE(jiahongyu): Below ops register kernel with customized_type_value, we
    // need to analysis and solve them one-by-one.
    "prior_box"};

static const std::unordered_set<std::string> cudnn_white_list = {
    // cudnn general ops
    "affine_grid",
    "affine_grid_grad",
    "conv2d_transpose",
    "conv2d_transpose_grad",
    "conv2d_transpose_grad_grad",
    "conv3d_transpose",
    "conv3d_transpose_grad",
    "depthwise_conv2d_transpose",
    "depthwise_conv2d_transpose_grad",
    "grid_sampler",
    "grid_sampler_grad",
    "pool2d",
    "pool2d_grad",
    "pool2d_double_grad",
    "pool3d",
    "pool3d_grad",
    "softmax",
    "softmax_grad",

    // WIP
    "conv2d",
    "conv2d_grad",
    "conv2d_grad_grad",
    "depthwise_conv2d",
    "depthwise_conv2d_grad",
    "depthwise_conv2d_grad_grad",
    "conv3d",
    "conv3d_grad",
    "conv3d_grad_grad",

    // activation mkldnn operator
    "soft_relu",
    "soft_relu_grad",
    "cos",
    "cos_grad",
    "tan",
    "tan_grad",
    "acos",
    "acos_grad",
    "sin",
    "sin_grad",
    "asin",
    "asin_grad",
    "atan",
    "atan_grad",
    "sinh",
    "sinh_grad",
    "cosh",
    "cosh_grad",
    "asinh",
    "asinh_grad",
    "acosh",
    "acosh_grad",
    "atanh",
    "atanh_grad",
    "brelu",
    "brelu_grad",
    "thresholded_relu",
    "thresholded_relu_grad",
    "relu6",
    "relu6_grad",
    "hard_shrink",
    "hard_shrink_grad",
    "softshrink",
    "softshrink_grad",
    "tanh_shrink",
    "tanh_shrink_grad",
    "silu",
    "silu_grad",
    "softsign",
    "softsign_grad",
    "hard_sigmoid",
    "hard_sigmoid_grad",
    "logsigmoid",
    "logsigmoid_grad",
    "expm1",
    "expm1_grad",
    "softplus",
    "softplus_grad",
    "mish",
    "mish_grad",
    "stanh",
    "stanh_grad",
    "reciprocal",
    "reciprocal_grad",
    "log2",
    "log2_grad",
    "log10",
    "log10_grad",
    "log1p",
    "log1p_grad",
    "hard_swish",
    "hard_swish_grad",
    "swish",
    "swish_grad",
    "round",
    "round_grad",
    "floor",
    "floor_grad",
    "ceil",
    "ceil_grad",
    "sigmoid",
    "sigmoid_grad",
    "sigmoid_grad_grad",
    "sigmoid_triple_grad",
    "tanh",
    "tanh_grad",
    "tanh_grad_grad",
    "tanh_triple_grad",
    "relu",
    "relu_grad",
    "relu_grad_grad",
    "leaky_relu",
    "leaky_relu_grad",
    "leaky_relu_grad_grad",
    "elu",
    "elu_grad",
    "elu_grad_grad",
    "celu",
    "celu_grad",
    "celu_grad_grad",
    "logit",
    "logit_grad",
    "sqrt",
    "sqrt_grad",
    "sqrt_grad_grad",
    "rsqrt",
    "rsqrt_grad",
    "rsqrt_grad_grad",
    "square",
    "square_grad",
    "square_grad_grad",
    "pow",
    "pow_grad",
    "exp",
    "exp_grad",
    "log",
    "log_grad",
    "log_grad_grad"};

inline bool in_mkldnn_white_list(const std::string& op_name) {
  return mkldnn_white_list.find(op_name) != mkldnn_white_list.end();
}

inline bool in_cudnn_white_list(const std::string& op_name) {
  return cudnn_white_list.find(op_name) != cudnn_white_list.end();
}

}  // namespace platform
}  // namespace paddle
#endif
