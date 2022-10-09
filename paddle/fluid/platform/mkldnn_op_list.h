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
    "abs",
    "abs_grad",
    "batch_norm",
    "batch_norm_grad",
    "clip",
    "clip_grad",
    "concat",
    "concat_grad",
    "data_norm",
    "data_norm_grad",
    "expand_v2",
    "expand_v2_grad",
    "fill_constant",
    "fusion_gru",
    "fusion_lstm",
    "gaussian_random",
    "log_softmax",
    "lrn",
    "lrn_grad",
    "matmul",
    "matmul_grad",
    "matmul_v2",
    "matmul_v2_grad",
    "scale",
    "shape",
    "shuffle_channel",
    "stack",
    "transpose",
    "transpose_grad",
    "transpose2_grad",
    "elementwise_add",
    "elementwise_add_grad",
    "elementwise_sub",
    "elementwise_sub_grad",
    "elementwise_mul",
    "elementwise_mul_grad",
    "elementwise_div",
    "elementwise_div_grad",
    "gelu",
    "gelu_grad",
    "prelu",
    "prelu_grad",
    "nearest_interp",
    "bilinear_interp",
    "nearest_interp_v2",
    "bilinear_interp_v2"};

inline bool in_mkldnn_white_list(const std::string& op_name) {
  return mkldnn_white_list.find(op_name) == mkldnn_white_list.end();
}

}  // namespace platform
}  // namespace paddle
#endif
