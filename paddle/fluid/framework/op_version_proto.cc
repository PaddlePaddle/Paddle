/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_version_proto.h"

namespace paddle::framework::compatible::pb {
const std::unordered_map<std::string, uint32_t>& GetLegacyOpVersions() {
  static std::unordered_map<std::string, uint32_t> op_versions = {
      {"not_equal", 1},
      {"assign_value", 0},
      {"fake_channel_wise_dequantize_max_abs", 2},
      {"yolo_box", 1},
      {"data_norm", 1},
      {"cumsum", 1},
      {"fake_channel_wise_quantize_abs_max", 1},
      {"greater_equal", 1},
      {"fill_constant", 2},
      {"conv_transpose", 1},
      {"fusion_gru", 1},
      {"flip", 1},
      {"elementwise_sub", 1},
      {"dequantize", 1},
      {"grid_sampler", 1},
      {"expand_as_v2", 1},
      {"linspace", 1},
      {"moving_average_abs_max_scale", 2},
      {"p_norm", 1},
      {"instance_norm", 1},
      {"lookup_table_v2", 1},
      {"seed", 1},
      {"softmax_with_cross_entropy", 1},
      {"rank_attention", 1},
      {"cudnn_lstm", 1},
      {"clip", 1},
      {"requantize", 1},
      {"for_pybind_test__", 4},
      {"print", 1},
      {"transfer_layout", 1},
      {"arg_min", 1},
      {"roll", 2},
      {"roi_pool", 2},
      {"conv2d_transpose", 2},
      {"roi_align", 3},
      {"softplus", 1},
      {"momentum", 1},
      {"trace", 1},
      {"matmul", 1},
      {"lookup_table", 1},
      {"lstsq", 1},
      {"conv3d_transpose", 1},
      {"depthwise_conv2d_transpose", 1},
      {"conv2d", 1},
      {"lamb", 1},
      {"send_and_recv", 1},
      {"gaussian_random", 1},
      {"unique_consecutive", 1},
      {"conv3d", 1},
      {"pixel_shuffle", 1},
      {"collect_fpn_proposals", 1},
      {"coalesce_tensor", 2},
      {"arg_max", 1},
      {"allclose", 2},
      {"matrix_nms", 1},
      {"less_than", 1},
      {"affine_grid", 1},
      {"hard_shrink", 1},
      {"set_value", 3},
      {"mish", 1},
      {"quantize", 2},
      {"distribute_fpn_proposals", 2},
      {"adam", 4},
      {"elementwise_pow", 1},
      {"elementwise_mul", 1},
      {"elementwise_mod", 1},
      {"auc", 1},
      {"elementwise_min", 1},
      {"elementwise_max", 1},
      {"gather", 1},
      {"elementwise_div", 1},
      {"elementwise_add", 1},
      {"leaky_relu", 1},
      {"generate_proposal_labels", 2},
      {"elementwise_floordiv", 1},
      {"less_equal", 1},
      {"generate_proposals", 2},
      {"depthwise_conv2d", 1},
      {"greater_than", 1},
      {"generate_proposals_v2", 1},
      {"equal", 1}};
  return op_versions;
}
}  // namespace paddle::framework::compatible::pb
