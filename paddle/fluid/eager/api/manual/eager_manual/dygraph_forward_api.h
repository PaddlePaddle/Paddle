// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/phi/api/include/tensor.h"

paddle::Tensor add_n_ad_func(const std::vector<paddle::Tensor>& x);

paddle::Tensor conv2d_ad_func(const paddle::Tensor& input,
                              const paddle::Tensor& filter,
                              std::vector<int> strides,
                              std::vector<int> paddings,
                              std::string padding_algorithm,
                              std::vector<int> dilations,
                              int groups,
                              std::string data_format);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
sync_batch_norm__ad_func(const paddle::experimental::Tensor& x,
                         const paddle::experimental::Tensor& scale,
                         const paddle::experimental::Tensor& bias,
                         paddle::experimental::Tensor& mean,      // NOLINT
                         paddle::experimental::Tensor& variance,  // NOLINT
                         float momentum,
                         float epsilon,
                         std::string data_layout,
                         bool is_test,
                         bool use_global_stats,
                         bool trainable_statistics,
                         bool fuse_with_relu);

namespace sparse {
std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
sync_batch_norm__ad_func(const paddle::experimental::Tensor& x,
                         const paddle::experimental::Tensor& scale,
                         const paddle::experimental::Tensor& bias,
                         paddle::experimental::Tensor& mean,      // NOLINT
                         paddle::experimental::Tensor& variance,  // NOLINT
                         float momentum,
                         float epsilon,
                         std::string data_layout,
                         bool is_test,
                         bool use_global_stats,
                         bool trainable_statistics,
                         bool fuse_with_relu);
}  // namespace sparse
