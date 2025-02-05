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
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/placement_types.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"

paddle::Tensor add_n_ad_func(const std::vector<paddle::Tensor>& x);

paddle::Tensor conv2d_ad_func(const paddle::Tensor& input,
                              const paddle::Tensor& filter,
                              std::vector<int> strides,
                              std::vector<int> paddings,
                              std::string padding_algorithm,
                              std::vector<int> dilations,
                              int groups,
                              std::string data_format);

paddle::Tensor multiply_ad_func(const paddle::Tensor& x,
                                const paddle::Tensor& y);
paddle::Tensor& multiply__ad_func(paddle::Tensor& x,  // NOLINT
                                  const paddle::Tensor& y);

std::tuple<paddle::Tensor,
           paddle::Tensor&,
           paddle::Tensor&,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor>
sync_batch_norm__ad_func(const paddle::Tensor& x,
                         paddle::Tensor& mean,      // NOLINT
                         paddle::Tensor& variance,  // NOLINT
                         const paddle::Tensor& scale,
                         const paddle::Tensor& bias,
                         bool is_test,
                         float momentum,
                         float epsilon,
                         std::string data_layout,
                         bool use_global_stats,
                         bool trainable_statistics);

paddle::Tensor reshard_ad_function(
    const paddle::Tensor& tensor,
    const phi::distributed::TensorDistAttr dist_attr);

paddle::Tensor dtensor_to_local_ad_function(const paddle::Tensor& input);

paddle::Tensor dtensor_from_local_ad_function(
    const paddle::Tensor& input,
    const phi::distributed::ProcessMesh& processmesh,
    const phi::distributed::Placements& placements);

namespace sparse {
std::tuple<paddle::Tensor,
           paddle::Tensor&,
           paddle::Tensor&,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor>
sync_batch_norm__ad_func(const paddle::Tensor& x,
                         paddle::Tensor& mean,      // NOLINT
                         paddle::Tensor& variance,  // NOLINT
                         const paddle::Tensor& scale,
                         const paddle::Tensor& bias,
                         bool is_test,
                         float momentum,
                         float epsilon,
                         std::string data_layout,
                         bool use_global_stats,
                         bool trainable_statistics);

paddle::Tensor multiply_ad_func(const paddle::Tensor& x,
                                const paddle::Tensor& y);

}  // namespace sparse
