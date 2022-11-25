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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/selected_rows.h"

namespace phi {
namespace sr {

template <typename T, typename Context>
void HierarchicalSigmoidGradKernel(const Context& ctx,
                                   const DenseTensor& x,
                                   const DenseTensor& w,
                                   const DenseTensor& label,
                                   const paddle::optional<DenseTensor>& path,
                                   const paddle::optional<DenseTensor>& code,
                                   const paddle::optional<DenseTensor>& bias,
                                   const DenseTensor& pre_out,
                                   const DenseTensor& out_grad,
                                   int num_classes,
                                   bool remote_prefetch,
                                   int trainer_id,
                                   const std::vector<int64_t>& height_sections,
                                   const std::vector<std::string>& epmap,
                                   const std::vector<std::string>& table_names,
                                   bool is_sparse,
                                   DenseTensor* x_grad,
                                   SelectedRows* w_grad,
                                   DenseTensor* bias_grad);

}  // namespace sr
}  // namespace phi
