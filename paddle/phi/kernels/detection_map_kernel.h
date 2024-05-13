// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/dense_tensor.h"
namespace phi {

template <typename T, typename Context>
void DetectionMAPOpKernel(const Context& dev_ctx,
                          const DenseTensor& detect_res,
                          const DenseTensor& label,
                          const paddle::optional<DenseTensor>& has_state,
                          const paddle::optional<DenseTensor>& pos_count,
                          const paddle::optional<DenseTensor>& true_pos,
                          const paddle::optional<DenseTensor>& false_pos,
                          int class_num,
                          int background_label,
                          float overlap_threshold,
                          bool evaluate_difficult,
                          const std::string& ap_type,
                          DenseTensor* accum_pos_count,
                          DenseTensor* accum_true_pos,
                          DenseTensor* accum_false_pos,
                          DenseTensor* m_ap);

}  // namespace phi
