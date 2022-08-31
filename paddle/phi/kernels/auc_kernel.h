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

#include <string>

#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void AucKernel(const Context& dev_ctx,
               const DenseTensor& input,
               const DenseTensor& label,
               const DenseTensor& stat_pos,
               const DenseTensor& stat_neg,
               const paddle::optional<DenseTensor>& ins_tag_weight,
               const std::string& curve,
               int num_thresholds,
               int slide_steps,
               DenseTensor* auc,
               DenseTensor* stat_pos_out,
               DenseTensor* stat_neg_out);

}  // namespace phi
