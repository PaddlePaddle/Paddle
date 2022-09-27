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

#include <string>
#include <vector>

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void BoxCoderKernel(const Context& dev_ctx,
                    const DenseTensor& prior_box,
                    const paddle::optional<DenseTensor>& prior_box_var,
                    const DenseTensor& target_box,
                    const std::string& code_type,
                    bool box_normalized,
                    int axis,
                    const std::vector<float>& variance,
                    DenseTensor* output_box);
}  // namespace phi
