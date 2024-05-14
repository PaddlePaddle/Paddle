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

#include <string.h>
#include <vector>
#include "paddle/phi/core/dense_tensor.h"
namespace phi {

template <typename T, typename Context>
void CTCAlignKernel(const Context& dev_ctx,
                    const DenseTensor& input,
                    const paddle::optional<DenseTensor>& input_length,
                    int blank,
                    bool merge_repeated,
                    int padding_value,
                    DenseTensor* output,
                    DenseTensor* output_length);
}  // namespace phi
