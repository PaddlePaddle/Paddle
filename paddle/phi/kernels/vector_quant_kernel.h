// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void VectorQuantKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const float epsilon,
                       const bool return_transpose,
                       const bool pow2_scale,
                       const bool permute_scale,
                       const bool transpose_scales,
                       DenseTensor* scale_inv,
                       DenseTensor* scale_inv_t,
                       DenseTensor* output,
                       DenseTensor* output_t);
}  // namespace phi
