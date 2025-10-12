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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void FP8QuantBlockWiseKernel(const Context& dev_ctx,
                             const DenseTensor& X,
                             float epsilon,
                             bool using_1x128_vec_quant,
                             bool input_transpose,
                             bool output_scale_transpose,
                             bool return_transpose_only,
                             bool using_e5m2,
                             bool using_pow2_scale,
                             DenseTensor* out,
                             DenseTensor* scale,
                             DenseTensor* out_transposed,
                             DenseTensor* scale_transposed);

}  // namespace phi
