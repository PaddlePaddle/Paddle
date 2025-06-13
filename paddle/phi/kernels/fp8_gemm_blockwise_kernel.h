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
#include "paddle/phi/core/device_context.h"

namespace phi {

template <typename T, typename Context>
void Fp8GemmBlockwiseKernel(const Context& dev_ctx,
                            const DenseTensor& A,
                            const DenseTensor& A_scale,
                            const DenseTensor& B,
                            const DenseTensor& B_scale,
                            const DenseTensor& bias,
                            const DenseTensor& pre_gelu,
                            const DenseTensor& workspace,
                            bool transa,
                            bool transb,
                            bool grad,
                            bool accumulate,
                            bool use_split_accumulator,
                            int math_sm_count,
                            bool is_A_1d_scaled,
                            bool is_B_1d_scaled,
                            DenseTensor* out,
                            DenseTensor* pre_gelu_out,
                            DenseTensor* workspace_out);

}  // namespace phi
