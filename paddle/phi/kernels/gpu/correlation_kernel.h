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

#include "paddle/phi/core/dense_tensor.h"

namespace phi {
template <typename T, typename Context>
void CorrelationCUDAKernel(const Context &dev_ctx,
                           const DenseTensor &input1,
                           const DenseTensor &input2,
                           int pad_size,
                           int kernel_size,
                           int max_displacement,
                           int stride1,
                           int stride2,
                           int corr_type_multiply,
                           DenseTensor *out);
}  // namespace phi
