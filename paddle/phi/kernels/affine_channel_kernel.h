// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
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
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

// AffineChannel CUDA kernel wrapper
template <typename T, typename Context>
void AffineChannelCUDAKernel(const Context& dev_ctx,
                             const DenseTensor& x_in,
                             const DenseTensor& scale_in,
                             const DenseTensor& bias_in,
                             const std::string& data_layout,
                             DenseTensor* out);

}  // namespace phi
