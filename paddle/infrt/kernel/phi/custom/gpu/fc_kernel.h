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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/impl/fc_kernel_impl.h"

namespace infrt {
namespace kernel {

extern template void FcKernel<float, ::phi::GPUContext>(
    const ::phi::GPUContext& dev_ctx,
    const ::phi::DenseTensor& input,
    const ::phi::DenseTensor& weight,
    const ::phiDenseTensor& bias,
    int in_num_col_dims,
    ::phi::DenseTensor* out);

extern template void FcKernel<double, ::phi::GPUContext>(
    const ::phi::GPUContext& dev_ctx,
    const ::phi::DenseTensor& input,
    const ::phi::DenseTensor& weight,
    const ::phiDenseTensor& bias,
    int in_num_col_dims,
    ::phi::DenseTensor* out);

}  // namespace kernel
}  // namespace infrt
