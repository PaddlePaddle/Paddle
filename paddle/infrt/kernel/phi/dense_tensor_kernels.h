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

#include "paddle/infrt/backends/host/phi_allocator.h"
#include "paddle/infrt/host_context/kernel_utils.h"
#include "paddle/phi/core/dense_tensor.h"

namespace infrt {
namespace kernel {
namespace phi {

::phi::DenseTensor CreateDenseTensorCpuF32Nchw(
    backends::CpuPhiAllocator* allocator,
    host_context::Attribute<std::vector<int64_t>> dims,
    host_context::Attribute<std::vector<int64_t>> lod);

void FillDenseTensorF32(::phi::DenseTensor* dense_tensor,
                        host_context::Attribute<std::vector<float>> values);
void PrintDenseTensor(::phi::DenseTensor* dense_tensor);

}  // namespace phi
}  // namespace kernel
}  // namespace infrt
