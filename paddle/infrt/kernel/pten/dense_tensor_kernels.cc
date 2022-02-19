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

#include "paddle/infrt/kernel/pten/dense_tensor_kernels.h"

namespace infrt {
namespace kernel {
namespace pten {

::pten::DenseTensor CreateDenseTensorCpuF32Nchw(
    backends::CpuPtenAllocator* allocator,
    host_context::Attribute<std::vector<int64_t>> dims,
    host_context::Attribute<std::vector<int64_t>> lod) {
  return ::pten::DenseTensor(
      allocator,
      ::pten::DenseTensorMeta(::pten::DataType::FLOAT32,
                              ::pten::framework::make_ddim(dims.get()),
                              ::pten::DataLayout::NCHW,
                              {}));
}

void FillDenseTensorF32(::pten::DenseTensor* dense_tensor,
                        host_context::Attribute<std::vector<int64_t>> values) {}

}  // namespace pten
}  // namespace kernel
}  // namespace infrt
