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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {
template <typename Context>
void LodArrayLengthKernel(const Context& dev_ctx,
                          const TensorArray& x,
                          DenseTensor* out) {
  out->Resize({1});
  *dev_ctx.template Alloc<int64_t>(out) = static_cast<int64_t>(x.size());
}
}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_DTYPE(lod_array_length,
                                 CPU,
                                 ALL_LAYOUT,
                                 phi::LodArrayLengthKernel<phi::CPUContext>) {
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
