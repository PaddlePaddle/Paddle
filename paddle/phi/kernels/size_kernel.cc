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

#include "paddle/phi/kernels/size_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

template <typename Context>
void SizeKernel(const Context& ctx,
                const DenseTensor& input,
                DenseTensor* out) {
  auto* out_data = ctx.template HostAlloc<int64_t>(out);
  out_data[0] = input.numel();
}

}  // namespace phi

PD_REGISTER_GENERAL_KERNEL(
    size, CPU, ALL_LAYOUT, phi::SizeKernel<phi::CPUContext>, ALL_DTYPE) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_GENERAL_KERNEL(
    size, GPU, ALL_LAYOUT, phi::SizeKernel<phi::GPUContext>, ALL_DTYPE) {
  kernel->OutputAt(0)
      .SetBackend(phi::Backend::CPU)
      .SetDataType(phi::DataType::INT64);
}
#endif
