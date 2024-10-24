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

#include "paddle/phi/kernels/dequantize_abs_max_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void DequantizeAbsMaxKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& scale,
                            float max_range,
                            DenseTensor* out) {
  const float* scale_factor = scale.data<float>();
  const T* input_data = x.data<T>();
  float* output_data = dev_ctx.template Alloc<float>(out);
  int ind = static_cast<int>(x.numel());
  for (size_t i = 0; i < (unsigned)ind; i++) {
    output_data[i] = scale_factor[0] * input_data[i] / max_range;
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(dequantize_abs_max,
                   CPU,
                   ALL_LAYOUT,
                   phi::DequantizeAbsMaxKernel,
                   int8_t,
                   int16_t) {}
