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

#include "paddle/phi/kernels/dequantize_log_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void DequantizeLogKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dict,
                         DenseTensor* out) {
  const float* dict_data = dict.data<float>();
  const T* input_data = x.data<T>();
  float* output_data = dev_ctx.template Alloc<float>(out);
  int ind = static_cast<int>(x.numel());
  for (size_t i = 0; i < (unsigned)ind; i++) {
    if (input_data[i] < 0) {
      output_data[i] = -dict_data[input_data[i] + 128];
    } else {
      output_data[i] = dict_data[input_data[i]];
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    dequantize_log, CPU, ALL_LAYOUT, phi::DequantizeLogKernel, int8_t) {}
