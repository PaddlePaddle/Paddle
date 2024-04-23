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

#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math.h"

namespace phi {

template <typename T>
__global__ void KeDequantize(const T* in,
                             const float* dict,
                             int num,
                             float* out) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num) {
    if (in[idx] < 0) {
      out[idx] = -dict[in[idx] + 128];
    } else {
      out[idx] = dict[in[idx]];
    }
  }
}

template <typename T, typename Context>
void DequantizeLogKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dict,
                         DenseTensor* out) {
  const T* in_data = x.data<T>();
  const float* dict_data = dict.data<float>();
  float* out_data = dev_ctx.template Alloc<float>(out);

  int num = x.numel();
  int block = 512;
  int grid = (num + block - 1) / block;

  KeDequantize<T>
      <<<grid, block, 0, dev_ctx.stream()>>>(in_data, dict_data, num, out_data);
}
}  // namespace phi

PD_REGISTER_KERNEL(
    dequantize_log, GPU, ALL_LAYOUT, phi::DequantizeLogKernel, int8_t) {}
