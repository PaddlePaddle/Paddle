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

#include <math.h>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/trunc_kernel.h"

namespace phi {

template <typename T, typename Context>
void TruncKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out) {
  size_t numel = x.numel();
  const T* x_data = x.data<T>();
  T* out_data = dev_ctx.template Alloc<T>(out);

  for (size_t i = 0; i < numel; i++) {
    out_data[i] = trunc(x_data[i]);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    trunc, CPU, ALL_LAYOUT, phi::TruncKernel, float, double, int, int64_t) {}
