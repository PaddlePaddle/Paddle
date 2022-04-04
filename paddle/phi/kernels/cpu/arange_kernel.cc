/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/arange_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/range_function.h"

namespace phi {

template <typename T, typename Context>
void ArangeKernel(const Context& dev_ctx,
                  const DenseTensor& start,
                  const DenseTensor& end,
                  const DenseTensor& step,
                  DenseTensor* out) {
  T start_value = start.data<T>()[0];
  T end_value = end.data<T>()[0];
  T step_value = step.data<T>()[0];
  int64_t size = 0;
  phi::funcs::GetSize(start_value, end_value, step_value, &size);
  out->Resize(phi::make_ddim({size}));
  T* out_data = dev_ctx.template Alloc<T>(out);
  T value = start_value;
  for (int64_t i = 0; i < size; ++i) {
    out_data[i] = value;
    value += step_value;
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    arange, CPU, ALL_LAYOUT, phi::ArangeKernel, float, double, int, int64_t) {}
