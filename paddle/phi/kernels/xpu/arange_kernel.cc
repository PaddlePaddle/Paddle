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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/range_function.h"

namespace phi {

template <typename T, typename Context>
void ArangeKernel(const Context& dev_ctx,
                  const DenseTensor& start,
                  const DenseTensor& end,
                  const DenseTensor& step,
                  DenseTensor* out) {
  T start_value = GetValue<T, Context>(dev_ctx, start);
  T end_value = GetValue<T, Context>(dev_ctx, end);
  T step_value = GetValue<T, Context>(dev_ctx, step);

  int64_t size = 0;
  phi::funcs::GetSize(start_value, end_value, step_value, &size);
  out->Resize(phi::make_ddim({size}));
  dev_ctx.template Alloc<T>(out);

  DenseTensor out_cpu;
  out_cpu.Resize({out->numel()});
  dev_ctx.template HostAlloc<T>(&out_cpu);
  T* out_cpu_data = out_cpu.data<T>();

  T value = start_value;
  for (int64_t i = 0; i < size; ++i) {
    out_cpu_data[i] = value;
    value += step_value;
  }
  phi::Copy(dev_ctx, out_cpu, out->place(), true, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    arange, XPU, ALL_LAYOUT, phi::ArangeKernel, float, double, int, int64_t) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
}
