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

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/range_function.h"

namespace phi {

template <typename T, typename Context>
void ArangeKernel(const Context& dev_ctx,
                  const DenseTensor& start,
                  const DenseTensor& end,
                  const DenseTensor& step,
                  DenseTensor* out) {
  auto place = dev_ctx.GetPlace();
  auto cpu_place = phi::CPUPlace();

  DenseTensor n_cpu;
  n_cpu.Resize({start.numel()});
  T* n_cpu_data = dev_ctx.template HostAlloc<T>(&n_cpu);

  paddle::memory::Copy(
      cpu_place, n_cpu_data, place, start.data<T>(), sizeof(T) * start.numel());
  T start_value = n_cpu_data[0];
  paddle::memory::Copy(
      cpu_place, n_cpu_data, place, end.data<T>(), sizeof(T) * end.numel());
  T end_value = n_cpu_data[0];
  paddle::memory::Copy(
      cpu_place, n_cpu_data, place, step.data<T>(), sizeof(T) * step.numel());
  T step_value = n_cpu_data[0];

  int64_t size = 0;
  phi::funcs::GetSize(start_value, end_value, step_value, &size);
  out->Resize(phi::make_ddim({size}));
  T* out_data = dev_ctx.template Alloc<T>(out);

  DenseTensor out_cpu;
  out_cpu.Resize({out->numel()});
  T* out_cpu_data = dev_ctx.template HostAlloc<T>(&out_cpu);
  T value = start_value;
  for (int64_t i = 0; i < size; ++i) {
    out_cpu_data[i] = value;
    value += step_value;
  }
  paddle::memory::Copy(
      place, out_data, cpu_place, out_cpu_data, out->numel() * sizeof(T));
}

}  // namespace phi

PD_REGISTER_KERNEL(
    arange, XPU, ALL_LAYOUT, phi::ArangeKernel, float, double, int, int64_t) {}
