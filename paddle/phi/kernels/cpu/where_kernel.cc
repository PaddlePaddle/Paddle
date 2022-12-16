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

#include "paddle/phi/kernels/where_kernel.h"

#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void WhereKernel(const Context& ctx,
                 const DenseTensor& condition,
                 const DenseTensor& x,
                 const DenseTensor& y,
                 DenseTensor* out) {
  const bool* cond_data = condition.data<bool>();
  const T* x_data = x.data<T>();
  const T* y_data = y.data<T>();
  auto x_numel = x.numel();

  T* out_data = ctx.template Alloc<T>(out);

  for (int i = 0; i < x_numel; i++) {
    out_data[i] = cond_data[i] ? x_data[i] : y_data[i];
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    where, CPU, ALL_LAYOUT, phi::WhereKernel, float, double, int, int64_t) {}
