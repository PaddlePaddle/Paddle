// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/nextafter_kernel.h"
#include <algorithm>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math.h"

namespace phi {

template <typename T, typename Context>
void NextafterKernel(const Context& ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  if (x.numel() == 0 || y.numel() == 0) {
    return;
  }
  auto out_data = ctx.template Alloc<T>(out);
  auto x_data = x.data<T>();
  auto y_data = y.data<T>();
  int x_numel = x.numel();

  for (int i = 0; i < x_numel; ++i) {
    out_data[i] = std::nextafter(x_data[i], y_data[i]);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    nextafter, GPU, ALL_LAYOUT, phi::NextafterKernel, float, double) {}
