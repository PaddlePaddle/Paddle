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

#include "paddle/phi/kernels/dist_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/p_norm_kernel.h"

namespace phi {

template <typename T, typename Context>
void DistKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                float p,
                DenseTensor* out) {
  auto t = Subtract<T, Context>(dev_ctx, x, y);
  PNormKernel<T, Context>(dev_ctx, t, p, -1, 1e-12, false, true, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(dist, CPU, ALL_LAYOUT, phi::DistKernel, float, double) {}
