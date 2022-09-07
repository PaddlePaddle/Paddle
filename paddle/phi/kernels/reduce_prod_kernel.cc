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

#include "paddle/phi/kernels/reduce_prod_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ProdKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const IntArray& dims,
                bool keep_dim,
                DenseTensor* out) {
  bool reduce_all = false;
  ProdRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    prod, CPU, ALL_LAYOUT, phi::ProdKernel, float, double, int, int64_t) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(
    prod, GPU, ALL_LAYOUT, phi::ProdKernel, float, double, int, int64_t) {}
#endif

#if defined(PADDLE_WITH_XPU_KP)
PD_REGISTER_KERNEL(prod, KPS, ALL_LAYOUT, phi::ProdKernel, float) {}
#endif
