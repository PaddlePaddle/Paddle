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

#include "paddle/phi/kernels/randint_kernel.h"

#include <random>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/memory/memcpy.h"

namespace phi {

template <typename T, typename Context>
void RandintRawKernel(const Context& dev_ctx,
                      int low,
                      int high,
                      const IntArray& shape,
                      DataType dtype,
                      int seed,
                      DenseTensor* out) {
  out->Resize(phi::make_ddim(shape.GetData()));
  T* data = dev_ctx.template Alloc<T>(out);
  funcs::uniform_distribution<uint32_t> dist;
  funcs::uniform_int_transform<T, uint32_t> trans(low, high);
  funcs::distribution_and_transform<T>(dev_ctx, out, dist, trans);
}

template <typename T, typename Context>
void RandintKernel(const Context& dev_ctx,
                   int low,
                   int high,
                   const IntArray& shape,
                   DataType dtype,
                   DenseTensor* out) {
  RandintRawKernel<T>(dev_ctx, low, high, shape, dtype, 0, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    randint_raw, GPU, ALL_LAYOUT, phi::RandintRawKernel, int, int64_t) {}

PD_REGISTER_KERNEL(randint, GPU, ALL_LAYOUT, phi::RandintKernel, int, int64_t) {
}
