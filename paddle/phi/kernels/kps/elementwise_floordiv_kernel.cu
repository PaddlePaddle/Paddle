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

#include "paddle/phi/kernels/elementwise_floordiv_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"

namespace phi {

// Create the definition of FloorDivide
DEFINE_CUDA_ELEMENTWISE_OP(FloorDivide)

template <typename T, typename Context>
void FloorDivideKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       DenseTensor* out) {
  int axis = -1;
  FloorDivideRawKernel<T>(dev_ctx, x, y, axis, out);
}

}  // namespace phi

#ifdef PADDLE_WITH_XPU_KP

PD_REGISTER_KERNEL(
    floor_divide_raw, KPS, ALL_LAYOUT, phi::FloorDivideRawKernel, int) {}

#else

PD_REGISTER_KERNEL(floor_divide_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::FloorDivideRawKernel,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(
    floor_divide, KPS, ALL_LAYOUT, phi::FloorDivideKernel, int, int64_t) {}

#endif
