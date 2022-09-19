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

#include "paddle/phi/kernels/uniform_random_kernel.h"

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/kernel_registry.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/gpu/gpu_context.h"
#endif
#ifdef PADDLE_WITH_XPU
#include "paddle/phi/backends/xpu/xpu_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void UniformRandomKernel(const Context& dev_ctx,
                         const IntArray& shape,
                         DataType dtype,
                         const Scalar& min,
                         const Scalar& max,
                         int seed,
                         DenseTensor* out) {
  UniformRandomRawKernel<T>(
      dev_ctx, shape, dtype, min, max, seed, 0, 0, 0.0f, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(uniform_random,
                   CPU,
                   ALL_LAYOUT,
                   phi::UniformRandomKernel,
                   float,
                   double,
                   phi::dtype::bfloat16) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(
    uniform_random, GPU, ALL_LAYOUT, phi::UniformRandomKernel, float, double) {}
#endif

#ifdef PADDLE_WITH_XPU
PD_REGISTER_KERNEL(
    uniform_random, XPU, ALL_LAYOUT, phi::UniformRandomKernel, float) {}
#endif
