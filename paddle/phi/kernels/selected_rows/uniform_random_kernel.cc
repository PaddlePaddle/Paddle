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

#include "paddle/phi/kernels/selected_rows/uniform_random_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/uniform_random_kernel.h"

namespace phi {
namespace sr {

template <typename T, typename Context>
void UniformRandomRawKernel(const Context& dev_ctx,
                            const IntArray& shape,
                            DataType dtype,
                            float min,
                            float max,
                            int seed,
                            int diag_num,
                            int diag_step,
                            float diag_val,
                            SelectedRows* out) {
  phi::UniformRandomRawKernel<T>(dev_ctx,
                                 shape,
                                 dtype,
                                 min,
                                 max,
                                 seed,
                                 diag_num,
                                 diag_step,
                                 diag_val,
                                 out->mutable_value());
}

template <typename T, typename Context>
void UniformRandomKernel(const Context& dev_ctx,
                         const IntArray& shape,
                         DataType dtype,
                         float min,
                         float max,
                         int seed,
                         SelectedRows* out) {
  phi::UniformRandomKernel<T>(
      dev_ctx, shape, dtype, min, max, seed, out->mutable_value());
}

}  // namespace sr
}  // namespace phi

PD_REGISTER_KERNEL(uniform_random_raw_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sr::UniformRandomRawKernel,
                   float,
                   double,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(uniform_random_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sr::UniformRandomKernel,
                   float,
                   double,
                   phi::dtype::bfloat16) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

PD_REGISTER_KERNEL(uniform_random_raw_sr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sr::UniformRandomRawKernel,
                   float,
                   double) {}

PD_REGISTER_KERNEL(uniform_random_sr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sr::UniformRandomKernel,
                   float,
                   double) {}
#endif
