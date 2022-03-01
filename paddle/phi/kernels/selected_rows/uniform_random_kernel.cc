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

#include "paddle/phi/kernels/uniform_random_kernel.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void UniformRandomRawSRKernel(const Context& dev_ctx,
                              const ScalarArray& shape,
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
void UniformRandomSRKernel(const Context& dev_ctx,
                           const ScalarArray& shape,
                           DataType dtype,
                           float min,
                           float max,
                           int seed,
                           SelectedRows* out) {
  phi::UniformRandomKernel<T>(
      dev_ctx, shape, dtype, min, max, seed, out->mutable_value());
}

}  // namespace phi

PD_REGISTER_KERNEL(uniform_random_raw_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::UniformRandomRawSRKernel,
                   float,
                   double,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(uniform_random_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::UniformRandomSRKernel,
                   float,
                   double,
                   phi::dtype::bfloat16) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

PD_REGISTER_KERNEL(uniform_random_raw_sr,
                   GPU,
                   ALL_LAYOUT,
                   phi::UniformRandomRawSRKernel,
                   float,
                   double) {}

PD_REGISTER_KERNEL(uniform_random_sr,
                   GPU,
                   ALL_LAYOUT,
                   phi::UniformRandomSRKernel,
                   float,
                   double) {}
#endif
