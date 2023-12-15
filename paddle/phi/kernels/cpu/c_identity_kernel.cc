/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/c_identity_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void CIdentityKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     int ring_id,
                     bool use_calc_stream,
                     bool use_model_parallel,
                     DenseTensor* out) {
  PADDLE_THROW(
      errors::Unavailable("Do not support c_identity for cpu kernel now."));
}

}  // namespace phi

PD_REGISTER_KERNEL(c_identity,
                   CPU,
                   ALL_LAYOUT,
                   phi::CIdentityKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
