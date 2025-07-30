/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#ifdef PADDLE_WITH_CUSTOM_DEVICE
namespace phi {

template <typename T, typename Context>
void CIdentityKernel(const Context& dev_ctx,
                     const DenseTensor& x_in,
                     int ring_id,
                     bool use_calc_stream,
                     bool use_model_parallel,
                     DenseTensor* out) {
  auto x = &x_in;

  int rid = ring_id;
  PADDLE_ENFORCE_GE(
      rid,
      0,
      common::errors::InvalidArgument(
          "The ring_id (%d) for c_identity op must be non-negative.", rid));
  dev_ctx.template Alloc<T>(out);

  phi::Copy(dev_ctx, *x, out->place(), false, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(c_identity,
                   Custom,
                   ALL_LAYOUT,
                   phi::CIdentityKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
#endif
