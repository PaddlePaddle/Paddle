// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/p_send_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_GLOO)
#include "paddle/phi/core/distributed/gloo_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void PSendKernel(const Context& dev_ctx UNUSED,
                 const DenseTensor& x UNUSED,
                 int peer UNUSED,
                 bool dynamic_shape UNUSED) {
  PADDLE_THROW(errors::Unavailable("Do not support send for cpu kernel now."));
}

template <typename T, typename Context>
void PSendArrayKernel(const Context& dev_ctx UNUSED,
                      const TensorArray& x UNUSED,
                      int peer UNUSED,
                      bool dynamic_shape UNUSED,
                      DenseTensor* out UNUSED) {
  PADDLE_THROW(
      errors::Unavailable("Do not support send array for cpu kernel now."));
}

}  // namespace phi

PD_REGISTER_KERNEL(p_send,
                   CPU,
                   ALL_LAYOUT,
                   phi::PSendKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int64_t,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(p_send_array,
                   CPU,
                   ALL_LAYOUT,
                   phi::PSendArrayKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int64_t,
                   phi::dtype::float16) {}
