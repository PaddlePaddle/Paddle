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

#include "paddle/phi/kernels/send_v3_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_GLOO)
#include "paddle/phi/core/distributed/gloo_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void SendV3Kernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  int peer,
                  bool dynamic_shape) {
  PADDLE_THROW(errors::Unavailable("Do not support send for cpu kernel now."));
}

template <typename T, typename Context>
void SendV3ArrayKernel(const Context& dev_ctx,
                       const TensorArray& x,
                       int peer,
                       bool dynamic_shape,
                       DenseTensor* out) {
  PADDLE_THROW(
      errors::Unavailable("Do not support send array for cpu kernel now."));
}

}  // namespace phi

PD_REGISTER_KERNEL(send_v3,
                   CPU,
                   ALL_LAYOUT,
                   phi::SendV3Kernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int64_t,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(send_v3_array,
                   CPU,
                   ALL_LAYOUT,
                   phi::SendV3ArrayKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int64_t,
                   phi::dtype::float16) {}
