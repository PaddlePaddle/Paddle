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

#include "paddle/phi/kernels/recv_v3_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_GLOO)
#include "paddle/phi/core/distributed/gloo_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void RecvV3Kernel(const Context& dev_ctx,
                  int peer,
                  DataType dtype,
                  bool dynamic_shape,
                  DenseTensor* out) {
  PADDLE_THROW(errors::Unavailable("Do not support recv for cpu kernel now."));
}

template <typename T, typename Context>
void RecvV3ArrayKernel(const Context& dev_ctx,
                       int peer,
                       DataType dtype,
                       const std::vector<int>& out_shape,
                       TensorArray* out_array) {
  PADDLE_THROW(
      errors::Unavailable("Do not support recv array for cpu kernel now."));
}

}  // namespace phi

PD_REGISTER_KERNEL(recv_v3,
                   CPU,
                   ALL_LAYOUT,
                   phi::RecvV3Kernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int64_t,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(recv_v3_array,
                   CPU,
                   ALL_LAYOUT,
                   phi::RecvV3ArrayKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int64_t,
                   phi::dtype::float16) {}
