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

#include "paddle/phi/kernels/dist_identity_kernel.h"

#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void DistIdentityKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        DenseTensor* out) {
  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
}

}  // namespace phi

#if NCCL_VERSION_CODE >= 21000
PD_REGISTER_KERNEL(dist_identity,
                   GPU,
                   ALL_LAYOUT,
                   phi::DistIdentityKernel,
                   float,
                   double,
                   int,
                   uint8_t,
                   int8_t,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(dist_identity,
                   GPU,
                   ALL_LAYOUT,
                   phi::DistIdentityKernel,
                   float,
                   double,
                   int,
                   uint8_t,
                   int8_t,
                   int64_t,
                   bool,
                   phi::dtype::float16) {}
#endif
