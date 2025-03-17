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

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/all_to_all_kernel.h"
#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/phi/core/distributed/bkcl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void BroadcastKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     int root,
                     DenseTensor* out) {
#if defined(PADDLE_WITH_XPU_BKCL)
  PADDLE_ENFORCE_GT(x.numel(),
                    0,
                    common::errors::InvalidArgument(
                        "Tensor need be broadcast must not empty."));

  dev_ctx.template Alloc<T>(out);
  auto comm_context =
      static_cast<distributed::BKCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(
      comm_context,
      nullptr,
      errors::Unavailable("BKCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));
  comm_context->Broadcast(out, x, root, comm_context->GetStream());
#else
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "PaddlePaddle should be compiled with XPU."));
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(broadcast,
                   XPU,
                   ALL_LAYOUT,
                   phi::BroadcastKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
