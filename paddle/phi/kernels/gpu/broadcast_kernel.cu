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

#include "paddle/phi/kernels/broadcast_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void BroadcastKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     int root,
                     DenseTensor* out) {
  PADDLE_ENFORCE_GT(x.numel(),
                    0,
                    common::errors::InvalidArgument(
                        "Tensor need be broadcast must not empty."));

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  dev_ctx.template Alloc<T>(out);
  gpuStream_t stream = dev_ctx.stream();
  auto comm_context =
      static_cast<distributed::NCCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(
      comm_context,
      nullptr,
      errors::Unavailable("NCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));
  comm_context->Broadcast(out, x, root, stream);
  out->set_lod(x.lod());
#else
  PADDLE_THROW(
      errors::PreconditionNotMet("PaddlePaddle should compile with GPU."));
#endif
}

}  // namespace phi

#if NCCL_VERSION_CODE >= 21000
PD_REGISTER_KERNEL(broadcast,
                   GPU,
                   ALL_LAYOUT,
                   phi::BroadcastKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#else
PD_REGISTER_KERNEL(broadcast,
                   GPU,
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
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif
