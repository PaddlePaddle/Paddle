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

#include "glog/logging.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/send_recv_functor.h"

#if defined(PADDLE_WITH_NCCL) || \
    defined(PADDLE_WITH_RCCL) && NCCL_VERSION_CODE >= 2703
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void PSendKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 int peer,
                 bool dynamic_shape) {
#if defined(PADDLE_WITH_NCCL) || \
    defined(PADDLE_WITH_RCCL) && NCCL_VERSION_CODE >= 2703
  auto comm_ctx =
      GetCommContext<Context, distributed::NCCLCommContext>(dev_ctx, peer);
  gpuStream_t stream = dev_ctx.stream();
  if (dynamic_shape) {
    send_shape_info<Context, distributed::NCCLCommContext, gpuStream_t>(
        dev_ctx, x, comm_ctx, peer, stream);
  }
  comm_ctx->Send(x, x.numel(), peer, stream);
#else
  PADDLE_THROW(
      errors::PreconditionNotMet("PaddlePaddle should compile with GPU."
                                 "and NCCL version >= 2.7.3 is needed."));
#endif
}

template <typename T, typename Context>
void PSendArrayKernel(const Context& dev_ctx,
                      const TensorArray& x_array,
                      int peer) {
#if defined(PADDLE_WITH_NCCL) || \
    defined(PADDLE_WITH_RCCL) && NCCL_VERSION_CODE >= 2703
  auto comm_ctx =
      GetCommContext<Context, distributed::NCCLCommContext>(dev_ctx, peer);
  gpuStream_t stream = dev_ctx.stream();
  for (size_t idx = 0; idx < x_array.size(); idx++) {
    VLOG(3) << "DenseTensorArray: idx(" << idx << ")";
    auto x = x_array.at(idx);
    int numel = x.numel();
    ncclDataType_t dtype = ToNCCLDataType(x.type());
    comm_ctx->Send(x, x.numel(), peer, stream);
    VLOG(3) << "rank " << comm_ctx->GetRank() << " send "
            << common::product(x.dims()) << " to " << peer;
  }
#else
  PADDLE_THROW(
      errors::PreconditionNotMet("PaddlePaddle should compile with GPU."
                                 "and NCCL version >= 2.7.3 is needed."));
#endif
}

}  // namespace phi

#if NCCL_VERSION_CODE >= 21000
PD_REGISTER_KERNEL(p_send,
                   GPU,
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
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(p_send_array,
                   GPU,
                   ALL_LAYOUT,
                   phi::PSendArrayKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int64_t,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(p_send,
                   GPU,
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
                   GPU,
                   ALL_LAYOUT,
                   phi::PSendArrayKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int64_t,
                   phi::dtype::float16) {}
#endif
