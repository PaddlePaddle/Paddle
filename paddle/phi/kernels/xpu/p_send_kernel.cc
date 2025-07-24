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

#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/phi/core/distributed/bkcl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void PSendKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 int peer,
                 bool dynamic_shape) {
#if defined(PADDLE_WITH_XPU_BKCL)
  auto comm_ctx =
      GetCommContext<Context, distributed::BKCLCommContext>(dev_ctx, peer);
  XPUStream stream = dev_ctx.stream();
  if (dynamic_shape) {
    send_shape_info<Context, distributed::BKCLCommContext, XPUStream>(
        dev_ctx, x, comm_ctx, peer, stream);
  }
  comm_ctx->Send(x, x.numel(), peer, stream);
#else
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "PaddlePaddle is not compiled with DWITH_XPU_BKCL, please recompile with "
      "DWITH_XPU_BKCL for using p_send kernel."));
#endif
}

template <typename T, typename Context>
void PSendArrayKernel(const Context& dev_ctx,
                      const TensorArray& x_array,
                      int peer) {
#if defined(PADDLE_WITH_XPU_BKCL)
  auto comm_ctx =
      GetCommContext<Context, distributed::BKCLCommContext>(dev_ctx, peer);
  XPUStream stream = dev_ctx.stream();
  for (size_t idx = 0; idx < x_array.size(); idx++) {
    VLOG(3) << "DenseTensorArray: idx(" << idx << ")";
    auto x = x_array.at(idx);
    comm_ctx->Send(x, x.numel(), peer, stream);
    VLOG(3) << "rank " << comm_ctx->GetRank() << " send "
            << common::product(x.dims()) << " to " << peer;
  }
#else
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "PaddlePaddle is not compiled with DWITH_XPU_BKCL, please recompile with "
      "DWITH_XPU_BKCL for using p_send_array kernel."));
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(p_send,
                   XPU,
                   ALL_LAYOUT,
                   phi::PSendKernel,
                   float,
                   double,
                   uint8_t,
                   int,
                   int64_t,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(p_send_array,
                   XPU,
                   ALL_LAYOUT,
                   phi::PSendArrayKernel,
                   float,
                   double,
                   uint8_t,
                   int,
                   int64_t,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
