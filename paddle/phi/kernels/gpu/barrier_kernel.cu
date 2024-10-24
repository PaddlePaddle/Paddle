// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "glog/logging.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_helper.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/platform/collective_helper.h"
COMMON_DECLARE_bool(dynamic_static_unified_comm);
#endif

namespace phi {

template <typename T, typename Context>
void BarrierKernel(const Context& dev_ctx,
                   const DenseTensor& x_in,
                   DenseTensor* out) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)

  if (FLAGS_dynamic_static_unified_comm) {
    auto comm_ctx =
        static_cast<distributed::NCCLCommContext*>(dev_ctx.GetCommContext());
    PADDLE_ENFORCE_NOT_NULL(
        comm_ctx,
        phi::errors::Unavailable(
            "NCCLCommContext is nullptr, collective op should "
            "has ring_id attr."));
    auto stream = comm_ctx->GetStream();
    ncclRedOp_t nccl_red_type = ncclSum;
    comm_ctx->AllReduce(out, x_in, nccl_red_type, stream);
    phi::backends::gpu::GpuStreamSync(stream);
  }
#else
  PADDLE_THROW(
      phi::errors::Unavailable("PaddlePaddle should compile with NCCL."));
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(barrier, GPU, ALL_LAYOUT, phi::BarrierKernel, int) {}
