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
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_helper.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/platform/collective_helper.h"
COMMON_DECLARE_bool(dynamic_static_unified_comm);
#endif

namespace phi {

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
template <typename T, typename Context>
void BarrierKernel(const Context& dev_ctx,
                   const DenseTensor& x_in,
                   int ring_id,
                   DenseTensor* out) {
  auto place = dev_ctx.GetPlace();
  ncclDataType_t dtype = phi::ToNCCLDataType(x_in.dtype());
  int64_t numel = x_in.numel();
  const void* sendbuff = x_in.data();
  void* recvbuff = dev_ctx.template Alloc<T>(out);

  const auto& comm_context_manager =
      phi::distributed::CommContextManager::GetInstance();
  if (FLAGS_dynamic_static_unified_comm) {
    PADDLE_ENFORCE_EQ(comm_context_manager.Has(std::to_string(ring_id)),
                      true,
                      phi::errors::InvalidArgument(
                          "You choose to use new communication library by "
                          "setting environment "
                          "variable FLAGS_dynamic_static_unified_comm True. "
                          "But ring_id(%d) is "
                          "not found in comm_context_manager.",
                          std::to_string(ring_id)));
    auto comm_ctx = static_cast<phi::distributed::NCCLCommContext*>(
        comm_context_manager.Get(std::to_string(ring_id)));
    PADDLE_ENFORCE_NOT_NULL(
        comm_ctx,
        phi::errors::Unavailable(
            "NCCLCommContext is nullptr, collective op should "
            "has ring_id attr."));
    auto stream = comm_ctx->GetStream();
    ncclRedOp_t nccl_red_type = ncclSum;
    comm_ctx->AllReduce(out, x_in, nccl_red_type, stream);
    phi::backends::gpu::GpuStreamSync(stream);
    VLOG(3) << "new NCCLCommContext has rid " << ring_id;
  } else {
    auto comm = phi::platform::NCCLCommContext::Instance().Get(ring_id, place);
    auto stream = dev_ctx.stream();
    ncclRedOp_t nccl_red_type = ncclSum;
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllReduce(
        sendbuff, recvbuff, numel, dtype, nccl_red_type, comm->comm(), stream));
    phi::backends::gpu::GpuStreamSync(stream);
    VLOG(3) << "old NCCLCommContext has rid " << ring_id;
  }
}
#endif

}  // namespace phi

PD_REGISTER_KERNEL(barrier, GPU, ALL_LAYOUT, phi::BarrierKernel, int) {}
