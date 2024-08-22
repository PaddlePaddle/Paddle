/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#ifndef PADDLE_FLUID_OPERATORS_COLLECTIVE_C_SYNC_COMM_STREAM_OP_H_
#define PADDLE_FLUID_OPERATORS_COLLECTIVE_C_SYNC_COMM_STREAM_OP_H_

#include <string>

#include "paddle/fluid/framework/op_registry.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/common/flags.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
COMMON_DECLARE_bool(dynamic_static_unified_comm);
#endif

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#elif defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/phi/core/distributed/bkcl_comm_context.h"
#include "paddle/phi/core/platform/collective_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class CSyncCommStreamKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto place = ctx.GetPlace();
    int ring_id = ctx.Attr<int>("ring_id");

    gpuStream_t stream = nullptr;
    const auto& comm_context_manager =
        phi::distributed::CommContextManager::GetInstance();
    if (FLAGS_dynamic_static_unified_comm) {
      PADDLE_ENFORCE_EQ(comm_context_manager.Has(std::to_string(ring_id)),
                        true,
                        common::errors::InvalidArgument(
                            "You choose to use new communication library by "
                            "setting environment "
                            "variable FLAGS_dynamic_static_unified_comm True. "
                            "But ring_id(%d) is "
                            "not found in comm_context_manager.",
                            std::to_string(ring_id)));
      auto comm_ctx = static_cast<phi::distributed::NCCLCommContext*>(
          comm_context_manager.Get(std::to_string(ring_id)));
      stream = comm_ctx->GetStream();
      VLOG(3) << "new comm_context_manager has rid " << ring_id;
    } else {
      stream =
          platform::NCCLCommContext::Instance().Get(ring_id, place)->stream();
      VLOG(3) << "old NCCLCommContext has rid " << ring_id;
    }

    phi::backends::gpu::GpuStreamSync(stream);

#elif defined(PADDLE_WITH_XPU_BKCL)
    auto place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(place.GetType() == phi::AllocationType::XPU,
                      true,
                      common::errors::PreconditionNotMet(
                          "Sync stream op can run on xpu place only for now."));
    int ring_id = ctx.Attr<int>("ring_id");
    XPUStream stream = nullptr;
    const auto& comm_context_manager =
        phi::distributed::CommContextManager::GetInstance();
    if (FLAGS_dynamic_static_unified_comm) {
      PADDLE_ENFORCE_EQ(comm_context_manager.Has(std::to_string(ring_id)),
                        true,
                        common::errors::InvalidArgument(
                            "You choose to use new communication library by "
                            "setting environment "
                            "variable FLAGS_dynamic_static_unified_comm True. "
                            "But ring_id(%d) is "
                            "not found in comm_context_manager.",
                            std::to_string(ring_id)));
      auto comm_ctx = static_cast<phi::distributed::BKCLCommContext*>(
          comm_context_manager.Get(std::to_string(ring_id)));
      stream = comm_ctx->GetStream();
      VLOG(3) << "new comm_context_manager has rid " << ring_id;
    } else {
      stream =
          platform::BKCLCommContext::Instance().Get(ring_id, place)->stream();
      VLOG(3) << "old BKCLCommContext has rid " << ring_id;
    }

    platform::XPUStreamSync(stream);

#else
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU or XPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle
#endif  // PADDLE_FLUID_OPERATORS_COLLECTIVE_C_SYNC_COMM_STREAM_OP_H_
