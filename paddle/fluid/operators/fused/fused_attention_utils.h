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

#pragma once

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/process_group_nccl.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/flags.h"
PHI_DECLARE_bool(dynamic_static_unified_comm);
#endif

#include "paddle/fluid/distributed/collective/utils.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/errors.h"

namespace phi {
namespace fusion {

template <typename T>
static void AllReduce(phi::DenseTensor &tensor,  // NOLINT
                      const int ring_id,
                      const phi::GPUContext &dev_ctx) {
  if (ring_id == -1) return;
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();

  if (map->has(ring_id)) {
    paddle::distributed::ProcessGroup *pg = map->get(ring_id);
    auto pg_nccl = static_cast<paddle::distributed::ProcessGroupNCCL *>(pg);
    paddle::distributed::AllreduceOptions opts;
    opts.reduce_op = paddle::distributed::ReduceOp::SUM;
    auto task = pg_nccl->AllReduce(&tensor, tensor, opts, true, true);
    task->Wait();
  } else {
    auto dtype = phi::ToNCCLDataType(tensor.dtype());
    int64_t numel = tensor.numel();
    const void *sendbuff = tensor.data<T>();
    auto place = dev_ctx.GetPlace();
    void *recvbuff =
        dev_ctx.template Alloc<T>(&tensor, tensor.numel() * sizeof(T));
    gpuStream_t stream = nullptr;
    paddle::platform::NCCLComm *comm = nullptr;
    phi::distributed::NCCLCommContext *comm_ctx = nullptr;

    const auto &comm_context_manager =
        phi::distributed::CommContextManager::GetInstance();

    if (FLAGS_dynamic_static_unified_comm) {
      // Use New Communication Library
      PADDLE_ENFORCE_EQ(comm_context_manager.Has(std::to_string(ring_id)),
                        true,
                        paddle::platform::errors::InvalidArgument(
                            "You choose to use new communication library by "
                            "setting environment "
                            "variable FLAGS_dynamic_static_unified_comm True. "
                            "But ring_id(%d) is "
                            "not found in comm_context_manager.",
                            std::to_string(ring_id)));
      comm_ctx = static_cast<phi::distributed::NCCLCommContext *>(
          comm_context_manager.Get(std::to_string(ring_id)));
      PADDLE_ENFORCE_NE(comm_ctx,
                        nullptr,
                        paddle::platform::errors::Unavailable(
                            "NCCLCommContext is nullptr, collective op should "
                            "has ring_id attr."));

      stream = comm_ctx->GetStream();
      VLOG(3) << "new comm_context_manager has ring_id" << ring_id;
    } else {
      comm = paddle::platform::NCCLCommContext::Instance().Get(ring_id, place);

      stream = dev_ctx.stream();
      VLOG(3) << "old NCCLCommContext has ring_id " << ring_id;
    }
    if (comm_ctx) {
      comm_ctx->AllReduce(&tensor, tensor, ncclSum, stream);
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllReduce(
          sendbuff, recvbuff, numel, dtype, ncclSum, comm->comm(), stream));
    }
  }
#else
  PADDLE_THROW(phi::errors::Unimplemented(
      "PaddlePaddle should compile with NCCL or RCCL when used tensor model "
      "parallel op."));
#endif
}

}  // namespace fusion
}  // namespace phi
