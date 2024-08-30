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

#include <algorithm>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/collective/process_group.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

#include "paddle/phi/core/distributed/comm_context_manager.h"

namespace phi {

template <typename T, typename Context>
void PartialSendCUDAKernel(const Context& dev_ctx,
                           const DenseTensor& x_in,
                           int ring_id,
                           int peer,
                           bool use_calc_stream,
                           int num,
                           int id) {
#if (defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)) && \
    NCCL_VERSION_CODE >= 2703
  auto x = &x_in;
  int numel = x->numel();
  int rid = ring_id;

  PADDLE_ENFORCE_GE(
      rid,
      0,
      common::errors::InvalidArgument(
          "The ring_id (%d) for partial_send op must be non-negative.", rid));
  PADDLE_ENFORCE_GE(
      peer,
      0,
      common::errors::InvalidArgument(
          "The peer (%d) for partial_send op must be non-negative.", peer));
  PADDLE_ENFORCE_GE(num,
                    1,
                    common::errors::InvalidArgument(
                        "The num (%d) for partial_send op must >=1", num));
  PADDLE_ENFORCE_EQ(
      (id >= 0 && id < num),
      true,
      common::errors::InvalidArgument(
          "The id (%d) for partial_send op must >=0 and <num (%d)", id, num));
  PADDLE_ENFORCE_EQ(
      (numel % num),
      0,
      common::errors::InvalidArgument(
          "The input numel (%d) must be divisible by num(%d)", numel, num));

  int64_t send_numel = numel / num;
  int64_t offset = send_numel * id;

  auto map = distributed::ProcessGroupMapFromGid::getInstance();
  if (map->has(rid)) {
    // Use ProcessGroup
    distributed::ProcessGroup* pg = map->get(rid);
    phi::DenseTensor tmp = *x;
    auto task = pg->Send(tmp, peer, offset, send_numel, /*sync_op*/ true);
    task->Wait();
  } else {
    gpuStream_t stream = nullptr;

    phi::distributed::NCCLCommContext* comm_ctx = nullptr;
    int nranks = 0;
    int rank = 0;

    const auto& comm_context_manager =
        phi::distributed::CommContextManager::GetInstance();

    PADDLE_ENFORCE_EQ(comm_context_manager.Has(std::to_string(rid)),
                      true,
                      common::errors::InvalidArgument(
                          "You choose to use new communication library by "
                          "setting environment "
                          "variable FLAGS_dynamic_static_unified_comm True. "
                          "But ring_id(%d) is "
                          "not found in comm_context_manager.",
                          std::to_string(rid)));
    comm_ctx = static_cast<phi::distributed::NCCLCommContext*>(
        comm_context_manager.Get(std::to_string(rid)));
    PADDLE_ENFORCE_NE(comm_ctx,
                      nullptr,
                      common::errors::Unavailable(
                          "NCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));

    stream = comm_ctx->GetStream();
    nranks = comm_ctx->GetSize();
    rank = comm_ctx->GetRank();

    VLOG(3) << "new comm_context_manager has ring_id " << rid;

    if (use_calc_stream) {
      // should ExecutionContext for calc stream.
      stream = dev_ctx.stream();
    }

    PADDLE_ENFORCE_LT(
        peer,
        nranks,
        common::errors::InvalidArgument("The value of peer (%d) you set must "
                                        "be less than ranks (%d).",
                                        peer,
                                        nranks));

    ncclDataType_t dtype = phi::ToNCCLDataType(x->dtype());

    auto send_buf = distributed::GetPartialTensor(*x, offset, send_numel);

    comm_ctx->Send(send_buf, send_numel, peer, stream);

    VLOG(3) << "rank " << rank << " send " << send_numel << " from offset["
            << offset << "] to " << peer;
  }
#else
  PADDLE_THROW(
      common::errors::Unavailable("PaddlePaddle should be compiled with NCCL "
                                  "and NCCL version >= 2.7.3 is needed."));
#endif
}
}  // namespace phi

#if (NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000) || \
    defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(partial_send,
                   GPU,
                   ALL_LAYOUT,
                   phi::PartialSendCUDAKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(partial_send,
                   GPU,
                   ALL_LAYOUT,
                   phi::PartialSendCUDAKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
#endif
