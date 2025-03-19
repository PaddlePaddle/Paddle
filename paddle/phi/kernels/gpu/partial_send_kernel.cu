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
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void PartialSendKernel(const Context& dev_ctx,
                       const DenseTensor& x_in,
                       int peer,
                       int num,
                       int id) {
#if (defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)) && \
    NCCL_VERSION_CODE >= 2703
  auto x = &x_in;
  int numel = x->numel();

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

  gpuStream_t stream = nullptr;

  phi::distributed::NCCLCommContext* comm_ctx = nullptr;
  int nranks = 0;
  int rank = 0;

  comm_ctx =
      static_cast<phi::distributed::NCCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(comm_ctx,
                    nullptr,
                    common::errors::Unavailable(
                        "NCCLCommContext is nullptr, collective op should "
                        "has ring_id attr."));

  stream = dev_ctx.stream();
  nranks = comm_ctx->GetSize();
  rank = comm_ctx->GetRank();

  PADDLE_ENFORCE_LT(
      peer,
      nranks,
      common::errors::InvalidArgument("The value of peer (%d) you set must "
                                      "be less than ranks (%d).",
                                      peer,
                                      nranks));

  auto send_buf = distributed::GetPartialTensor(*x, offset, send_numel);
  comm_ctx->Send(send_buf, send_numel, peer, stream);

  VLOG(3) << "rank " << rank << " send " << send_numel << " from offset["
          << offset << "] to " << peer;

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
                   phi::PartialSendKernel,
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
                   phi::PartialSendKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
#endif
