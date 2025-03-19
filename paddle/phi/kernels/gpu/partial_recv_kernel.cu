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
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/kernel_registry.h"
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void PartialRecvKernel(const Context& dev_ctx,
                       int peer,
                       DataType type,
                       const std::vector<int>& out_shape,
                       int num,
                       int id,
                       DenseTensor* out) {
#if (defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)) && \
    NCCL_VERSION_CODE >= 2703
  auto out_dims = out->dims();
  auto numel = out->numel();

  PADDLE_ENFORCE_GE(
      peer,
      0,
      common::errors::InvalidArgument(
          "The peer (%d) for partial_recv op must be non-negative.", peer));
  PADDLE_ENFORCE_GE(num,
                    1,
                    common::errors::InvalidArgument(
                        "The num (%d) for partial_recv op must >=1", num));
  PADDLE_ENFORCE_EQ(
      (id >= 0 && id < num),
      true,
      common::errors::InvalidArgument(
          "The id (%d) for partial_recv op must >=0 and <num (%d)", id, num));
  PADDLE_ENFORCE_EQ(
      (numel % num),
      0,
      common::errors::InvalidArgument(
          "The input numel (%d) must be divisible by num(%d)", numel, num));

  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);
  int64_t recv_numel = numel / num;
  int64_t offset = recv_numel * id;

  gpuStream_t stream = nullptr;
  phi::distributed::NCCLCommContext* comm_ctx = nullptr;
  int nranks = 0;
  int rank = 0;

  comm_ctx =
      static_cast<distributed::NCCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(comm_ctx,
                    nullptr,
                    common::errors::Unavailable(
                        "NCCLCommContext is nullptr, collective op should "
                        "has ring_id attr."));

  stream = comm_ctx->GetStream();
  nranks = comm_ctx->GetSize();
  rank = comm_ctx->GetRank();

  PADDLE_ENFORCE_LT(
      peer,
      nranks,
      common::errors::InvalidArgument("The value of peer (%d) you set must "
                                      "be less than nranks (%d).",
                                      peer,
                                      nranks));

  auto recv_buf = distributed::GetPartialTensor(*out, offset, recv_numel);
  comm_ctx->Recv(&recv_buf, recv_numel, peer, stream);

  VLOG(3) << "rank " << rank << " recv " << recv_numel << " from offset["
          << offset << "] from " << peer;
#else
  PADDLE_THROW(common::errors::Unavailable(
      "PaddlePaddle should be compiled with NCCL and "
      "NCCL version >= 2.7.3 is needed."));
#endif
}
}  // namespace phi

#if (NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000) || \
    defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(partial_recv,
                   GPU,
                   ALL_LAYOUT,
                   phi::PartialRecvKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(partial_recv,
                   GPU,
                   ALL_LAYOUT,
                   phi::PartialRecvKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
#endif
