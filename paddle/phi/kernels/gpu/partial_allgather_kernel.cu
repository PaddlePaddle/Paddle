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
void PartialAllGatherOpCUDAKernel(const Context& dev_ctx,
                                  const DenseTensor& x_in,
                                  int nranks,
                                  int rank,
                                  DenseTensor* out) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  auto in = &x_in;
  int64_t numel = in->numel();
  ncclDataType_t dtype = phi::ToNCCLDataType(in->dtype());

  gpuStream_t stream = nullptr;
  phi::distributed::NCCLCommContext* comm_ctx = nullptr;

  int real_nranks = 0;
  int real_rank = 0;

  comm_ctx =
      static_cast<phi::distributed::NCCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(comm_ctx,
                    nullptr,
                    common::errors::Unavailable(
                        "NCCLCommContext is nullptr, collective op should "
                        "has ring_id attr."));

  stream = dev_ctx.stream();
  real_nranks = comm_ctx->GetSize();
  real_rank = comm_ctx->GetRank();

  PADDLE_ENFORCE_EQ(nranks,
                    real_nranks,
                    common::errors::InvalidArgument(
                        "nranks: %s should equal to %s", nranks, real_nranks));
  PADDLE_ENFORCE_EQ(rank,
                    real_rank,
                    common::errors::InvalidArgument(
                        "rank: %s should equal to %s", rank, real_rank));

  PADDLE_ENFORCE_EQ((numel % nranks),
                    0,
                    common::errors::InvalidArgument(
                        "The input numel (%d) must be divisible by nranks(%d)",
                        numel,
                        nranks));

  phi::DDim dims = in->dims();
  out->Resize(dims);
  dev_ctx.template Alloc<T>(out);

  int64_t send_numel = numel / nranks;
  int offset = send_numel * rank;

  auto send_buf = distributed::GetPartialTensor(*in, offset, send_numel);
  comm_ctx->AllGather(out, send_buf, stream);
#else
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "PaddlePaddle should compile with GPU."));
#endif
}

}  // namespace phi

#if (NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000) || \
    defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(partial_allgather,
                   GPU,
                   ALL_LAYOUT,
                   phi::PartialAllGatherOpCUDAKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(partial_allgather,
                   GPU,
                   ALL_LAYOUT,
                   phi::PartialAllGatherOpCUDAKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
#endif
