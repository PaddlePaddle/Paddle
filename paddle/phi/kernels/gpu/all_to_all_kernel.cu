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

#include "paddle/phi/kernels/all_to_all_kernel.h"
#include "glog/logging.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/distributed/utils.h"
#endif

namespace phi {

template <typename T, typename Context>
void AllToAllKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    DenseTensor* out) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#if NCCL_VERSION_CODE >= 2703
  auto x_dims = x.dims();
  out->Resize(x_dims);
  dev_ctx.template Alloc<T>(out);

  auto comm_ctx =
      static_cast<distributed::NCCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(
      comm_ctx,
      nullptr,
      errors::Unavailable("NCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));
  gpuStream_t stream = dev_ctx.stream();
  PADDLE_ENFORCE_NOT_NULL(stream,
                          errors::NotFound("Should initialize NCCL firstly."));

  int nranks = comm_ctx->GetSize();
  int send_numel = x.numel() / nranks;
  size_t offset = 0;

  PADDLE_ENFORCE_EQ(
      x_dims[0] % nranks,
      0,
      errors::InvalidArgument(
          "The first dimension size (%d) of the input tensor must be "
          "divisible by the number of ranks (%d).",
          x_dims[0],
          nranks));

  comm_ctx->GroupStart();

  const auto* send_buf = x.data<T>();
  auto* recv_buf = out->data<T>();
  for (auto i = 0; i < nranks; ++i) {
    auto send_buf = phi::distributed::GetPartialTensor(x, offset, send_numel);
    comm_ctx->Send(send_buf, send_numel, i, stream);
    auto recv_buf =
        phi::distributed::GetPartialTensor(*out, offset, send_numel);
    comm_ctx->Recv(&recv_buf, send_numel, i, stream);
    offset += send_numel;
  }
  comm_ctx->GroupEnd();
#else
  PADDLE_THROW(common::errors::Unavailable("NCCL version >= 2.7.3 is needed."));
#endif
#else
  PADDLE_THROW(
      errors::PreconditionNotMet("PaddlePaddle should compile with GPU."));
#endif
}

}  // namespace phi

#if (NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000) || \
    defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(all_to_all,
                   GPU,
                   ALL_LAYOUT,
                   phi::AllToAllKernel,
                   float,
                   double,
                   int,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(all_to_all,
                   GPU,
                   ALL_LAYOUT,
                   phi::AllToAllKernel,
                   float,
                   double,
                   int,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int64_t,
                   bool,
                   phi::dtype::float16) {}
#endif
