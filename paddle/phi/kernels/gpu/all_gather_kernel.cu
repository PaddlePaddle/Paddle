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

#include "paddle/phi/kernels/all_gather_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void AllGatherKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     int nranks,
                     DenseTensor* out) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  auto out_dims = x.dims();
  out_dims[0] *= nranks;
  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);

  auto comm_ctx =
      static_cast<distributed::NCCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(
      comm_ctx,
      nullptr,
      errors::Unavailable("NCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));
  PADDLE_ENFORCE_EQ(
      nranks,
      comm_ctx->GetSize(),
      errors::InvalidArgument(
          "nranks: %s should equal to %s", nranks, comm_ctx->GetSize()));

  gpuStream_t stream = dev_ctx.stream();
  comm_ctx->AllGather(out, x, stream);
#else
  PADDLE_THROW(
      errors::PreconditionNotMet("PaddlePaddle should compile with GPU."));
#endif
}

}  // namespace phi

// TODO(yuwentao01) the embedded macro definition will get an error under
// windows, need to be solved in phi
#if NCCL_VERSION_CODE >= 21000
PD_REGISTER_KERNEL(all_gather,
                   GPU,
                   ALL_LAYOUT,
                   phi::AllGatherKernel,
                   float,
                   double,
                   int,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#else
PD_REGISTER_KERNEL(all_gather,
                   GPU,
                   ALL_LAYOUT,
                   phi::AllGatherKernel,
                   float,
                   double,
                   int,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif
