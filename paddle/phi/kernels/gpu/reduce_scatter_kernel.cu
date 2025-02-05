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

#include "paddle/phi/kernels/reduce_scatter_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void ReduceScatterKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         int nranks,
                         DenseTensor* out) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  gpuStream_t stream = dev_ctx.stream();
  auto comm_context =
      static_cast<distributed::NCCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(
      comm_context,
      nullptr,
      errors::Unavailable("NCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));

  auto out_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      out_dims[0] % nranks,
      0,
      errors::InvalidArgument("The input tensor X's "
                              "dim[0] (%d) should be divisible by nranks(%d)",
                              out_dims[0],
                              nranks));
  out_dims[0] = out_dims[0] / nranks;

  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);
  comm_context->ReduceScatter(out, x, ncclSum, stream);
#else
  PADDLE_THROW(
      errors::PreconditionNotMet("PaddlePaddle should compile with GPU."));
#endif
}

}  // namespace phi

#if NCCL_VERSION_CODE >= 21000
PD_REGISTER_KERNEL(reduce_scatter,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReduceScatterKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int64_t,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(reduce_scatter,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReduceScatterKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int64_t,
                   phi::dtype::float16) {}
#endif
