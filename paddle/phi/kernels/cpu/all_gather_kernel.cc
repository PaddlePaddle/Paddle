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

#if defined(PADDLE_WITH_GLOO)
#include "paddle/phi/core/distributed/gloo_comm_context.h"
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/core/distributed/xccl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void AllGatherKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     int nranks,
                     DenseTensor* out) {
#if defined(PADDLE_WITH_GLOO)
  dev_ctx.template Alloc<T>(out);
  auto out_dims = x.dims();
  out_dims[0] *= nranks;
  out->Resize(out_dims);

  auto comm_ctx =
      static_cast<distributed::GlooCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_EQ(
      nranks,
      comm_ctx->GetSize(),
      errors::InvalidArgument(
          "nranks: %s should equal to %s", nranks, comm_ctx->GetSize()));

  comm_ctx->AllGather(out, x);
#else
  PADDLE_THROW(errors::Unavailable(
      "PaddlePaddle should compile with GLOO by setting WITH_GLOO=ON"));
#endif
}

#ifdef PADDLE_WITH_CUSTOM_DEVICE
template <typename T>
void AllGatherKernel(const phi::CustomContext& dev_ctx,
                     const DenseTensor& x,
                     int nranks,
                     DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto out_dims = x.dims();
  out_dims[0] *= nranks;
  out->Resize(out_dims);

  auto comm_ctx =
      static_cast<distributed::XCCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_EQ(
      nranks,
      comm_ctx->GetSize(),
      errors::InvalidArgument(
          "nranks: %s should equal to %s", nranks, comm_ctx->GetSize()));

  comm_ctx->AllGather(out, x, *dev_ctx.GetStream());
}
#endif
}  // namespace phi

PD_REGISTER_KERNEL(all_gather,
                   CPU,
                   ALL_LAYOUT,
                   phi::AllGatherKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#ifdef PADDLE_WITH_CUSTOM_DEVICE
PD_REGISTER_KERNEL(all_gather,
                   Custom,
                   ALL_LAYOUT,
                   phi::AllGatherKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif
