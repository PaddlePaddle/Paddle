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

#include "paddle/phi/kernels/reduce_kernel.h"

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
void ReduceKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  int root,
                  int reduce_type,
                  DenseTensor* out) {
  PADDLE_ENFORCE_GT(x.numel(),
                    0,
                    common::errors::InvalidArgument(
                        "Tensor need be reduced must not empty."));
#if defined(PADDLE_WITH_GLOO)
  out->Resize(x.dims());
  dev_ctx.template Alloc<T>(out);

  auto comm_ctx =
      static_cast<distributed::GlooCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(
      comm_ctx,
      nullptr,
      errors::Unavailable("NCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));
  comm_ctx->Reduce(out, x, reduce_type, root);
#else
  PADDLE_THROW(
      errors::PreconditionNotMet("PaddlePaddle should compile with GPU."));
#endif
}

#ifdef PADDLE_WITH_CUSTOM_DEVICE
template <typename T>
void ReduceKernel(const phi::CustomContext& dev_ctx,
                  const DenseTensor& x,
                  int root,
                  int reduce_type,
                  DenseTensor* out) {
  PADDLE_ENFORCE_GT(x.numel(),
                    0,
                    common::errors::InvalidArgument(
                        "Tensor need be reduced must not empty."));
  out->Resize(x.dims());
  dev_ctx.template Alloc<T>(out);

  auto comm_ctx =
      static_cast<distributed::XCCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(
      comm_ctx,
      nullptr,
      errors::Unavailable("XCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));
  comm_ctx->Reduce(out,
                   x,
                   phi::ccl::ToXCCLReduceOp(reduce_type),
                   root,
                   *dev_ctx.GetStream());
}
#endif

}  // namespace phi

PD_REGISTER_KERNEL(reduce,
                   CPU,
                   ALL_LAYOUT,
                   phi::ReduceKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int64_t,
                   phi::dtype::float16) {}

#ifdef PADDLE_WITH_CUSTOM_DEVICE
PD_REGISTER_KERNEL(reduce,
                   Custom,
                   ALL_LAYOUT,
                   phi::ReduceKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int64_t,
                   phi::dtype::float16) {}
#endif
