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

#include "paddle/phi/kernels/reduce_kernel.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/phi/core/distributed/bkcl_comm_context.h"
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
#if defined(PADDLE_WITH_XPU_BKCL)
  out->Resize(x.dims());
  dev_ctx.template Alloc<T>(out);

  auto comm_ctx =
      static_cast<distributed::BKCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(
      comm_ctx,
      nullptr,
      errors::Unavailable("BKCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));
  XPUStream stream = nullptr;
  stream = comm_ctx->GetStream();
  PADDLE_ENFORCE_NOT_NULL(stream,
                          errors::NotFound("Should initialize NCCL firstly."));

  BKCLOp bkcl_red_type = BKCL_ADD;
  switch (static_cast<ReduceType>(reduce_type)) {
    case ReduceType::kRedSum:
      bkcl_red_type = BKCL_ADD;
      break;

    case ReduceType::kRedMax:
      bkcl_red_type = BKCL_MAX;
      break;

    case ReduceType::kRedMin:
      bkcl_red_type = BKCL_MIN;
      break;

    case ReduceType::kRedProd:
      bkcl_red_type = BKCL_PRODUCT;
      break;

    default:
      PADDLE_THROW(common::errors::InvalidArgument("Invalid reduce type: %d",
                                                   reduce_type));
  }
  comm_ctx->Reduce(out, x, bkcl_red_type, root, stream);
#else
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "PaddlePaddle should be compiled with XPU."));
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(reduce, XPU, ALL_LAYOUT, phi::ReduceKernel, float) {}
