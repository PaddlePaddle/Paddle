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

#include "paddle/phi/kernels/reduce_v2_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void ReduceV2Kernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    int root,
                    int reduce_type,
                    DenseTensor* out) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  out->Resize(x.dims());
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

  ncclRedOp_t red_type = ncclSum;
  switch (reduce_type) {
    case kRedSum:
      red_type = ncclSum;
      break;
    case kRedMax:
      red_type = ncclMax;
      break;
    case kRedMin:
      red_type = ncclMin;
      break;
    case kRedProd:
      red_type = ncclProd;
      break;
  }
  comm_ctx->Reduce(out, x, red_type, root, stream);
#else
  PADDLE_THROW(
      errors::PreconditionNotMet("PaddlePaddle should compile with GPU."));
#endif
}

}  // namespace phi

#if NCCL_VERSION_CODE >= 21000
PD_REGISTER_KERNEL(reduce_v2,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReduceV2Kernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int64_t,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(reduce_v2,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReduceV2Kernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int64_t,
                   phi::dtype::float16) {}
#endif
