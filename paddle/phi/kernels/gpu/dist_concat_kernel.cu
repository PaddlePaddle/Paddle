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

#include "paddle/phi/kernels/dist_concat_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void DistConcatKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      int nranks,
                      DenseTensor* out) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  DenseTensor temp_out;
  auto temp_out_dims = x.dims();
  temp_out_dims[0] *= nranks;
  temp_out.Resize(temp_out_dims);
  dev_ctx.template Alloc<T>(&temp_out);

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
  comm_ctx->AllGather(&temp_out, x, stream);

  std::vector<DenseTensor> inputs;
  int axis = x.dims().size() - 1;
  auto out_dims = x.dims();
  out_dims[out_dims.size() - 1] *= nranks;
  int rows_per_tensor = x.dims()[0];
  int offset = 0;
  for (int i = 0; i < nranks; i++) {
    DenseTensor temp =
        temp_out.Slice(static_cast<int64_t>(offset),
                       static_cast<int64_t>(offset + rows_per_tensor));
    inputs.emplace_back(temp);
    offset += rows_per_tensor;
  }
  phi::funcs::ConcatFunctor<Context, T> functor;
  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);
  functor(dev_ctx, inputs, axis, out);
#else
  PADDLE_THROW(
      errors::PreconditionNotMet("PaddlePaddle should compile with GPU."));
#endif
}

}  // namespace phi

#if NCCL_VERSION_CODE >= 21000
PD_REGISTER_KERNEL(dist_concat,
                   GPU,
                   ALL_LAYOUT,
                   phi::DistConcatKernel,
                   float,
                   double,
                   int,
                   uint8_t,
                   int8_t,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(dist_concat,
                   GPU,
                   ALL_LAYOUT,
                   phi::DistConcatKernel,
                   float,
                   double,
                   int,
                   uint8_t,
                   int8_t,
                   int64_t,
                   bool,
                   phi::dtype::float16) {}
#endif
