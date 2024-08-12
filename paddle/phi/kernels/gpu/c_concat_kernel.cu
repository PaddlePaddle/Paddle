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

#include "paddle/phi/kernels/c_concat_kernel.h"

// #include "paddle/phi/backends/all_context.h"
// #include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void CConcatKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   int nranks,
                   int rank,
                   bool use_model_parallel,
                   DenseTensor* out) {
  PADDLE_ENFORCE_GE(rank,
                    0,
                    common::errors::PreconditionNotMet(
                        "The value of rank (%d) for c_concat must be "
                        "greater than or equal to 0.",
                        rank));
  PADDLE_ENFORCE_GE(nranks,
                    2,
                    common::errors::PreconditionNotMet(
                        "The value of nranks (%d) for c_concat must be "
                        "greater than or equal to 2.",
                        nranks));
  PADDLE_ENFORCE_LT(rank,
                    nranks,
                    common::errors::PreconditionNotMet(
                        "The value of rank (%d) for c_concat must be "
                        "less than that of nranks (%d).",
                        rank,
                        nranks));
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  phi::DenseTensor temp_out;
  phi::DDim temp_out_dims = x.dims();
  temp_out.Resize(temp_out_dims);
  dev_ctx.template Alloc<T>(temp_out);

  auto comm_ctx =
      static_cast<distributed::NCCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(
      comm_ctx,
      nullptr,
      errors::Unavailable("NCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));
  gpuStream_t stream = comm_ctx->GetStream();
  comm_ctx->AllGather(&temp_out, *x, stream);

  std::vector<phi::DenseTensor> inputs;
  int axis = x.dims().size() - 1;
  int rows_per_tensor = x.dims()[0];
  int offset = 0;
  for (int i = 0; i < nranks; i++) {
    phi::DenseTensor temp = temp_out.Slice(offset, offset + rows_per_tensor);
    inputs.emplace_back(temp);
    offset += rows_per_tensor;
  }

  auto out_dims = x.dims();
  out_dims[out_dims.size() - 1] *= nranks;
  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);

  phi::funcs::ConcatFunctor<phi::GPUContext, T> functor;
  auto& dev_ctx2 = ctx.template device_context<phi::GPUContext>();
  functor(dev_ctx2, inputs, axis, out);
#else
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "PaddlePaddle should compile with GPU."));
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(c_concat_v2,
                   GPU,
                   ALL_LAYOUT,
                   ops::CConcatKernel,
                   float,
                   double,
                   int,
                   int64_t,
#if (NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000) || \
    defined(PADDLE_WITH_HIP)
                   phi::dtype::bfloat16,
#endif
                   phi::dtype::float16) {
}
