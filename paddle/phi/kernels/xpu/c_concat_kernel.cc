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

#include <vector>
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"

#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/phi/core/distributed/bkcl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void CConcatKernel(const Context& dev_ctx,
                   const DenseTensor& x_in,
                   int rank,
                   int nranks,
                   int ring_id UNUSED,
                   bool use_calc_stream UNUSED,
                   bool use_model_parallel UNUSED,
                   DenseTensor* out) {
#if defined(PADDLE_WITH_XPU_BKCL)
  auto x = &x_in;
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

  phi::DenseTensor temp_out;
  phi::DDim temp_out_dims = x->dims();
  temp_out_dims[0] *= nranks;
  temp_out.Resize(temp_out_dims);
  dev_ctx.template Alloc(&temp_out, x->dtype());

  XPUStream stream = nullptr;
  phi::distributed::BKCLCommContext* comm_ctx = nullptr;
  comm_ctx =
      static_cast<phi::distributed::BKCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(comm_ctx,
                    nullptr,
                    common::errors::Unavailable(
                        "BKCLCommContext is nullptr, collective op should "
                        "has ring_id attr."));
  stream = dev_ctx.stream();
  comm_ctx->AllGather(&temp_out, *x, stream);

  std::vector<phi::DenseTensor> inputs;
  int axis = x->dims().size() - 1;
  auto out_dims = x->dims();
  out_dims[out_dims.size() - 1] *= nranks;
  int rows_per_tensor = x->dims()[0];
  int offset = 0;
  for (int i = 0; i < nranks; i++) {
    phi::DenseTensor temp = temp_out.Slice(offset, offset + rows_per_tensor);
    inputs.emplace_back(temp);
    offset += rows_per_tensor;
  }

  phi::funcs::ConcatFunctor<phi::XPUContext, T> functor;
  out->Resize(out_dims);
  dev_ctx.template Alloc(out, x->dtype());
  functor(dev_ctx, inputs, axis, out);
#else
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "PaddlePaddle should compile with XPU."));
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(c_concat,
                   XPU,
                   ALL_LAYOUT,
                   phi::CConcatKernel,
                   float,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
