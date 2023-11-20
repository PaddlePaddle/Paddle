// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/expand_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"

namespace phi {

template <typename T, typename Context>
void ExpandKernel(const Context& ctx,
                  const DenseTensor& x,
                  const IntArray& shape,
                  DenseTensor* out) {
  auto expand_shape = shape.GetData();
  auto diff = expand_shape.size() - x.dims().size();
  auto out_shape = common::vectorize<int64_t>(x.dims());
  out_shape.insert(out_shape.begin(), diff, 1);
  for (size_t i = 0; i < out_shape.size(); ++i) {
    PADDLE_ENFORCE_NE(
        expand_shape[i],
        0,
        phi::errors::InvalidArgument("The expanded size cannot be zero."));
    if (i < diff) {
      PADDLE_ENFORCE_GT(
          expand_shape[i],
          0,
          phi::errors::InvalidArgument(
              "The expanded size (%d) for non-existing dimensions must be "
              "positive for expand kernel.",
              expand_shape[i]));
      out_shape[i] = expand_shape[i];
    } else if (expand_shape[i] > 0) {
      if (out_shape[i] != 1) {
        PADDLE_ENFORCE_EQ(
            out_shape[i],
            expand_shape[i],
            phi::errors::InvalidArgument(
                "The value (%d) of the non-singleton dimension does not match"
                " the corresponding value (%d) in shape for expand kernel.",
                out_shape[i],
                expand_shape[i]));
      } else {
        out_shape[i] = expand_shape[i];
      }
    } else {
      PADDLE_ENFORCE_EQ(
          expand_shape[i],
          -1,
          phi::errors::InvalidArgument(
              "When the value in shape is negative for expand_v2 op, "
              "only -1 is supported, but the value received is %d.",
              expand_shape[i]));
    }
  }

  out->Resize(common::make_ddim(out_shape));
  ctx.template Alloc<T>(out);
  std::vector<const DenseTensor*> ins = {&x};
  std::vector<DenseTensor*> outs = {out};
  phi::funcs::BroadcastKernel<T>(ctx, ins, &outs, kps::IdentityFunctor<T>());
}

}  // namespace phi

PD_REGISTER_KERNEL(expand,
                   GPU,
                   ALL_LAYOUT,
                   phi::ExpandKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   int16_t,
                   uint8_t,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
