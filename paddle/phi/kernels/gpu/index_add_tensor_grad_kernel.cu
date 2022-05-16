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

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"
#include "paddle/phi/kernels/gpu/index_add_funcs.h"
#include "paddle/phi/kernels/gpu/reduce.h"
// #include "paddle/phi/kernels/index_add_grad_kernel.h"
#include "paddle/phi/kernels/index_add_tensor_grad_kernel.h"

namespace phi {

template <typename T, typename Context>
void IndexAddTensorGradKernel(const Context& dev_ctx,
                               const DenseTensor& out_grad,
                               const IntArray& index_arr,
                               const Scalar& axis_scalar,
                               DenseTensor* x_grad,
                               DenseTensor* add_tensor_grad) {
  DenseTensor raw_add_tensor_grad;
  raw_add_tensor_grad.Resize(out_grad.dims());
  dev_ctx.template Alloc<T>(&raw_add_tensor_grad);
  phi::funcs::SetConstant<Context, T> set_zero;
  set_zero(dev_ctx, &raw_add_tensor_grad, static_cast<T>(0));

    // TODO: see index_add_tensor_grad_kernel.cc
  phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
  float add_value = 0.0;
  IndexAddBaseKernel<T, Context>(dev_ctx,
                                  out_grad,
                                  index_arr,
                                  axis_scalar,
                                  add_value,
                                //   x_grad,
                                  nullptr, 
                                  &raw_add_tensor_grad);

  phi::Reduce<T, kps::AddFunctor, kps::IdentityFunctor>(
      dev_ctx,
      raw_add_tensor_grad,
      true,
      vectorize(out_grad.dims()),
      false,
      out_grad.dtype(),
      add_tensor_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_add_tensor_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexAddTensorGradKernel,
                   bool,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   double,
                   int,
                   int64_t) {}