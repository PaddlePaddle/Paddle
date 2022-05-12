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
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/cpu/index_fill_impl.h"
#include "paddle/phi/kernels/cpu/reduce.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"
#include "paddle/phi/kernels/index_fill_grad_kernel.h"

namespace phi {

template <typename T, typename Context>
void IndexFillTensorGradKernel(const Context& dev_ctx,
                               const MetaTensor& out_grad,
                               const MetaTensor& index,
                               const Scalar& axis_scalar,
                               DenseTensor* x_grad,
                               DenseTensor* fill_tensor_grad) {
  DenseTensor raw_fill_tensor_grad;
  raw_fill_tensor_grad.Resize(out_grad.dims());
  raw_fill_tensor_grad->set_dtype(out_grad.dtype);
  dev_ctx.template Alloc<T>(&raw_fill_tensor_grad);
  phi::funcs::SetConstant<Context, T> set_zero;
  set_zero(dev_ctx, raw_fill_tensor_grad, static_cast<T>(0));

  float fill_val = 0.0;
  IndexFillBaseKernel<T, Context>(dev_ctx,
                                  out_grad,
                                  index_arr,
                                  axis_scalar,
                                  fill_val,
                                  x_grad,
                                  &raw_fill_tensor_grad);

  // sum
  phi::Reduce<CPUContext, T, phi::funcs::SumFunctor>(dev_ctx,
                                                     &raw_fill_tensor_grad,
                                                     true,
                                                     dims,
                                                     false,
                                                     out_grad.dtype(),
                                                     fill_tensor_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_fill_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::IndexFillGradKernel,
                   bool,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   double,
                   int,
                   int64_t) {}
