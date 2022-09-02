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

#pragma once

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"
#include "paddle/phi/kernels/impl/reduce_grad.h"
#include "paddle/phi/kernels/reduce_prod_grad_kernel.h"

namespace phi {

template <typename T, typename Context>
void ReduceProdGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& out,
                          const DenseTensor& out_grad,
                          const IntArray& dims,
                          bool keep_dim,
                          bool reduce_all,
                          DenseTensor* x_grad) {
  ReduceGradKernel<Context, T, funcs::ProdGradFunctor>(
      dev_ctx, x, out, out_grad, dims.GetData(), keep_dim, reduce_all, x_grad);
}

}  // namespace phi
