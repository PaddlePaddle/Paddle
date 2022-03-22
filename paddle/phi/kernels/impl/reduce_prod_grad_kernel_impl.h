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

#include "paddle/phi/kernels/reduce_grad_kernel.h"

#include "paddle/phi/kernels/funcs/reduce_functor.h"
#include "paddle/phi/kernels/impl/reduce_grad.h"

namespace phi {

template <typename T, typename Context>
void ReduceProdGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& out,
                          const DenseTensor& out_grad,
                          const std::vector<int64_t>& dims,
                          bool keep_dim,
                          bool reduce_all,
                          DataType in_dtype,
                          DataType out_dtype,
                          DenseTensor* x_grad) {
  ReduceGradKernel<Context, T, funcs::ProdGradFunctor>(dev_ctx,
                                                       x,
                                                       out,
                                                       out_grad,
                                                       dims,
                                                       keep_dim,
                                                       reduce_all,
                                                       in_dtype,
                                                       out_dtype,
                                                       x_grad);
}

}  // namespace phi
