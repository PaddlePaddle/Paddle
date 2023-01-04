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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/impl/renorm_impl.h"
#include "paddle/phi/kernels/renorm_grad_kernel.h"

namespace phi {

template <typename T, typename Context>
void RenormGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& dout,
                      float p,
                      int axis,
                      float max_norm,
                      DenseTensor* dx) {
  int64_t numel = dout.numel();
  const T* dout_data = dout.template data<T>();
  const T* x_data = x.template data<T>();
  auto input_dims = x.dims();
  int dim = axis;
  auto dimension_each = input_dims[dim];
  dx->Resize(x.dims());
  dev_ctx.template Alloc<T>(dx);
  phi::funcs::RenormGradFunc(dev_ctx,
                             x_data,
                             dout_data,
                             dx->data<T>(),
                             p,
                             dim,
                             max_norm,
                             dimension_each,
                             input_dims,
                             numel);
}
}  // namespace phi
