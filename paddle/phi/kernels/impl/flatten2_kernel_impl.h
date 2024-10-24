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

#pragma once
#include <vector>

#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/flatten_grad_kernel.h"
#include "paddle/phi/kernels/flatten_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/flatten2_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void Flatten2Kernel(const Context &dev_ctx,
                    const DenseTensor &x,
                    int axis,
                    DenseTensor *out,
                    DenseTensor *x_shape) {
  auto &axes = axis;

  auto *in = &x;
  auto x_dims = in->dims();

  auto out_dims = common::make_ddim(phi::funcs::GetOutputShape(axes, x_dims));

  dev_ctx.Alloc(out, x.dtype());
  phi::Copy(dev_ctx, *in, dev_ctx.GetPlace(), false, out);
  out->Resize(out_dims);
}

template <typename T, typename Context>
void Flatten2GradKernel(const Context &dev_ctx,
                        const DenseTensor &x,
                        const DenseTensor &x_shape,
                        const DenseTensor &out_grad,
                        int axis,
                        DenseTensor *x_grad) {
  auto *d_x = x_grad;
  auto *d_out = &out_grad;

  auto xshape_dims = x_shape.dims();
  auto x_dims = common::slice_ddim(xshape_dims, 1, xshape_dims.size());

  dev_ctx.Alloc(x_grad, out_grad.dtype());
  phi::Copy(dev_ctx, *d_out, dev_ctx.GetPlace(), false, d_x);
  d_x->Resize(x_dims);
}
}  // namespace phi
