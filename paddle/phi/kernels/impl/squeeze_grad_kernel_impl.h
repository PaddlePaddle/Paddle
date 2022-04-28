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
#include "paddle/phi/kernels/copy_kernel.h"

namespace phi {
template <typename T, typename Context>
void SqueezeGradKernel(const Context& dev_ctx,
                       const DenseTensor& xshape,
                       const DenseTensor& dout,
                       const std::vector<int>& axes,
                       DenseTensor* dx) {
  auto xshape_dims = xshape.dims();
  auto x_dims = phi::slice_ddim(xshape_dims, 1, xshape_dims.size());

  dev_ctx.template Alloc<T>(dx);
  phi::Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
  dx->Resize(x_dims);
}
}  // namespace phi
