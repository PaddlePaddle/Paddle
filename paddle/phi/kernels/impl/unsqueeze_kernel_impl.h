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
#include "paddle/phi/kernels/funcs/unsqueeze.h"

namespace phi {
template <typename T, typename Context>
void UnsqueezeKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const ScalarArray& axes,
                     DenseTensor* xshape,
                     DenseTensor* out) {
  auto x_dims = x.dims();
  auto out_dims = out->dims();
  if (axes.FromTensor()) {
    std::vector<int32_t> tmp;
    tmp.reserve(axes.GetData().size());
    std::for_each(axes.GetData().begin(),
                  axes.GetData().end(),
                  [&tmp](const int64_t& t) { tmp.push_back(t); });
    out_dims = funcs::GetUnsqueezeShape(tmp, x_dims);
  }
  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);
  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  out->Resize(out_dims);  // copy will reset the dims.
}
}  // namespace phi
