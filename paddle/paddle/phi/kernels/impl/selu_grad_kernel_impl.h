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
#include "paddle/phi/kernels/impl/selu_kernel_impl.h"

namespace phi {
template <typename T, typename Context>
void SeluGradKernel(const Context& dev_ctx,
                    const DenseTensor& out,
                    const DenseTensor& dout,
                    float scale,
                    float alpha,
                    DenseTensor* dx) {
  auto dx_ptr = dev_ctx.template Alloc<T>(dx);
  SeluGradFunctor<T> functor(
      out.data<T>(), dout.data<T>(), alpha, scale, dx_ptr);
  size_t limit = static_cast<size_t>(out.numel());
  phi::funcs::ForRange<Context> for_range(dev_ctx, limit);
  for_range(functor);
}
}  // namespace phi
