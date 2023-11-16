// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/kernels/copysign_grad_kernel.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename T>
struct CopySignGradFunctor {
  CopySignGradFunctor(
      const T* x_data, const T* y_data, const T* dout, T* dx, int64_t numel)
      : x_data_(x_data), y_data_(y_data), dout_(dout), dx_(dx), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    if (x_data_[idx] == T(0))
      dx_[idx] = T(0);
    else
      dx_[idx] = T(dout_[idx]) *
                 (T(copysign(x_data_[idx], y_data_[idx]) / x_data_[idx]));
  }

  const T* x_data_;
  const T* y_data_;
  const T* dout_;
  T* dx_;
  int64_t numel_;
};

template <typename T, typename Context>
void CopySignGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const DenseTensor& out_grad,
                        DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  auto x_data = x.data<T>(), y_data = y.data<T>(),
       out_grad_data = out_grad.data<T>();
  auto x_grad_data = x_grad->data<T>();
  phi::funcs::ForRange<Context> for_range(dev_ctx, x.numel());
  phi::CopySignGradFunctor<T> functor(
      x_data, y_data, out_grad_data, x_grad_data, x.numel());
  for_range(functor);
}
}  // namespace  phi
