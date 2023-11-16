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
#include "paddle/phi/kernels/copysign_kernel.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename T>
struct CopySignFunctor {
  CopySignFunctor(const T* x_data, const T* y_data, T* out_data, int64_t numel)
      : x_data_(x_data), y_data_(y_data), out_data_(out_data), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    out_data_[idx] = T(copysign(x_data_[idx], y_data_[idx]));
  }

  const T* x_data_;
  const T* y_data_;
  T* out_data_;
  int64_t numel_;
};

template <typename T, typename Context>
void CopySignKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto x_data = x.data<T>(), y_data = y.data<T>();
  auto out_data = out->data<T>();
  phi::funcs::ForRange<Context> for_range(dev_ctx, x.numel());
  phi::CopySignFunctor<T> functor(x_data, y_data, out_data, x.numel());
  for_range(functor);
}
}  // namespace  phi
