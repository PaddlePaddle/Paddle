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
#include "paddle/phi/kernels/atan2_kernel.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {
template <typename T>
struct Atan2Out {
  using type = T;
};

template <>
struct Atan2Out<int32_t> {
  using type = double;
};

template <>
struct Atan2Out<int64_t> {
  using type = double;
};

template <typename T>
struct Atan2Functor {
  Atan2Functor(const T* x1,
               const T* x2,
               typename Atan2Out<T>::type* out,
               int64_t numel)
      : x1_(x1), x2_(x2), out_(out), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    out_[idx] = static_cast<typename Atan2Out<T>::type>(
        ::atan2f(static_cast<float>(x1_[idx]), static_cast<float>(x2_[idx])));
  }

  const T* x1_;
  const T* x2_;
  typename Atan2Out<T>::type* out_;
  int64_t numel_;
};

template <>
struct Atan2Functor<double> {
  Atan2Functor(const double* x1, const double* x2, double* out, int64_t numel)
      : x1_(x1), x2_(x2), out_(out), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    out_[idx] = ::atan2(x1_[idx], x2_[idx]);
  }

  const double* x1_;
  const double* x2_;
  double* out_;
  int64_t numel_;
};

template <typename T, typename Context>
void Atan2Kernel(const Context& ctx,
                 const DenseTensor& x,
                 const DenseTensor& y,
                 DenseTensor* out) {
  auto numel = x.numel();
  auto x_data = x.data<T>();
  auto y_data = y.data<T>();

  auto* out_data = ctx.template Alloc<typename Atan2Out<T>::type>(
      out, size_t(x.numel() * sizeof(typename Atan2Out<T>::type)));

  phi::funcs::ForRange<Context> for_range(ctx, numel);
  phi::Atan2Functor<T> functor(x_data, y_data, out_data, numel);
  for_range(functor);
}

}  // namespace phi
