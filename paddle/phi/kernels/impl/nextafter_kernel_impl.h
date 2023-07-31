/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <algorithm>
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/math.h"
#include "paddle/phi/kernels/nextafter_kernel.h"
namespace phi {
template <typename T>
struct NextafterOut {
  using type = T;
};

template <>
struct NextafterOut<int32_t> {
  using type = double;
};

template <>
struct NextafterOut<int64_t> {
  using type = double;
};
template <typename T>
struct NextafterFunctor {
  NextafterFunctor(const T* x,
                   const T* y,
                   typename NextafterOut<T>::type* out,
                   int64_t numel)
      : x_(x), y_(y), out_(out), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    out_[idx] = static_cast<typename NextafterOut<T>::type>(std::nextafter(
        static_cast<float>(x_[idx]), static_cast<float>(y_[idx])));
  }
  const T* x_;
  const T* y_;
  typename NextafterOut<T>::type* out_;
  int64_t numel_;
};
template <>
struct NextafterFunctor<double> {
  NextafterFunctor(const double* x, const double* y, double* out, int64_t numel)
      : x_(x), y_(y), out_(out), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    out_[idx] = std::nextafter(x_[idx], y_[idx]);
  }

  const double* x_;
  const double* y_;
  double* out_;
  int64_t numel_;
};

template <typename T, typename Context>
void NextafterKernel(const Context& ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  auto* out_data = ctx.template Alloc<T>(out);
  auto x_data = x.data<T>();
  auto y_data = y.data<T>();
  auto x_numel = x.numel();

  phi::funcs::ForRange<Context> for_range(ctx, x_numel);
  phi::NextafterFunctor<T> functor(x_data, y_data, out_data, x_numel);
  for_range(functor);
}

}  // namespace phi
