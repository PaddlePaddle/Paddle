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

#include <unsupported/Eigen/SpecialFunctions>
#include "paddle/fluid/platform/for_range.h"
#include "paddle/pten/core/dense_tensor.h"

namespace pten {

template <typename T>
struct DigammaGradFunctor {
  DigammaGradFunctor(const T* dout, const T* x, T* output, int64_t numel)
      : dout_(dout), x_(x), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = dout_[idx] * Eigen::numext::polygamma(T(1), x_[idx]);
  }

 private:
  const T* dout_;
  const T* x_;
  T* output_;
  int64_t numel_;
};

template <typename T, typename Context>
void DigammaGradKernel(const Context& ctx,
                       const DenseTensor& out_grad,
                       const DenseTensor& x,
                       DenseTensor* x_grad) {
  x_grad->mutable_data<T>(ctx.GetPlace());

  auto* dout_data = out_grad.data<T>();
  auto* x_data = x.data<T>();
  auto* dx_data = x_grad->data<T>();
  auto numel = out_grad.numel();
  paddle::platform::ForRange<Context> for_range(ctx, numel);
  DigammaGradFunctor<T> functor(dout_data, x_data, dx_data, numel);
  for_range(functor);
}

}  // namespace pten
