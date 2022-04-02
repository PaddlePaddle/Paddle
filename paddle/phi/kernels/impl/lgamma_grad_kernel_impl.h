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
#include "paddle/phi/kernels/funcs/for_range.h"
namespace phi {
template <typename T>
struct LgammaGradFunctor {
  LgammaGradFunctor(const T* dout, const T* x, T* output, int64_t numel)
      : dout_(dout), x_(x), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = dout_[idx] * Eigen::numext::digamma(x_[idx]);
  }

 private:
  const T* dout_;
  const T* x_;
  T* output_;
  int64_t numel_;
};
template <typename T, typename Context>
void LgammaGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& d_out,
                      DenseTensor* d_x) {
  auto numel = d_out.numel();
  auto* dout_data = d_out.data<T>();
  auto* x_data = x.data<T>();
  auto* dx_data =
      dev_ctx.template Alloc<T>(d_x, static_cast<size_t>(numel * sizeof(T)));
  phi::funcs::ForRange<Context> for_range(dev_ctx, numel);
  LgammaGradFunctor<T> functor(dout_data, x_data, dx_data, numel);
  for_range(functor);
}
}  // namespace phi
