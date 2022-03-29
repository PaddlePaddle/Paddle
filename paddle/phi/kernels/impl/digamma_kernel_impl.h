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
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename T>
struct DigammaFunctor {
  DigammaFunctor(const T* input, T* output, int64_t numel)
      : input_(input), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = Eigen::numext::digamma(input_[idx]);
  }

 private:
  const T* input_;
  T* output_;
  int64_t numel_;
};

template <typename T, typename Context>
void DigammaKernel(const Context& ctx, const DenseTensor& x, DenseTensor* out) {
  out->mutable_data<T>(ctx.GetPlace());
  auto* x_data = x.data<T>();
  auto* out_data = out->data<T>();
  auto numel = x.numel();
  phi::funcs::ForRange<Context> for_range(ctx, numel);
  DigammaFunctor<T> functor(x_data, out_data, numel);
  for_range(functor);
}

}  // namespace phi
