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

#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/huber_loss_kernel.h"

namespace phi {

template <typename T>
struct HuberLossForward {
  HOSTDEVICE HuberLossForward(const T& delta) : delta(delta) {}

  HOSTDEVICE T operator()(const T& val) const {
    T abs_val = std::abs(val);
    if (abs_val <= delta) {
      return static_cast<T>(0.5) * val * val;
    } else {
      return delta * (abs_val - static_cast<T>(0.5) * delta);
    }
  }

  T delta;
};

template <typename T, typename Context>
void HuberLossKernel(const Context& dev_ctx,
                     const DenseTensor& input,
                     const DenseTensor& label,
                     float delta,
                     DenseTensor* out,
                     DenseTensor* residual) {
  T delta_ = static_cast<T>(delta);
  auto& place = *dev_ctx.eigen_device();

  auto x = EigenVector<T>::Flatten(input);
  auto y = EigenVector<T>::Flatten(label);

  dev_ctx.template Alloc<T>(residual);
  auto eigen_residual = EigenVector<T>::Flatten(*residual);
  eigen_residual.device(place) = y - x;

  dev_ctx.template Alloc<T>(out);
  auto loss = EigenVector<T>::Flatten(*out);
  loss.device(place) = eigen_residual.unaryExpr(HuberLossForward<T>(delta_));
}

}  // namespace phi
