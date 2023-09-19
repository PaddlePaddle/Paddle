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

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/huber_loss_grad_kernel.h"

namespace phi {

template <typename T>
struct HuberLossBackward {
  HOSTDEVICE HuberLossBackward(const T& delta, T sign)
      : sign(sign), delta(delta) {}

  HOSTDEVICE T operator()(const T& val) const {
    T abs_val = abs(val);
    if (abs_val <= delta) {
      return sign * val;
    } else {
      if (val > static_cast<T>(0)) {
        return sign * delta;
      } else {
        return static_cast<T>(-1) * sign * delta;
      }
    }
  }

  T sign;
  T delta;
};

template <typename T, typename Context>
void HuberLossGradKernel(const Context& dev_ctx,
                         const DenseTensor& residual,
                         const DenseTensor& out_grad,
                         float delta,
                         DenseTensor* input_grad,
                         DenseTensor* label_grad) {
  T delta_ = static_cast<T>(delta);
  auto& place = *dev_ctx.eigen_device();

  auto eigen_residual = EigenVector<T>::Flatten(residual);
  auto eigen_out_grad = EigenVector<T>::Flatten(out_grad);

  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    auto eigen_input_grad = EigenVector<T>::Flatten(*input_grad);
    eigen_input_grad.device(place) = eigen_residual.unaryExpr(
        HuberLossBackward<T>(delta_, static_cast<T>(-1.0)));
    eigen_input_grad.device(place) = eigen_out_grad * eigen_input_grad;
  }

  if (label_grad) {
    dev_ctx.template Alloc<T>(label_grad);
    auto eigen_label_grad = EigenVector<T>::Flatten(*label_grad);
    eigen_label_grad.device(place) = eigen_residual.unaryExpr(
        HuberLossBackward<T>(delta_, static_cast<T>(1.0)));
    eigen_label_grad.device(place) = eigen_out_grad * eigen_label_grad;
  }
}

}  // namespace phi
