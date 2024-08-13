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
#include <string>

#include "paddle/common/hostdevice.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {
using Array1 = Eigen::DSizes<int64_t, 1>;
template <typename T>
struct KLDivLossBackward {
  bool log_target = false;

  HOSTDEVICE KLDivLossBackward(bool logTarget) : log_target(logTarget) {}

  HOSTDEVICE T operator()(const T& target, const T& grad) const {
    if (log_target) {
      return static_cast<T>(-1.) * std::exp(target) * grad;
    } else {
      if (target <= 0) {
        return 0;
      } else {
        return static_cast<T>(-1.) * target * grad;
      }
    }
  }
};

template <typename T, typename Context>
void KLDivLossGradKernel(const Context& dev_ctx,
                         const DenseTensor& x UNUSED,
                         const DenseTensor& label,
                         const DenseTensor& d_out,
                         const std::string& reduction,
                         bool log_target,
                         DenseTensor* d_x) {
  auto& place = *dev_ctx.eigen_device();
  auto* target = &label;
  auto* input_grad = d_x;
  auto* loss_grad = &d_out;

  const int n = input_grad->dims()[0];
  const int numel = input_grad->numel();
  const int expand = numel / loss_grad->numel();

  dev_ctx.template Alloc<T>(input_grad);

  auto target_t = phi::EigenVector<T>::Flatten(*target);

  auto input_grad_t = phi::EigenVector<T>::Flatten(*input_grad);
  auto loss_grad_t = phi::EigenVector<T>::Flatten(*loss_grad);

  auto loss_grad_expand = loss_grad_t.broadcast(Array1(expand));
  auto grad_t = loss_grad_expand;
  input_grad_t.device(place) =
      target_t.binaryExpr(grad_t, KLDivLossBackward<T>(log_target));

  if ("mean" == reduction) {
    input_grad_t.device(place) = input_grad_t / static_cast<T>(numel);
  } else if ("batchmean" == reduction) {
    input_grad_t.device(place) = input_grad_t / static_cast<T>(n);
  }
}
}  // namespace phi
