// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

namespace phi {

template <typename T, typename Context>
void HingeLossKernel(const Context& dev_ctx,
                     const DenseTensor& logits,
                     const DenseTensor& labels,
                     DenseTensor* loss) {
  auto* pred = &logits;
  auto* label = &labels;

  auto& place = *dev_ctx.eigen_device();

  auto x = phi::EigenVector<T>::Flatten(*pred);
  auto y = phi::EigenVector<T>::Flatten(*label);
  dev_ctx.template Alloc<T>(loss);
  auto l = phi::EigenVector<T>::Flatten(*loss);
  phi::funcs::EigenHingeLoss<std::decay_t<decltype(place)>, T>::Eval(
      place, l, x, y);
}

template <typename T, typename Context>
void HingeLossGradKernel(const Context& dev_ctx,
                         const DenseTensor& logits,
                         const DenseTensor& labels,
                         const DenseTensor& loss_grad,
                         DenseTensor* logits_grad) {
  auto* pred = &logits;
  auto* label = &labels;
  auto* dloss = &loss_grad;
  auto* dpred = logits_grad;
  auto& place = *dev_ctx.eigen_device();

  auto x = phi::EigenVector<T>::Flatten(*pred);
  auto y = phi::EigenVector<T>::Flatten(*label);
  auto dl = phi::EigenVector<T>::Flatten(*dloss);

  if (dpred) {
    dev_ctx.template Alloc<T>(dpred);
    auto dx = phi::EigenVector<T>::Flatten(*dpred);
    phi::funcs::EigenHingeLossGrad<std::decay_t<decltype(place)>, T>::Eval(
        place, dx, dl, x, y);
  }
}

}  // namespace phi
