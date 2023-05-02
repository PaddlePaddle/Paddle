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

namespace phi {

template <typename T, typename Context>
void LogLossGradKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const DenseTensor& label,
                       const DenseTensor& out_grad,
                       float epsilon,
                       DenseTensor* in_grad) {
  auto prediction = EigenVector<T>::Flatten(input);
  auto label_out = EigenVector<T>::Flatten(label);

  auto dl = EigenVector<T>::Flatten(out_grad);
  auto& place = *dev_ctx.eigen_device();

  if (in_grad) {
    dev_ctx.template Alloc<T>(in_grad);
    auto dx = EigenVector<T>::Flatten(*in_grad);
    phi::funcs::EigenLogLossGrad<std::decay_t<decltype(place)>, T>::Eval(
        place, dx, dl, prediction, label_out, epsilon);
  }
}

}  // namespace phi
