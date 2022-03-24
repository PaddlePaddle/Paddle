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
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {
using Array1 = Eigen::DSizes<int64_t, 1>;
template <typename T>
struct KLDivLossForward {
  HOSTDEVICE KLDivLossForward() {}

  HOSTDEVICE T operator()(const T& target, const T& input) const {
    if (target <= 0) {
      return 0;
    } else {
      return target * (std::log(target) - input);
    }
  }
};
template <typename T, typename Context>
void KLDivLossKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& label,
                     const std::string& reduction,
                     DenseTensor* out) {
  auto& place = *(dev_ctx.eigen_device());
  auto* input = &x;
  auto* target = &label;
  auto* loss = out;

  const int n = input->dims()[0];
  dev_ctx.template Alloc<T>(loss);

  auto input_t = phi::EigenVector<T>::Flatten(*input);
  auto target_t = phi::EigenVector<T>::Flatten(*target);
  auto loss_t = phi::EigenVector<T>::Flatten(*loss);
  auto output = target_t.binaryExpr(input_t, KLDivLossForward<T>());
  if ("none" == reduction) {
    loss_t.device(place) = output;
  } else if ("batchmean" == reduction) {
    auto output_sum = output.sum();
    if (n > 0) {
      loss_t.device(place) = output_sum / output_sum.constant(n);
    } else {
      loss_t.device(place) = output_sum;
    }
  } else if ("mean" == reduction) {
    loss_t.device(place) = output.mean();
  } else if ("sum" == reduction) {
    loss_t.device(place) = output.sum();
  }
}
}  // namespace phi
