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

#include "paddle/phi/kernels/sigmoid_cross_entropy_with_logits_kernel.h"

#include <algorithm>
#include <limits>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SigmoidCrossEntropyWithLogitsKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& label,
    const paddle::optional<DenseTensor>& pos_weight,
    bool normalize,
    int ignore_index,
    DenseTensor* out) {
  auto out_data = dev_ctx.template Alloc<T>(out);
  int limit = static_cast<int>(out->numel());
  auto x_data = x.data<T>();
  auto label_data = label.data<T>();
  auto pos_weight_data =
      (pos_weight.get_ptr() == nullptr ? nullptr
                                       : pos_weight.get_ptr()->data<T>());

  for (int idx = 0; idx < limit; ++idx) {
    T x = x_data[idx];
    T label = label_data[idx];
    if (static_cast<int>(label) == ignore_index) {
      out_data[idx] = static_cast<T>(0.);
    } else {
      if (pos_weight_data == nullptr) {
        T term1 = (x > 0) ? x : 0;
        T term2 = x * label;
        T term3 = std::log(static_cast<T>(1) + std::exp(-std::abs(x)));
        out_data[idx] = term1 - term2 + term3;
      } else {
        T max_val = x < 0 ? -x : 0;
        T term1 = (static_cast<T>(1.) - label) * x;
        T term2 = std::log(std::exp(-max_val) + std::exp(-x - max_val));
        out_data[idx] = term1 + pos_weight_data[idx] * (term2 + max_val);
      }
    }
  }

  if (normalize) {
    int norm = 0;
    T eps = static_cast<T>(1e-6);
    for (int idx = 0; idx < limit; ++idx) {
      T diff = label_data[idx] - static_cast<T>(ignore_index);
      if ((diff < -eps) || (diff > eps)) {
        norm += 1;
      }
    }
    eps = static_cast<T>(1e-5);
    norm = norm > eps ? norm : eps;
    std::for_each(out_data, out_data + limit, [norm](T& v) { v = v / norm; });
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(sigmoid_cross_entropy_with_logits,
                   CPU,
                   ALL_LAYOUT,
                   phi::SigmoidCrossEntropyWithLogitsKernel,
                   float,
                   double) {}
