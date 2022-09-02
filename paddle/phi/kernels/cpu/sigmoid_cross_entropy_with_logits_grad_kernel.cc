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

#include "paddle/phi/kernels/sigmoid_cross_entropy_with_logits_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SigmoidCrossEntropyWithLogitsGradKernel(const Context& dev_ctx,
                                             const DenseTensor& x,
                                             const DenseTensor& label,
                                             const DenseTensor& out_grad,
                                             bool normalize,
                                             int ignore_index,
                                             DenseTensor* in_grad) {
  auto dx_data = dev_ctx.template Alloc<T>(in_grad);

  int limit = in_grad->numel();
  auto x_data = x.data<T>();
  auto label_data = label.data<T>();
  auto dout_data = out_grad.data<T>();
  for (int idx = 0; idx < limit; ++idx) {
    T x = x_data[idx];
    T label = label_data[idx];
    T dout = dout_data[idx];
    if (static_cast<int>(label) == ignore_index) {
      dx_data[idx] = static_cast<T>(0.);
    } else {
      T simoid_x = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
      T diff = simoid_x - label;
      dx_data[idx] = dout * diff;
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
    std::for_each(dx_data, dx_data + limit, [norm](T& v) { v = v / norm; });
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(sigmoid_cross_entropy_with_logits_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::SigmoidCrossEntropyWithLogitsGradKernel,
                   float,
                   double) {}
