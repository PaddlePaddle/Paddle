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

#include "paddle/phi/kernels/label_smooth_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void LabelSmoothKernel(const Context& ctx,
                       const DenseTensor& label,
                       paddle::optional<const DenseTensor&> prior_dist,
                       float epsilon,
                       DenseTensor* out) {
  auto label_dim = label.dims()[label.dims().size() - 1];
  ctx.template Alloc<T>(out);
  auto& dev = *ctx.eigen_device();
  if (label_dim != 0) {
    auto eigen_out = EigenVector<T>::Flatten(*out);
    auto eigen_in = EigenVector<T>::Flatten(label);
    if (prior_dist.is_initialized()) {
      auto dist = EigenVector<T>::Flatten(*prior_dist.get_ptr());
      eigen_out.device(dev) =
          static_cast<T>(1 - epsilon) * eigen_in +
          static_cast<T>(epsilon) *
              dist.broadcast(Eigen::DSizes<int, 1>(label.numel() / label_dim));
    } else {
      eigen_out.device(dev) = static_cast<T>(1 - epsilon) * eigen_in +
                              static_cast<T>(epsilon / label_dim);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    label_smooth, CPU, ALL_LAYOUT, phi::LabelSmoothKernel, float, double) {}
