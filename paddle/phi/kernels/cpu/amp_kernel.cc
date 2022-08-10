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

#include "paddle/phi/kernels/amp_kernel.h"

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void CheckFiniteAndUnscaleKernel(const Context& dev_ctx,
                                 const std::vector<const DenseTensor*>& xs,
                                 const DenseTensor& scale,
                                 std::vector<DenseTensor*> outs,
                                 DenseTensor* found_infinite) {
  const T* scale_data = scale.data<T>();
  bool* found_inf_data = dev_ctx.template Alloc<bool>(found_infinite);

  *found_inf_data = false;
  DenseTensor is_finite = Empty<bool>(dev_ctx, {1});
  bool* is_finite_data = is_finite.template data<bool>();

  auto& dev = *dev_ctx.eigen_device();

  T inverse_scale = 1.0 / *scale_data;
  for (size_t i = 0; i < xs.size(); ++i) {
    const auto* x = xs[i];
    auto* out = outs[i];
    dev_ctx.template Alloc<T>(out);
    if (!(*found_inf_data)) {
      paddle::framework::TensorIsfinite(*x, &is_finite);
      *found_inf_data = !(*is_finite_data);
    }
    auto eigen_out = EigenVector<T>::Flatten(*out);
    auto eigen_in = EigenVector<T>::Flatten(*x);
    if (!(*found_inf_data)) {
      eigen_out.device(dev) = eigen_in * inverse_scale;
    } else {
      eigen_out.device(dev) = eigen_in * static_cast<T>(0);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(check_finite_and_unscale,
                   CPU,
                   ALL_LAYOUT,
                   phi::CheckFiniteAndUnscaleKernel,
                   float,
                   double) {}
