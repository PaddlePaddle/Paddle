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

#include "paddle/phi/kernels/dropout_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void DropoutNdGradKernel(const Context& dev_ctx,
                         const DenseTensor& mask,
                         const DenseTensor& out_grad,
                         const Scalar& p,
                         bool is_test,
                         const std::string& mode,
                         const std::vector<int>& axis,
                         DenseTensor* x_grad) {
  auto* grad_x = x_grad;
  auto* grad_y = &out_grad;
  dev_ctx.template Alloc<T>(grad_x);

  auto dX = EigenVector<T>::Flatten(*grad_x);
  auto dY = EigenVector<T>::Flatten(*grad_y);
  float prob = p.to<float>();

  auto& place = *dev_ctx.eigen_device();
  auto& dropout_implementation = mode;
  if (is_test == true) {
    if (dropout_implementation == "upscale_in_train") {
      dX.device(place) = static_cast<T>(1) * dY;
    } else {
      dX.device(place) = dY * static_cast<T>(1.0f - prob);
    }
  } else {
    std::vector<int64_t> out_dims = common::vectorize(out_grad.dims());
    auto M = EigenVector<uint8_t>::Flatten(mask);
    if (dropout_implementation == "upscale_in_train") {
      if (prob == 1.0f) {
        dX.device(place) = static_cast<T>(0) * dY;
      } else {
        if (axis.empty()) {
          dX.device(place) = dY * M.cast<T>() / static_cast<T>(1.0f - prob);
        } else {
          dX.device(place) = dY * M.broadcast(out_dims).cast<T>() /
                             static_cast<T>(1.0f - prob);
        }
      }
    } else {
      if (axis.empty()) {
        dX.device(place) = dY * M.cast<T>();
      } else {
        dX.device(place) = dY * M.broadcast(out_dims).cast<T>();
      }
    }
  }
}

template <typename T, typename Context>
void DropoutGradRawKernel(const Context& dev_ctx,
                          const DenseTensor& mask,
                          const DenseTensor& out_grad,
                          const Scalar& p,
                          bool is_test,
                          const std::string& mode,
                          DenseTensor* x_grad) {
  DropoutNdGradKernel<T, Context>(
      dev_ctx, mask, out_grad, p.to<float>(), is_test, mode, {}, x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(dropout_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::DropoutGradRawKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(
    dropout_nd_grad, CPU, ALL_LAYOUT, phi::DropoutNdGradKernel, float, double) {
}
