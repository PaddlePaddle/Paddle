/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/kernels/scale_kernel.h"

#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/funcs/eigen/common.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/pten/common/bfloat16.h"
namespace pten {

template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const Scalar& scale,
                 float bias,
                 bool bias_after_scale,
                 DenseTensor* out) {
  // calc
  out->mutable_data<T>(dev_ctx.GetPlace());
  auto eigen_out = pten::EigenVector<T>::Flatten(*out);
  auto eigen_x = pten::EigenVector<T>::Flatten(x);
  auto& dev = *dev_ctx.eigen_device();
  // TODO(chenweihang): now the eigen function here need the dtype of scale,
  // eigen_x, bias should be same, so here need cast for two scalar arg,
  // maybe we declare that the type of scale and bias is T?
  paddle::operators::EigenScale<std::decay_t<decltype(dev)>, T>::Eval(
      dev,
      eigen_out,
      eigen_x,
      scale.to<T>(),
      static_cast<T>(bias),
      bias_after_scale);
}

}  // namespace pten

PT_REGISTER_KERNEL(scale,
                   CPU,
                   ALL_LAYOUT,
                   pten::ScaleKernel,
                   float,
                   double,
                   paddle::platform::bfloat16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
