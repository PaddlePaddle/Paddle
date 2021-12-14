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

#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/hybird/eigen/common.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/float16.h"

namespace pten {

// TODO(chenweihang): replaced by include public context header
using CPUContext = paddle::platform::CPUDeviceContext;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
using CUDAContext = paddle::platform::CUDADeviceContext;
#endif

template <typename T, typename ContextT>
void Scale(const ContextT& dev_ctx,
           const DenseTensor& x,
           const Scalar& scale,
           float bias,
           bool bias_after_scale,
           DenseTensor* out) {
  // calc
  out->mutable_data<T>();
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
      static_cast<T>(scale.to<float>()),
      static_cast<T>(bias),
      bias_after_scale);
}

}  // namespace pten

using float16 = paddle::platform::float16;

// TODO(chenweihang): Use EigenContext to specialize the ContextT parameter,
// and only register the backend as Eigen's kernel during registration,
// instead of using macros to register the CPU and CUDA kernels separately

PT_REGISTER_CTX_KERNEL(scale,
                       CPU,
                       ANY,
                       pten::Scale,
                       float,
                       double,
                       paddle::platform::bfloat16,
                       uint8_t,
                       int8_t,
                       int16_t,
                       int,
                       int64_t) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_REGISTER_CTX_KERNEL(scale,
                       CUDA,
                       ANY,
                       pten::Scale,
                       float,
                       double,
                       float16,
                       uint8_t,
                       int8_t,
                       int16_t,
                       int,
                       int64_t) {}
#endif
