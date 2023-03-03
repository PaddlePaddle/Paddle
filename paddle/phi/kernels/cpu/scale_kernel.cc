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

#include "paddle/phi/kernels/scale_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

template <typename T, typename Context>
void ScaleImpl(const Context& dev_ctx,
               const DenseTensor& x,
               const Scalar& scale,
               float bias,
               bool bias_after_scale,
               DenseTensor* out) {
  // calc
  dev_ctx.template Alloc<T>(out);
  auto eigen_out = phi::EigenVector<T>::Flatten(*out);
  auto eigen_x = phi::EigenVector<T>::Flatten(x);
  auto& dev = *dev_ctx.eigen_device();
  // TODO(chenweihang): now the eigen function here need the dtype of scale,
  // eigen_x, bias should be same, so here need cast for two scalar arg,
  // maybe we declare that the type of scale and bias is T?
  if (x.numel() <= 0 || (!x.IsInitialized())) {
    return;
  }
  phi::funcs::EigenScale<std::decay_t<decltype(dev)>, T>::Eval(
      dev,
      eigen_out,
      eigen_x,
      scale.to<T>(),
      static_cast<T>(bias),
      bias_after_scale);
}

template <typename T, typename Context>
typename std::enable_if<!std::is_integral<T>::value>::type ScaleKernelImpl(
    const Context& dev_ctx,
    const DenseTensor& x,
    const Scalar& scale,
    float bias,
    bool bias_after_scale,
    DenseTensor* out) {
  ScaleImpl<T, Context>(dev_ctx, x, scale, bias, bias_after_scale, out);
}

template <typename T, typename Context>
typename std::enable_if<std::is_integral<T>::value>::type ScaleKernelImpl(
    const Context& dev_ctx,
    const DenseTensor& x,
    const Scalar& scale,
    float bias,
    bool bias_after_scale,
    DenseTensor* out) {
  float float_scale = scale.to<float>();
  if (ceilf(float_scale) == float_scale) {
    // If `T` is an integer type and `scale` is an integer
    ScaleImpl<T, Context>(dev_ctx, x, scale, bias, bias_after_scale, out);
  } else {
    // If `T` is an integer type and `scale` is a float number
    // XXX: Current implementation includes two extra tensor casting operations
    // (with memory allocation) and is thus less efficient. T to float
    phi::DenseTensor float_x =
        phi::Cast<T, Context>(dev_ctx, x, DataType::FLOAT32);
    // Create temporary tensor
    phi::DenseTensor float_out =
        phi::EmptyLike<float, Context>(dev_ctx, float_x);
    // Calculate results of floating type
    ScaleImpl<float, Context>(
        dev_ctx, float_x, scale, bias, bias_after_scale, &float_out);
    // float to T
    phi::CastKernel<float, Context>(dev_ctx, float_out, out->dtype(), out);
  }
}

template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const Scalar& scale,
                 float bias,
                 bool bias_after_scale,
                 DenseTensor* out) {
  ScaleKernelImpl<T, Context>(dev_ctx, x, scale, bias, bias_after_scale, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(scale,
                   CPU,
                   ALL_LAYOUT,
                   phi::ScaleKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
