/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/infermeta/unary.h"

namespace phi {

#define DECLARE_ACTIVATION_KERNEL(name)   \
  template <typename T, typename Context> \
  void name##Kernel(                      \
      const Context& dev_ctx, const DenseTensor& x, DenseTensor* out);

#define DECLARE_ACTIVATION_KERNEL_WITH_ONE_ATTRS(name, attr) \
  template <typename T, typename Context>                    \
  void name##Kernel(const Context& dev_ctx,                  \
                    const DenseTensor& x,                    \
                    float attr,                              \
                    DenseTensor* out);

#define DECLARE_ACTIVATION_KERNEL_WITH_TWO_ATTRS(name, attr1, attr2) \
  template <typename T, typename Context>                            \
  void name##Kernel(const Context& dev_ctx,                          \
                    const DenseTensor& x,                            \
                    float attr1,                                     \
                    float attr2,                                     \
                    DenseTensor* out);

DECLARE_ACTIVATION_KERNEL(Sin)
DECLARE_ACTIVATION_KERNEL(Cos)
DECLARE_ACTIVATION_KERNEL(Tan)
DECLARE_ACTIVATION_KERNEL(Asin)
DECLARE_ACTIVATION_KERNEL(Atan)
DECLARE_ACTIVATION_KERNEL(Acos)
DECLARE_ACTIVATION_KERNEL(Sinh)
DECLARE_ACTIVATION_KERNEL(Cosh)
DECLARE_ACTIVATION_KERNEL(Asinh)
DECLARE_ACTIVATION_KERNEL(Acosh)
DECLARE_ACTIVATION_KERNEL(Atanh)
DECLARE_ACTIVATION_KERNEL(Relu)
DECLARE_ACTIVATION_KERNEL(Tanh)
DECLARE_ACTIVATION_KERNEL(TanhShrink)
DECLARE_ACTIVATION_KERNEL(Silu)
DECLARE_ACTIVATION_KERNEL(Exp)
DECLARE_ACTIVATION_KERNEL(Expm1)
DECLARE_ACTIVATION_KERNEL(Reciprocal)
DECLARE_ACTIVATION_KERNEL(Square)
DECLARE_ACTIVATION_KERNEL(Sqrt)
DECLARE_ACTIVATION_KERNEL(Rsqrt)
DECLARE_ACTIVATION_KERNEL(Softsign)
DECLARE_ACTIVATION_KERNEL(Sigmoid)
DECLARE_ACTIVATION_KERNEL(LogSigmoid)
DECLARE_ACTIVATION_KERNEL(Log)
DECLARE_ACTIVATION_KERNEL(Log2)
DECLARE_ACTIVATION_KERNEL(Log10)
DECLARE_ACTIVATION_KERNEL(Log1p)
DECLARE_ACTIVATION_KERNEL(Floor)
DECLARE_ACTIVATION_KERNEL(Ceil)
DECLARE_ACTIVATION_KERNEL(Negative)

DECLARE_ACTIVATION_KERNEL_WITH_ONE_ATTRS(LeakyRelu, alpha)
DECLARE_ACTIVATION_KERNEL_WITH_ONE_ATTRS(SoftShrink, lambda)
DECLARE_ACTIVATION_KERNEL_WITH_ONE_ATTRS(Mish, threshold)
DECLARE_ACTIVATION_KERNEL_WITH_ONE_ATTRS(HardShrink, threshold)
DECLARE_ACTIVATION_KERNEL_WITH_ONE_ATTRS(SoftShrink, lambda)
DECLARE_ACTIVATION_KERNEL_WITH_ONE_ATTRS(Elu, alpha)
DECLARE_ACTIVATION_KERNEL_WITH_ONE_ATTRS(Celu, alpha)
DECLARE_ACTIVATION_KERNEL_WITH_ONE_ATTRS(Logit, eps)

DECLARE_ACTIVATION_KERNEL_WITH_TWO_ATTRS(HardTanh, t_min, t_max)
DECLARE_ACTIVATION_KERNEL_WITH_TWO_ATTRS(STanh, scale_a, scale_b)
DECLARE_ACTIVATION_KERNEL_WITH_TWO_ATTRS(Softplus, beta, threshold)
DECLARE_ACTIVATION_KERNEL_WITH_TWO_ATTRS(HardSigmoid, slope, offset)
DECLARE_ACTIVATION_KERNEL_WITH_TWO_ATTRS(ThresholdedRelu, threshold, value)

template <typename T, typename Context>
void HardSwishKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     DenseTensor* out);

template <typename T, typename Context>
void Relu6Kernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out);

template <typename T, typename Context>
void RoundKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const int decimals,
                 DenseTensor* out);

template <typename T, typename Context>
void SwishKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out);

template <typename T, typename Context>
void PowKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const Scalar& factor,
               DenseTensor* out);

template <typename T, typename Context>
DenseTensor Pow(const Context& dev_ctx,
                const DenseTensor& x,
                const Scalar& factor) {
  DenseTensor out;
  MetaTensor meta_out(out);
  UnchangedInferMeta(x, &meta_out);
  PowKernel<T, Context>(dev_ctx, x, factor, &out);
  return out;
}

}  // namespace phi
