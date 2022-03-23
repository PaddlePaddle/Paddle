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

DECLARE_ACTIVATION_KERNEL(Cos)
DECLARE_ACTIVATION_KERNEL(Tan)
DECLARE_ACTIVATION_KERNEL(Acos)
DECLARE_ACTIVATION_KERNEL(Sin)
DECLARE_ACTIVATION_KERNEL(Asin)
DECLARE_ACTIVATION_KERNEL(Atan)
DECLARE_ACTIVATION_KERNEL(Sinh)
DECLARE_ACTIVATION_KERNEL(Cosh)
DECLARE_ACTIVATION_KERNEL(Asinh)
DECLARE_ACTIVATION_KERNEL(Acosh)
DECLARE_ACTIVATION_KERNEL(Atanh)
DECLARE_ACTIVATION_KERNEL(Relu)
DECLARE_ACTIVATION_KERNEL(Tanh)
DECLARE_ACTIVATION_KERNEL(TanhShrink)
DECLARE_ACTIVATION_KERNEL(Silu)
DECLARE_ACTIVATION_KERNEL(Sigmoid)
DECLARE_ACTIVATION_KERNEL(LogSigmoid)
DECLARE_ACTIVATION_KERNEL(Log)
DECLARE_ACTIVATION_KERNEL(Log2)
DECLARE_ACTIVATION_KERNEL(Log10)
DECLARE_ACTIVATION_KERNEL(Log1p)

DECLARE_ACTIVATION_KERNEL_WITH_ONE_ATTRS(LeakyRelu, alpha)
DECLARE_ACTIVATION_KERNEL_WITH_ONE_ATTRS(ThresholdedRelu, threshold)
DECLARE_ACTIVATION_KERNEL_WITH_ONE_ATTRS(SoftShrink, lambda)
DECLARE_ACTIVATION_KERNEL_WITH_ONE_ATTRS(HardShrink, threshold)
DECLARE_ACTIVATION_KERNEL_WITH_ONE_ATTRS(Elu, alpha)

DECLARE_ACTIVATION_KERNEL_WITH_TWO_ATTRS(BRelu, t_min, t_max)
DECLARE_ACTIVATION_KERNEL_WITH_TWO_ATTRS(HardSigmoid, slope, offset)
}  // namespace phi
