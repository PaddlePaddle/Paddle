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

template <typename T, typename Context>
void BReluKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 float t_min,
                 float t_max,
                 DenseTensor* out);

template <typename T, typename Context>
void LeakyReluKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     float alpha,
                     DenseTensor* out);

template <typename T, typename Context>
void ThresholdedReluKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           float threshold,
                           DenseTensor* out);

}  // namespace phi
