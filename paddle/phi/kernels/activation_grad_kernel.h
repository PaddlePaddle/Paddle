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

#define DECLARE_ACTIVATION_GRAD_KERNEL_DepX(name) \
  template <typename T, typename Context>         \
  void name##GradKernel(const Context& dev_ctx,   \
                        const DenseTensor& x,     \
                        const DenseTensor& dout,  \
                        DenseTensor* dx);

#define DECLARE_ACTIVATION_GRAD_KERNEL_DepOut(name) \
  template <typename T, typename Context>           \
  void name##GradKernel(const Context& dev_ctx,     \
                        const DenseTensor& out,     \
                        const DenseTensor& dout,    \
                        DenseTensor* dx);

template <typename T, typename Context>
void ReluDoubleGradKernel(const Context& dev_ctx,
                          const DenseTensor& out,
                          const DenseTensor& ddx,
                          DenseTensor* ddout);

template <typename T, typename Context>
void TanhDoubleGradKernel(const Context& dev_ctx,
                          const DenseTensor& out,
                          const DenseTensor& ddx,
                          const DenseTensor& dout,
                          DenseTensor* dout_new,
                          DenseTensor* ddout);

template <typename T, typename Context>
void TanhTripleGradKernel(const Context& dev_ctx,
                          const DenseTensor& out,
                          const DenseTensor& ddx,
                          const DenseTensor& dout,
                          const DenseTensor& d_ddout,
                          const DenseTensor& d_dout_new,
                          DenseTensor* d_out_new,
                          DenseTensor* d_dout,
                          DenseTensor* d_ddx);

template <typename T, typename Context>
void BReluGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& dout,
                     float t_min,
                     float t_max,
                     DenseTensor* dx);

template <typename T, typename Context>
void LeakyReluGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dout,
                         float alpha,
                         DenseTensor* dx);

template <typename T, typename Context>
void LeakyReluDoubleGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& ddx,
                               float alpha,
                               DenseTensor* ddout);

template <typename T, typename Context>
void ThresholdedReluGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& dout,
                               float threshold,
                               DenseTensor* dx);

DECLARE_ACTIVATION_GRAD_KERNEL_DepX(Cos);
DECLARE_ACTIVATION_GRAD_KERNEL_DepX(Tan);
DECLARE_ACTIVATION_GRAD_KERNEL_DepX(Acos);
DECLARE_ACTIVATION_GRAD_KERNEL_DepX(Sin);
DECLARE_ACTIVATION_GRAD_KERNEL_DepX(Asin);
DECLARE_ACTIVATION_GRAD_KERNEL_DepX(Atan);
DECLARE_ACTIVATION_GRAD_KERNEL_DepX(Sinh);
DECLARE_ACTIVATION_GRAD_KERNEL_DepX(Cosh);
DECLARE_ACTIVATION_GRAD_KERNEL_DepX(Asinh);
DECLARE_ACTIVATION_GRAD_KERNEL_DepX(Acosh);
DECLARE_ACTIVATION_GRAD_KERNEL_DepX(Atanh);
DECLARE_ACTIVATION_GRAD_KERNEL_DepOut(Relu);
DECLARE_ACTIVATION_GRAD_KERNEL_DepOut(Tanh);
DECLARE_ACTIVATION_GRAD_KERNEL_DepOut(Exp);

}  // namespace phi
