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
#include "paddle/utils/optional.h"

namespace phi {

#define DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(name) \
  template <typename T, typename Context>         \
  void name##GradKernel(const Context& dev_ctx,   \
                        const DenseTensor& x,     \
                        const DenseTensor& dout,  \
                        DenseTensor* dx);

#define DECLARE_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(name, attr) \
  template <typename T, typename Context>                       \
  void name##GradKernel(const Context& dev_ctx,                 \
                        const DenseTensor& x,                   \
                        const DenseTensor& dout,                \
                        float attr,                             \
                        DenseTensor* dx);

#define DECLARE_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPX(name, attr1, attr2) \
  template <typename T, typename Context>                               \
  void name##GradKernel(const Context& dev_ctx,                         \
                        const DenseTensor& x,                           \
                        const DenseTensor& dout,                        \
                        float attr1,                                    \
                        float attr2,                                    \
                        DenseTensor* dx);

#define DECLARE_ACTIVATION_GRAD_KERNEL_DEPOUT(name) \
  template <typename T, typename Context>           \
  void name##GradKernel(const Context& dev_ctx,     \
                        const DenseTensor& out,     \
                        const DenseTensor& dout,    \
                        DenseTensor* dx);

#define DECLARE_ACTIVATION_GRAD_KERNEL_NODEP(name) \
  template <typename T, typename Context>          \
  void name##GradKernel(                           \
      const Context& dev_ctx, const DenseTensor& dout, DenseTensor* dx);

#define DECLARE_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPOUT(name, attr) \
  template <typename T, typename Context>                         \
  void name##GradKernel(const Context& dev_ctx,                   \
                        const DenseTensor& out,                   \
                        const DenseTensor& dout,                  \
                        float attr,                               \
                        DenseTensor* dx);

#define DECLARE_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPOUT(name, attr1, attr2) \
  template <typename T, typename Context>                                 \
  void name##GradKernel(const Context& dev_ctx,                           \
                        const DenseTensor& out,                           \
                        const DenseTensor& dout,                          \
                        float attr1,                                      \
                        float attr2,                                      \
                        DenseTensor* dx);

template <typename T, typename Context>
void SiluGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& out,
                    const DenseTensor& dout,
                    DenseTensor* dx);
template <typename T, typename Context>
void ReluDoubleGradKernel(const Context& dev_ctx,
                          const DenseTensor& out,
                          const DenseTensor& ddx,
                          DenseTensor* ddout);

template <typename T, typename Context>
void SinDoubleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dout,
                         const DenseTensor& ddx,
                         DenseTensor* dx,
                         DenseTensor* ddout);

template <typename T, typename Context>
void CosDoubleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dout,
                         const DenseTensor& ddx,
                         DenseTensor* dx,
                         DenseTensor* ddout);

template <typename T, typename Context>
void TanhDoubleGradKernel(const Context& dev_ctx,
                          const DenseTensor& out,
                          const DenseTensor& dout,
                          const DenseTensor& ddx,
                          DenseTensor* dout_new,
                          DenseTensor* ddout);

template <typename T, typename Context>
void TanhTripleGradKernel(const Context& dev_ctx,
                          const DenseTensor& out,
                          const DenseTensor& dout,
                          const DenseTensor& ddx,
                          const paddle::optional<DenseTensor>& d_dout_new,
                          const paddle::optional<DenseTensor>& d_ddout,
                          DenseTensor* d_out_new,
                          DenseTensor* d_dout,
                          DenseTensor* d_ddx);

template <typename T, typename Context>
void SinTripleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const paddle::optional<DenseTensor>& dout,
                         const paddle::optional<DenseTensor>& ddx,
                         const DenseTensor& d_dx_new,
                         const paddle::optional<DenseTensor>& d_ddout,
                         DenseTensor* d_x_new,
                         DenseTensor* d_dout,
                         DenseTensor* d_ddx);

template <typename T, typename Context>
void CosTripleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const paddle::optional<DenseTensor>& dout,
                         const paddle::optional<DenseTensor>& ddx,
                         const DenseTensor& d_dx_new,
                         const paddle::optional<DenseTensor>& d_ddout,
                         DenseTensor* d_x_new,
                         DenseTensor* d_dout,
                         DenseTensor* d_ddx);

template <typename T, typename Context>
void LeakyReluDoubleGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& ddx,
                               float alpha,
                               DenseTensor* ddout);

template <typename T, typename Context>
void EluGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& out,
                   const DenseTensor& dout,
                   float alpha,
                   DenseTensor* dx);

template <typename T, typename Context>
void EluDoubleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dout,
                         const DenseTensor& ddx,
                         float alpha,
                         DenseTensor* dx,
                         DenseTensor* ddout);

template <typename T, typename Context>
void SigmoidDoubleGradKernel(const Context& dev_ctx,
                             const DenseTensor& out,
                             const DenseTensor& dout,
                             const DenseTensor& ddx,
                             DenseTensor* dout_new,
                             DenseTensor* ddout);

template <typename T, typename Context>
void SigmoidTripleGradKernel(const Context& dev_ctx,
                             const DenseTensor& out,
                             const DenseTensor& dout,
                             const DenseTensor& ddx,
                             const DenseTensor& d_dout_new,
                             const paddle::optional<DenseTensor>& d_ddout,
                             DenseTensor* d_out_new,
                             DenseTensor* d_dout,
                             DenseTensor* d_ddx);

template <typename T, typename Context>
void LogDoubleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dout,
                         const DenseTensor& ddx,
                         DenseTensor* dx,
                         DenseTensor* ddout);

template <typename T, typename Context>
void SqrtDoubleGradKernel(const Context& dev_ctx,
                          const DenseTensor& out,
                          const DenseTensor& dx,
                          const DenseTensor& ddx,
                          DenseTensor* dout,
                          DenseTensor* ddout);

template <typename T, typename Context>
void RsqrtDoubleGradKernel(const Context& dev_ctx,
                           const DenseTensor& out,
                           const DenseTensor& dx,
                           const DenseTensor& ddx,
                           DenseTensor* dout,
                           DenseTensor* ddout);

template <typename T, typename Context>
void CeluDoubleGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& dout,
                          const DenseTensor& ddx,
                          float alpha,
                          DenseTensor* dx,
                          DenseTensor* ddout);

template <typename T, typename Context>
void SquareDoubleGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& dout,
                            const DenseTensor& ddx,
                            DenseTensor* dx,
                            DenseTensor* ddout);

template <typename T, typename Context>
void HardSwishGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dout,
                         DenseTensor* dx);

template <typename T, typename Context>
void PowGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& dout,
                   const Scalar& factor,
                   DenseTensor* dx);

template <typename T, typename Context>
void PowDoubleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dout,
                         const DenseTensor& ddx,
                         const Scalar& factor,
                         DenseTensor* dx,
                         DenseTensor* ddout);

template <typename T, typename Context>
void PowTripleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dout,
                         const DenseTensor& ddx,
                         const DenseTensor& d_dx,
                         const paddle::optional<DenseTensor>& d_ddout,
                         const Scalar& factor,
                         DenseTensor* out_d_x,
                         DenseTensor* out_d_dout,
                         DenseTensor* out_d_ddx);

template <typename T, typename Context>
void SoftplusDoubleGradKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& dout,
                              const DenseTensor& ddx,
                              float beta,
                              float threshold,
                              DenseTensor* dx,
                              DenseTensor* ddout);

DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(Cos);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(Tan);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(Acos);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(Sin);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(Asin);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(Atan);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(Sinh);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(Cosh);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(Asinh);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(Acosh);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(Atanh);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(TanhShrink);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(Square);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(Softsign);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(LogSigmoid);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(Log);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(Log2);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(Log10);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(Log1p);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPX(Swish);

DECLARE_ACTIVATION_GRAD_KERNEL_DEPOUT(Exp);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPOUT(Expm1);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPOUT(Reciprocal);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPOUT(Rsqrt);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPOUT(Relu);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPOUT(Tanh);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPOUT(Sigmoid);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPOUT(Sqrt);
DECLARE_ACTIVATION_GRAD_KERNEL_DEPOUT(Relu6);

DECLARE_ACTIVATION_GRAD_KERNEL_NODEP(Round);
DECLARE_ACTIVATION_GRAD_KERNEL_NODEP(Floor);
DECLARE_ACTIVATION_GRAD_KERNEL_NODEP(Ceil);

DECLARE_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(LeakyRelu, alpha);
DECLARE_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(SoftShrink, lambda);
DECLARE_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(HardShrink, threshold);
DECLARE_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(Logit, eps);
DECLARE_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(Mish, threshold);
DECLARE_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(Celu, alpha);

DECLARE_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPX(HardTanh, t_min, t_max);
DECLARE_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPX(STanh, scale_a, scale_b);
DECLARE_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPX(Softplus, beta, threshold);
DECLARE_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPOUT(HardSigmoid, slope, offset);
DECLARE_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPX(ThresholdedRelu, threshold, value);

}  // namespace phi
