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
#include "paddle/utils/optional.h"

namespace phi {

template <typename T, typename Context>
void AddGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   const DenseTensor& dout,
                   int axis,
                   DenseTensor* dx,
                   DenseTensor* dy);

template <typename T, typename Context>
void AddDoubleGradKernel(const Context& dev_ctx,
                         const DenseTensor& y,
                         paddle::optional<const DenseTensor&> ddx,
                         paddle::optional<const DenseTensor&> ddy,
                         const DenseTensor& dout,
                         int axis,
                         DenseTensor* ddout);

template <typename T, typename Context>
void AddTripleGradKernel(const Context& dev_ctx,
                         const DenseTensor& ddx,
                         const DenseTensor& ddy,
                         const DenseTensor& d_ddout,
                         int axis,
                         DenseTensor* d_ddx,
                         DenseTensor* d_ddy);

template <typename T, typename Context>
void SubtractGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const DenseTensor& dout,
                        int axis,
                        DenseTensor* dx,
                        DenseTensor* dy);

template <typename T, typename Context>
void SubtractDoubleGradKernel(const Context& dev_ctx,
                              const DenseTensor& y,
                              paddle::optional<const DenseTensor&> ddx,
                              paddle::optional<const DenseTensor&> ddy,
                              const DenseTensor& dout,
                              int axis,
                              DenseTensor* ddout);

template <typename T, typename Context>
void DivideGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      const DenseTensor& out,
                      const DenseTensor& dout,
                      int axis,
                      DenseTensor* dx,
                      DenseTensor* dy);

template <typename T, typename Context>
void DivideDoubleGradKernel(const Context& dev_ctx,
                            const DenseTensor& y,
                            const DenseTensor& out,
                            const DenseTensor& dx,
                            paddle::optional<const DenseTensor&> ddx,
                            paddle::optional<const DenseTensor&> ddy,
                            int axis,
                            DenseTensor* dy,
                            DenseTensor* dout,
                            DenseTensor* ddout);

template <typename T, typename Context>
void MultiplyGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const DenseTensor& dout,
                        int axis,
                        DenseTensor* dx,
                        DenseTensor* dy);

template <typename T, typename Context>
void MultiplyDoubleGradKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              paddle::optional<const DenseTensor&> ddx,
                              paddle::optional<const DenseTensor&> ddy,
                              int axis,
                              DenseTensor* dx,
                              DenseTensor* dy,
                              DenseTensor* ddout);

template <typename T, typename Context>
void MultiplyTripleGradKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              paddle::optional<const DenseTensor&> ddx,
                              paddle::optional<const DenseTensor&> ddy,
                              const DenseTensor& d_dx,
                              const DenseTensor& d_dy,
                              paddle::optional<const DenseTensor&> d_ddout,
                              int axis,
                              DenseTensor* d_x,
                              DenseTensor* d_y,
                              DenseTensor* d_dout,
                              DenseTensor* d_ddx,
                              DenseTensor* d_ddy);

template <typename T, typename Context>
void ElementwiseFMaxGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               const DenseTensor& out_grad,
                               int axis,
                               DenseTensor* x_grad,
                               DenseTensor* y_grad);

template <typename T, typename Context>
void ElementwiseFMinGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               const DenseTensor& out_grad,
                               int axis,
                               DenseTensor* x_grad,
                               DenseTensor* y_grad);

template <typename T, typename Context>
void MaximumGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       const DenseTensor& dout,
                       int axis,
                       DenseTensor* dx,
                       DenseTensor* dy);

template <typename T, typename Context>
void MinimumGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       const DenseTensor& dout,
                       int axis,
                       DenseTensor* dx,
                       DenseTensor* dy);
}  // namespace phi
