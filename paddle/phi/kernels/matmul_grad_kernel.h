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

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/utils/optional.h"

namespace phi {

template <typename T, typename Context>
void MatmulGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      const DenseTensor& dout,
                      bool transpose_x,
                      bool transpose_y,
                      DenseTensor* dx,
                      DenseTensor* dy);

template <typename T, typename Context>
void MatmulDoubleGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& y,
                            const DenseTensor& dout,
                            const paddle::optional<DenseTensor>& ddx,
                            const paddle::optional<DenseTensor>& ddy,
                            bool transpose_x,
                            bool transpose_y,
                            DenseTensor* dx,
                            DenseTensor* dy,
                            DenseTensor* ddout);

template <typename T, typename Context>
void MatmulTripleGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& y,
                            const DenseTensor& dout,
                            const paddle::optional<DenseTensor>& ddx,
                            const paddle::optional<DenseTensor>& ddy,
                            const paddle::optional<DenseTensor>& d_dx,
                            const paddle::optional<DenseTensor>& d_dy,
                            const paddle::optional<DenseTensor>& d_ddout,
                            bool transpose_x,
                            bool transpose_y,
                            DenseTensor* out_d_x,
                            DenseTensor* out_d_y,
                            DenseTensor* out_d_dout,
                            DenseTensor* out_d_ddx,
                            DenseTensor* out_d_ddy);

template <typename T, typename Context>
void MatmulWithFlattenGradKernel(const Context& dev_ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& y,
                                 const DenseTensor& out_grad,
                                 int x_num_col_dims,
                                 int y_num_col_dims,
                                 DenseTensor* x_grad,
                                 DenseTensor* y_grad);

template <typename T, typename Context>
void MatmulWithFlattenDoubleGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& y,
    const DenseTensor& out_grad,
    const paddle::optional<DenseTensor>& x_grad_grad,
    const paddle::optional<DenseTensor>& y_grad_grad,
    int x_num_col_dims,
    int y_num_col_dims,
    DenseTensor* x_grad,
    DenseTensor* y_grad,
    DenseTensor* out_grad_grad);

}  // namespace phi
