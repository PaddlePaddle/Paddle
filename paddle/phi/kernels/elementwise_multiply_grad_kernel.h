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
                              const paddle::optional<DenseTensor>& ddx,
                              const paddle::optional<DenseTensor>& ddy,
                              int axis,
                              DenseTensor* dx,
                              DenseTensor* dy,
                              DenseTensor* ddout);

template <typename T, typename Context>
void MultiplyTripleGradKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              const paddle::optional<DenseTensor>& ddx,
                              const paddle::optional<DenseTensor>& ddy,
                              const DenseTensor& d_dx,
                              const DenseTensor& d_dy,
                              const paddle::optional<DenseTensor>& d_ddout,
                              int axis,
                              DenseTensor* d_x,
                              DenseTensor* d_y,
                              DenseTensor* d_dout,
                              DenseTensor* d_ddx,
                              DenseTensor* d_ddy);

}  // namespace phi
