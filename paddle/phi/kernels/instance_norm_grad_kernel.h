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

namespace phi {

template <typename T, typename Context>
void InstanceNormGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const paddle::optional<DenseTensor>& scale,
                            const DenseTensor& saved_mean,
                            const DenseTensor& saved_variance,
                            const DenseTensor& y_grad,
                            float epsilon,
                            DenseTensor* x_grad,
                            DenseTensor* scale_grad,
                            DenseTensor* bias_grad);

template <typename T, typename Context>
void InstanceNormDoubleGradKernel(const Context& dev_ctx,
                                  const DenseTensor& x,
                                  const paddle::optional<DenseTensor>& scale,
                                  const DenseTensor& saved_mean,
                                  const DenseTensor& saved_variance,
                                  const DenseTensor& dy,
                                  const paddle::optional<DenseTensor>& ddx,
                                  const paddle::optional<DenseTensor>& ddscale,
                                  const paddle::optional<DenseTensor>& ddbias,
                                  float epsilon,
                                  DenseTensor* dx,
                                  DenseTensor* dscale,
                                  DenseTensor* ddy);

}  // namespace phi
