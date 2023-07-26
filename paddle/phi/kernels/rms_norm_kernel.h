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
#include "paddle/phi/core/selected_rows.h"

namespace phi {

template <typename T, typename Context>
void RmsNormKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& weight,
                   const paddle::optional<DenseTensor>& bias,
                   float epsilon,
                   int begin_norm_axis,
                   DenseTensor* out);

template <typename T, typename Context>
void RmsNormWrapper(const Context& ctx,
                    const T* x,
                    const T* weight,
                    const T* bias,
                    const float epsilon,
                    const int rows,
                    const int cols,
                    T* output);

template <typename T, typename Context>
void ResidualAddRmsNormWrapper(const Context& ctx,
                               const T* x,
                               const T* residual,
                               const T* bias,
                               const T* norm_weight,
                               const T* norm_bias,
                               const float epsilon,
                               const int rows,
                               const int cols,
                               T* residual_output,
                               T* output);

template <typename T, typename Context>
void RmsNormInt8OutWrapper(const Context& ctx,
                           const T* x,
                           const T* weight,
                           const T* bias,
                           const float epsilon,
                           const int rows,
                           const int cols,
                           const float in_scale,
                           const int quant_round_type,
                           const float quant_max_bound,
                           const float quant_min_bound,
                           int8_t* output);

template <typename T, typename Context>
void ResidualAddRmsNormInt8OutWrapper(const Context& ctx,
                                      const T* x,
                                      const T* residual,
                                      const T* bias,
                                      const T* norm_weight,
                                      const T* norm_bias,
                                      const float epsilon,
                                      const int rows,
                                      const int cols,
                                      const float in_scale,
                                      const int quant_round_type,
                                      const float quant_max_bound,
                                      const float quant_min_bound,
                                      T* residual_output,
                                      int8_t* output);

}  // namespace phi
