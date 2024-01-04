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

#include <string>

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void BatchNormKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& mean,
                     const DenseTensor& variance,
                     const paddle::optional<DenseTensor>& scale,
                     const paddle::optional<DenseTensor>& bias,
                     bool is_test,
                     float momentum,
                     float epsilon,
                     const std::string& data_layout,
                     bool use_global_stats,
                     bool trainable_statistics,
                     DenseTensor* y,
                     DenseTensor* mean_out,
                     DenseTensor* variance_out,
                     DenseTensor* saved_mean,
                     DenseTensor* saved_variance,
                     DenseTensor* reserve_space);

template <typename T, typename Context>
void BatchNormInferKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& mean,
                          const DenseTensor& variance,
                          const DenseTensor& scale,
                          const DenseTensor& bias,
                          float momentum,
                          float epsilon,
                          const std::string& data_layout,
                          DenseTensor* y,
                          DenseTensor* mean_out,
                          DenseTensor* variance_out);
#define PD_DECLARE_BN_GRAD_FUNCTOR(dtype, backend)                         \
  template void phi::BatchNormGradFunctor<dtype, ::phi::backend##Context>( \
      const ::phi::backend##Context& dev_ctx,                              \
      const DenseTensor& x,                                                \
      const paddle::optional<DenseTensor>& scale,                          \
      const paddle::optional<DenseTensor>& bias,                           \
      const paddle::optional<DenseTensor>& mean,                           \
      const paddle::optional<DenseTensor>& variance,                       \
      const DenseTensor& saved_mean,                                       \
      const DenseTensor& saved_variance,                                   \
      const paddle::optional<DenseTensor>& reserve_space,                  \
      const DenseTensor& y_grad,                                           \
      float momentum,                                                      \
      float epsilon,                                                       \
      const std::string& data_layout,                                      \
      bool is_test,                                                        \
      bool use_global_stats,                                               \
      bool trainable_statistics,                                           \
      bool is_inplace,                                                     \
      DenseTensor* x_grad,                                                 \
      DenseTensor* scale_grad,                                             \
      DenseTensor* bias_grad)
}  // namespace phi
