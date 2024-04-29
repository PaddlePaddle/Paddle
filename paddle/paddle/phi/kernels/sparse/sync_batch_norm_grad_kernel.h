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

#include <string>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void SyncBatchNormCooGradKernel(
    const Context& dev_ctx,
    const SparseCooTensor& x,
    const DenseTensor& scale,
    const DenseTensor& bias,
    const DenseTensor& saved_mean,
    const DenseTensor& saved_variance,
    const paddle::optional<DenseTensor>& reserve_space,
    const SparseCooTensor& y_grad,
    float momentum,
    float epsilon,
    const std::string& data_layout,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics,
    SparseCooTensor* x_grad,
    DenseTensor* scale_grad,
    DenseTensor* bias_grad);

}  // namespace sparse
}  // namespace phi
