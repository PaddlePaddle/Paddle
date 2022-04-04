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

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace experimental {

std::vector<Tensor> add_n_grad_impl(const std::vector<Tensor>& x,
                                    const Tensor& out_grad);

Tensor copy_to_impl(const Tensor& x, Place place, bool blocking);

std::vector<Tensor> split_impl(const Tensor& x,
                               const IntArray& num_or_sections,
                               const Scalar& axis);

std::tuple<Tensor, Tensor, Tensor> momentum_impl(
    const Tensor& param,
    const Tensor& grad,
    const Tensor& velocity,
    const Tensor& learning_rate,
    paddle::optional<const Tensor&> master_param,
    float mu,
    bool use_nesterov,
    const std::string& regularization_method,
    float regularization_coeff,
    bool multi_precision,
    float rescale_grad);

std::vector<Tensor> concat_grad_impl(const std::vector<Tensor>& x,
                                     const Tensor& out_grad,
                                     const Scalar& axis);

}  // namespace experimental
}  // namespace paddle
