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

#include <tuple>
#include <vector>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace experimental {

// NOTE: Separate forward and backward(grad) api impl
// NOTE: The api_impl in this file are arranged in alphabetic order.

////////////////// Forward api impls //////////////////////

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> adam_impl(
    const Tensor& param,
    const Tensor& grad,
    const Tensor& learning_rate,
    const Tensor& moment1,
    const Tensor& moment2,
    const Tensor& beta1_pow,
    const Tensor& beta2_pow,
    paddle::optional<const Tensor&> master_param,
    paddle::optional<const Tensor&> skip_update,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    bool lazy_mode,
    int64_t min_row_size_to_use_multithread,
    bool multi_precision,
    bool use_global_beta_pow);

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> adamw_impl(
    const Tensor& param,
    const Tensor& grad,
    const Tensor& learning_rate,
    const Tensor& moment1,
    const Tensor& moment2,
    const Tensor& beta1_pow,
    const Tensor& beta2_pow,
    paddle::optional<const Tensor&> master_param,
    paddle::optional<const Tensor&> skip_update,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    float lr_ratio,
    float coeff,
    bool with_decay,
    bool lazy_mode,
    int64_t min_row_size_to_use_multithread,
    bool multi_precision,
    bool use_global_beta_pow);

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> batch_norm_impl(
    const Tensor& x,
    const Tensor& scale,
    const Tensor& bias,
    const Tensor& mean,
    const Tensor& variance,
    float momentum,
    float epsilon,
    const std::string& data_layout,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics,
    bool fuse_with_relu);

Tensor conv2d_impl(const Tensor& input,
                   const Tensor& filter,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   const std::string& paddding_algorithm,
                   int groups,
                   const std::vector<int>& dilations,
                   const std::string& data_format,
                   bool use_addto,
                   int workspace_size_MB,
                   bool exhaustive_search);

std::vector<std::vector<Tensor>> conv2d_grad_impl(
    const Tensor& input,
    const Tensor& filter,
    const Tensor& out_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& paddding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format,
    bool use_addto,
    int workspace_size_MB,
    bool exhaustive_search);

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

std::tuple<Tensor, Tensor> sgd_impl(
    const Tensor& param,
    const Tensor& learning_rate,
    const Tensor& grad,
    paddle::optional<const Tensor&> master_param,
    bool multi_precision);

////////////////// Backward(grad) api impls //////////////////////

std::vector<Tensor> add_n_grad_impl(const std::vector<Tensor>& x,
                                    const Tensor& out_grad);

Tensor imag_grad_impl(const Tensor& x);

Tensor real_grad_impl(const Tensor& x);

}  // namespace experimental
}  // namespace paddle
