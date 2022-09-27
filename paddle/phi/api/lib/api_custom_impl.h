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

Tensor add_n_impl(const std::vector<Tensor>& x);

Tensor copy_to_impl(const Tensor& x, Place place, bool blocking);

////////////////// Backward(grad) api impls //////////////////////

void imag_grad_impl(const Tensor& out_grad, Tensor* x_grad);

void embedding_grad_impl(const Tensor& x,
                         const Tensor& weight,
                         const Tensor& out_grad,
                         int64_t padding_idx,
                         bool sparse,
                         Tensor* weight_grad);

void real_grad_impl(const Tensor& out_grad, Tensor* x_grad);

}  // namespace experimental
}  // namespace paddle
