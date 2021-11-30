/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/pten/api/include/tensor.h"
#include "paddle/pten/common/scalar.h"

namespace paddle {
namespace experimental {

PADDLE_API Tensor add(const Tensor& x, const Tensor& y);

PADDLE_API Tensor subtract(const Tensor& x, const Tensor& y);

PADDLE_API Tensor divide(const Tensor& x, const Tensor& y);

PADDLE_API Tensor multiply(const Tensor& x, const Tensor& y);

// TODO(chenweihang): move mean API into stat.h/cc
PADDLE_API Tensor mean(const Tensor& x,
                       const std::vector<int64_t>& axis,
                       bool keep_dim);

PADDLE_API Tensor sum(const Tensor& x,
                      const std::vector<int64_t>& axis,
                      DataType dtype,
                      bool keep_dim);

// TODO(chenweihang): Follow-up discussion on the handling of `act` argument
PADDLE_API Tensor scale(const Tensor& x,
                        const Scalar& scale,
                        float bias,
                        bool bias_after_scale);

}  // namespace experimental
}  // namespace paddle
