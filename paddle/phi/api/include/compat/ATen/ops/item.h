// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <ATen/core/Tensor.h>
#include <ATen/ops/_local_scalar_dense.h>

namespace at {}  // namespace at

namespace at {

inline at::Scalar Tensor::item() const {
  auto numel = this->sym_numel();
  PD_CHECK(numel == 1,
           "a Tensor with ",
           numel,
           " elements cannot be converted to Scalar");
  if (this->is_sparse()) {
    if (this->_nnz() == 0) return Scalar(0);
    if (this->is_coalesced()) return at::_local_scalar_dense(this->_values());
    return at::_local_scalar_dense(this->_values().sum());
  } else {
    return _local_scalar_dense(*this);
  }
}

template <typename T>
T Tensor::item() const {
  return item().to<T>();
}

}  // namespace at
