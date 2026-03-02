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
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"

namespace at {}  // namespace at

namespace at {

inline int64_t Tensor::_nnz() const {
  PD_CHECK(this->is_sparse(),
           "_nnz expected sparse tensor layout but got ",
           layout());
  if (tensor_.layout() == common::DataLayout::SPARSE_COO) {
    auto sparse_coo_tensor =
        std::dynamic_pointer_cast<phi::SparseCooTensor>(tensor_.impl());
    PD_CHECK(sparse_coo_tensor != nullptr,
             "_nnz: failed to cast tensor impl to SparseCooTensor");
    return sparse_coo_tensor->nnz();
  } else {
    auto sparse_csr_tensor =
        std::dynamic_pointer_cast<phi::SparseCsrTensor>(tensor_.impl());
    PD_CHECK(sparse_csr_tensor != nullptr,
             "_nnz: failed to cast tensor impl to SparseCsrTensor");
    return sparse_csr_tensor->nnz();
  }
}

}  // namespace at
