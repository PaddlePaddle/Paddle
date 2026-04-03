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

namespace at {

inline at::Tensor select(const at::Tensor& self, int64_t dim, int64_t index) {
  // Normalize dim to positive value for error messages
  int64_t orig_dim = dim;
  if (dim < 0) {
    dim += self.dim();
  }
  // Check dim is valid
  if (dim < 0 || dim >= self.dim()) {
    PD_CHECK(false,
             "select(): index ",
             orig_dim,
             " out of range for tensor of size ",
             self.sizes(),
             " at dimension ",
             orig_dim);
  }
  // Handle negative index
  int64_t orig_index = index;
  if (index < 0) {
    index = self.size(dim) + index;
  }
  // Check index is valid
  if (index < 0 || index >= self.size(dim)) {
    PD_CHECK(false,
             "select(): index ",
             orig_index,
             " out of range for tensor of size ",
             self.sizes(),
             " at dimension ",
             orig_dim < 0 ? orig_dim + self.dim() : orig_dim);
  }

  return Tensor(
      paddle::experimental::slice(self._PD_GetInner(),
                                  /*axes=*/{static_cast<int>(dim)},
                                  /*starts=*/{index},
                                  /*ends=*/{index + 1},
                                  /*infer_flags=*/{1},
                                  /*decrease_axis=*/{static_cast<int>(dim)}));
}

inline at::Tensor select_symint(const at::Tensor& self,
                                int64_t dim,
                                c10::SymInt index) {
  return select(self, dim, static_cast<int64_t>(index));
}

}  // namespace at

namespace at {

inline at::Tensor Tensor::select(int64_t dim, int64_t index) const {
  return at::select(*this, dim, index);
}

inline at::Tensor Tensor::select_symint(int64_t dim, c10::SymInt index) const {
  return at::select_symint(*this, dim, index);
}

}  // namespace at
