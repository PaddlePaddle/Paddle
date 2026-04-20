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

#include <ATen/TensorIndexing.h>
#include <ATen/core/Tensor.h>

namespace at::indexing {

inline TensorIndex::TensorIndex(const at::Tensor& tensor)
    : tensor_(std::make_shared<at::Tensor>(tensor)),
      type_(TensorIndexType::Tensor) {}

inline const at::Tensor& TensorIndex::tensor() const { return *tensor_; }

}  // namespace at::indexing

namespace at {

inline at::Tensor index(const at::Tensor& self,
                        ArrayRef<at::indexing::TensorIndex> indices) {
  if (indices.size() == 0) {
    return self;
  }

  bool has_slice = false;
  bool has_tensor_like = false;
  for (const auto& index : indices) {
    has_slice = has_slice || index.is_slice();
    has_tensor_like = has_tensor_like || index.is_tensor() || index.is_none();
    PD_CHECK(!index.is_ellipsis(), "Ellipsis index is not supported yet.");
    PD_CHECK(!index.is_integer(), "Integer index is not supported yet.");
    PD_CHECK(!index.is_boolean(), "Boolean index is not supported yet.");
  }

  if (has_slice && !has_tensor_like) {
    std::vector<int64_t> axes;
    std::vector<int64_t> starts;
    std::vector<int64_t> ends;
    std::vector<int64_t> strides;
    axes.reserve(indices.size());
    starts.reserve(indices.size());
    ends.reserve(indices.size());
    strides.reserve(indices.size());

    int64_t dim = 0;
    for (const auto& index : indices) {
      const auto& slice = index.slice();
      axes.push_back(dim++);
      starts.push_back(static_cast<int64_t>(slice.start()));
      ends.push_back(static_cast<int64_t>(slice.stop()));
      strides.push_back(static_cast<int64_t>(slice.step()));
    }

    return paddle::experimental::slice(
        self._PD_GetInner(), axes, starts, ends, strides, {});
  }

  PD_CHECK(!has_slice,
           "Mixed slice and tensor/None indexing is not supported yet.");
  c10::List<::std::optional<at::Tensor>> tensor_indices;
  for (const auto& index : indices) {
    if (index.is_none()) {
      tensor_indices.push_back(::std::nullopt);
    } else if (index.is_tensor()) {
      tensor_indices.push_back(index.tensor());
    }
  }
  return self.index(tensor_indices);
}

inline at::Tensor index(
    const at::Tensor& self,
    std::initializer_list<at::indexing::TensorIndex> indices) {
  return at::index(self, ArrayRef<at::indexing::TensorIndex>(indices));
}

}  // namespace at

namespace at {

inline at::Tensor Tensor::index(
    ArrayRef<at::indexing::TensorIndex> indices) const {
  return at::index(*this, indices);
}

}  // namespace at
