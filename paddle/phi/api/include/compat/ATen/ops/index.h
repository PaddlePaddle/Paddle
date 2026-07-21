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
#include <c10/core/List.h>

namespace at::indexing {

inline TensorIndex::TensorIndex(const at::Tensor& tensor)
    : tensor_(std::make_shared<at::Tensor>(tensor)),
      type_(TensorIndexType::Tensor) {}

inline const at::Tensor& TensorIndex::tensor() const { return *tensor_; }

}  // namespace at::indexing

namespace at::detail {

inline bool _PD_is_full_slice(const at::indexing::Slice& slice) {
  return static_cast<int64_t>(slice.start()) == 0 &&
         static_cast<int64_t>(slice.stop()) == at::indexing::INDEX_MAX &&
         static_cast<int64_t>(slice.step()) == 1;
}

inline at::Tensor _PD_apply_tensor_index(
    const at::Tensor& self, ArrayRef<at::indexing::TensorIndex> indices) {
  int64_t output_dim = 0;
  int tensor_index_count = 0;
  at::Tensor result = self;

  for (const auto& index : indices) {
    if (index.is_tensor()) {
      ++tensor_index_count;
      PD_CHECK(tensor_index_count == 1,
               "Multiple tensor indices mixed with None/Slice are not "
               "supported yet.");
      result = paddle::experimental::index_select(
          result._PD_GetInner(), index.tensor()._PD_GetInner(), output_dim);
      ++output_dim;
    } else if (index.is_none()) {
      result =
          paddle::experimental::unsqueeze(result._PD_GetInner(), {output_dim});
      ++output_dim;
    } else if (index.is_slice()) {
      const auto& slice = index.slice();
      PD_CHECK(_PD_is_full_slice(slice),
               "Only full Slice() is supported when mixed with tensor/None "
               "indices.");
      ++output_dim;
    }
  }

  return result;
}

inline at::Tensor _PD_index_tensor_indices(
    const at::Tensor& self, ArrayRef<at::indexing::TensorIndex> indices) {
  if (indices.size() == 0) {
    PD_THROW("index() cannot be called with an empty index list");
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

  if (has_slice) {
    return _PD_apply_tensor_index(self, indices);
  }

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

}  // namespace at::detail

namespace at {

inline at::Tensor index(const at::Tensor& self,
                        const c10::List<::std::optional<at::Tensor>>& indices) {
  if (indices.empty()) {
    return self;
  }

  bool all_none = true;
  for (const auto& idx : indices) {
    if (idx.has_value()) {
      all_none = false;
      break;
    }
  }
  if (all_none) {
    return self;
  }

  std::vector<paddle::Tensor> pd_indices;
  std::vector<bool> has_index(indices.size(), false);
  pd_indices.reserve(indices.size());

  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i].has_value()) {
      pd_indices.push_back(indices[i].value()._PD_GetInner());
      has_index[i] = true;
    } else {
      pd_indices.push_back(paddle::Tensor());
    }
  }

  int non_none_count = 0;
  size_t first_non_none = 0;
  for (size_t i = 0; i < has_index.size(); ++i) {
    if (has_index[i]) {
      non_none_count++;
      first_non_none = i;
    }
  }

  if (non_none_count == 1 && first_non_none == 0) {
    return paddle::experimental::index_select(
        self._PD_GetInner(), pd_indices[0], 0);
  }

  if (non_none_count == static_cast<int>(indices.size())) {
    auto stacked_indices = paddle::experimental::stack(pd_indices, -1);
    return paddle::experimental::gather_nd(self._PD_GetInner(),
                                           stacked_indices);
  }

  auto self_dims = self._PD_GetInner().dims();
  int self_rank = self_dims.size();
  at::Tensor result = self;

  for (size_t i = 0; i < indices.size() && i < static_cast<size_t>(self_rank);
       ++i) {
    if (has_index[i]) {
      paddle::Tensor pd_result = result._PD_GetInner();
      result = paddle::experimental::index_select(
          pd_result, pd_indices[i], static_cast<int>(i));
    }
  }

  return result;
}

}  // namespace at

namespace at {

inline at::Tensor Tensor::index(
    ArrayRef<at::indexing::TensorIndex> indices) const {
  return at::detail::_PD_index_tensor_indices(*this, indices);
}

}  // namespace at
