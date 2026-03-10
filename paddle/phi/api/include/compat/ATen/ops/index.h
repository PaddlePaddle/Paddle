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
#include <ATen/indexing.h>

namespace at {

// TODO(wangyanpeng04): modify the api to
// Tensor index(ArrayRef<at::indexing::TensorIndex> indices) const;
inline at::Tensor index(const at::Tensor& self,
                        const std::vector<at::indexing::Slice>& indices) {
  std::vector<int64_t> starts(indices.size());
  std::vector<int64_t> ends(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    starts[i] = indices[i].start();
    ends[i] = indices[i].stop();
  }
  return paddle::experimental::slice(
             self._PD_GetInner(), {0, 1}, starts, ends, {1}, {})
      .contiguous();
}

}  // namespace at

namespace at {

inline at::Tensor Tensor::index(
    const std::vector<at::indexing::Slice>& indices) const {
  return at::index(*this, indices);
}

}  // namespace at
