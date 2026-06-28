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
#include <ATen/ops/expand.h>

#include <vector>

namespace at {

// broadcast_to - broadcasts tensor to new size
// In PyTorch, broadcast_to is equivalent to expand without the implicit flag.
inline Tensor broadcast_to(const Tensor& self, at::IntArrayRef size) {
  return self.expand(size);
}

// broadcast_to_symint - SymInt overload for broadcast_to
//
// NOTE: This implementation delegates to the IntArrayRef overload above.
// Each SymInt is unwrapped via guard_int(), which works for concrete
// SymInt values but will throw if the SymInt contains an unbacked
// symbolic expression. True symbolic support (preserving SymInt through
// the entire expand path) requires implementing expand_symint in
// expand.h first, which in turn needs paddle::experimental::expand to
// accept symbolic shapes.
inline Tensor broadcast_to_symint(const Tensor& self,
                                  c10::SymIntArrayRef size) {
  std::vector<int64_t> size_vec;
  size_vec.reserve(size.size());
  for (const auto& dim : size) {
    size_vec.push_back(dim.guard_int(__FILE__, __LINE__));
  }
  return at::broadcast_to(self, at::IntArrayRef(size_vec));
}

// Member function: Tensor::broadcast_to
inline Tensor Tensor::broadcast_to(at::IntArrayRef size) const {
  return at::broadcast_to(*this, size);
}

inline Tensor Tensor::broadcast_to_symint(c10::SymIntArrayRef size) const {
  return at::broadcast_to_symint(*this, size);
}

}  // namespace at
