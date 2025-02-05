// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>
#include "paddle/cinn/ir/tensor.h"

namespace cinn::ir::utils {

// FIXME(Aurelius84): Return [Expr(1)] for 0D Tensor as the shape.
static inline std::vector<Expr> GetCompatibleShape(
    const std::vector<Expr>& shape) {
  return shape.empty() ? std::vector<Expr>({Expr(1)}) : shape;
}

// FIXME(Aurelius84): Actually we can't distinguish 0D Tensor from 1D Tensor
// with shape [1]. So name it with Maybe prefix.
static inline bool MaybeZeroRankTensor(const Tensor& tensor) {
  return tensor.ndims() == 1 && tensor->shape[0].is_constant() &&
         static_cast<int>(tensor->shape[0].get_constant()) == 1;
}

// FIXME(Aurelius84): Return [Expr(0)] for 0D Tensor as the indices.
static inline std::vector<Expr> GetCompatibleStoreLoadIndices(
    const Tensor& tensor, const std::vector<Expr>& indices) {
  const bool should_fill_zero = indices.empty() && MaybeZeroRankTensor(tensor);
  return should_fill_zero ? std::vector<Expr>({Expr(0)}) : indices;
}

}  // namespace cinn::ir::utils
