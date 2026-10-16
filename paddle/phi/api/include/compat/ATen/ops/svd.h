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
#include <ATen/ops/transpose.h>
#include <ATen/ops/zeros.h>
#include <c10/core/TensorOptions.h>

#include <tuple>
#include <vector>

#include "paddle/phi/api/include/api.h"

namespace at {

// PyTorch: svd(self, bool some=true, bool compute_uv=true) -> (U, S, V)
// Paddle:  svd(x, bool full_matrices=false) -> (U, S, VH)
//
// Mapping:
//   some=true  -> full_matrices=false (partial, k=min(m,n))
//   some=false -> full_matrices=true  (full)
//
// Paddle returns VH. Libtorch's legacy at::svd returns V, so convert VH to V
// with transpose and conjugate for complex tensors.
//
// For compute_uv=false, Paddle always computes U and VH internally.
// We return zero-filled U and V to match PyTorch behavior.
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> svd(
    const at::Tensor& self, bool some = true, bool compute_uv = true) {
  bool full_matrices = !some;

  if (compute_uv) {
    auto [pd_U, pd_S, pd_VH] =
        paddle::experimental::svd(self._PD_GetInner(), full_matrices);

    at::Tensor U(pd_U);
    at::Tensor S(pd_S);
    at::Tensor VH(pd_VH);

    at::Tensor V = VH.transpose(-2, -1);
    if (V.is_complex()) {
      V = at::Tensor(paddle::experimental::conj(V._PD_GetInner()));
    }

    return std::make_tuple(U, S, V);
  } else {
    // compute_uv=false: compute S only, return zero-filled U and V
    // Note: PyTorch ignores 'some' when compute_uv=false,
    // always returning U:(m,m) and V:(n,n).
    // Since U and VH are discarded, use full_matrices=false to reduce
    // unnecessary computation (partial SVD, k=min(m,n)).
    auto [pd_U, pd_S, pd_VH] =
        paddle::experimental::svd(self._PD_GetInner(), /*full_matrices=*/false);

    at::Tensor S(pd_S);

    auto dims = self.sizes();
    int64_t m = dims[dims.size() - 2];
    int64_t n = dims[dims.size() - 1];

    std::vector<int64_t> u_sizes(dims.begin(), dims.end());
    std::vector<int64_t> v_sizes(dims.begin(), dims.end());

    u_sizes[u_sizes.size() - 1] = m;
    v_sizes[v_sizes.size() - 2] = n;
    v_sizes[v_sizes.size() - 1] = n;

    at::Tensor U = at::zeros(u_sizes, self.options());
    at::Tensor V = at::zeros(v_sizes, self.options());

    return std::make_tuple(U, S, V);
  }
}

}  // namespace at

namespace at {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> Tensor::svd(
    bool some, bool compute_uv) const {
  return at::svd(*this, some, compute_uv);
}

}  // namespace at
