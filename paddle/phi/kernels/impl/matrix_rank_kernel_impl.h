// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/matrix_rank_kernel.h"

namespace phi {

namespace detail {
static DDim GetEigenvalueDim(const DDim& dim, int k) {
  auto vec = common::vectorize(dim);
  vec.erase(vec.end() - 2, vec.end());
  vec.push_back(k);
  return common::make_ddim(vec);
}

static DDim NewAxisDim(const DDim& dim, int k) {
  auto vec = common::vectorize(dim);
  vec.push_back(k);
  return common::make_ddim(vec);
}

static DDim RemoveLastDim(const DDim& dim) {
  auto vec = common::vectorize(dim);
  if (vec.size() <= 1) {
    return common::make_ddim({1});
  }
  vec.erase(vec.end() - 1, vec.end());
  return common::make_ddim(vec);
}

static DDim GetUDDim(const DDim& x_dim, int k) {
  auto x_vec = common::vectorize(x_dim);
  x_vec[x_vec.size() - 1] = k;
  return common::make_ddim(x_vec);
}

static DDim GetVHDDim(const DDim& x_dim, int k) {
  auto x_vec = common::vectorize(x_dim);
  x_vec[x_vec.size() - 2] = k;
  return common::make_ddim(x_vec);
}
}  // namespace detail

template <typename T>
struct GreaterElementFunctor {
  HOSTDEVICE T operator()(const T a, const T b) const {
    if (a > b) {
      return a;
    } else {
      return b;
    }
  }
};

}  // namespace phi
