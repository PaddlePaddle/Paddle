// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/controlflow/compare_op.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using DDim = framework::DDim;

namespace detail {
static DDim GetEigenvalueDim(const DDim& dim, int k) {
  auto vec = framework::vectorize(dim);
  vec.erase(vec.end() - 2, vec.end());
  vec.push_back(k);
  return framework::make_ddim(vec);
}

static DDim NewAxisDim(const DDim& dim, int k) {
  auto vec = framework::vectorize(dim);
  vec.push_back(k);
  return framework::make_ddim(vec);
}

static DDim RemoveLastDim(const DDim& dim) {
  auto vec = framework::vectorize(dim);
  if (vec.size() <= 1) {
    return framework::make_ddim({1});
  }
  vec.erase(vec.end() - 1, vec.end());
  return framework::make_ddim(vec);
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

}  // namespace operators
}  // namespace paddle
