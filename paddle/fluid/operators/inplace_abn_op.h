//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

USE_PHI_FUNCTOR(LeakyRelu)

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

enum InplaceABNActivationType { identity = 0, leakyrelu = 1, elu = 2 };

inline InplaceABNActivationType GetInplaceABNActivationType(
    const std::string& type) {
  if (type == "leaky_relu") {
    return InplaceABNActivationType::leakyrelu;
  } else if (type == "elu") {
    return InplaceABNActivationType::elu;
  } else if (type == "identity" || type == "") {
    return InplaceABNActivationType::identity;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "unsupported activation type %s for Op(inplace_abn)", type));
  }
}

}  // namespace operators
}  // namespace paddle
