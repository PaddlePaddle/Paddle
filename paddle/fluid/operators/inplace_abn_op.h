//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

enum InplaceABNActivationType { identity = 0, leakyrelu = 1, elu = 2 };

inline InplaceABNActivationType GetInplaceABNActivationType(
    const std::string &type) {
  if (type == "leakyrelu") {
    return InplaceABNActivationType::leakyrelu;
  } else if (type == "elu") {
    return InplaceABNActivationType::elu;
  } else if (type == "identity" || type == "") {
    return InplaceABNActivationType::identity;
  } else {
    // use the identity by default
    return InplaceABNActivationType::identity;
  }
}

template <typename DeviceContext, typename T>
class InplaceABNActivation {
 public:
  template <typename Device, typename X, typename Y>
  void Compute(const int act_type, const Device &d, X x, Y y) {
    if (act_type == InplaceABNActivationType::identity)
      y.device(d) = x;
    else if (act_type == InplaceABNActivationType::leakyrelu)
      LeakyReluFunctor<T>()(d, x, y);
    else if (act_type == InplaceABNActivationType::elu)
      ELUFunctor<T>()(d, x, y);
    else
      // by default to use identify
      y.device(d) = x;
  }

  template <typename Device, typename X, typename Y, typename DX, typename DY>
  void GradCompute(const int act_type, const Device &d, X x, Y y, DX dx,
                   DY dy) {
    if (act_type == InplaceABNActivationType::identity)
      dx.device(d) = dy;
    else if (act_type == InplaceABNActivationType::leakyrelu)
      LeakyReluGradFunctor<T>()(d, x, y, dy, dx);
    else if (act_type == InplaceABNActivationType::elu)
      ELUGradFunctor<T>()(d, x, y, dy, dx);
    else
      dx.device(d) = dy;
  }
};

}  // namespace operators
}  // namespace paddle
