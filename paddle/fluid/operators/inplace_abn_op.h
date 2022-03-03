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

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T, int MajorType = Eigen::RowMajor,
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

template <typename DeviceContext, typename T>
class InplaceABNActivation {
 private:
  template <typename Functor>
  void setAttrs(const framework::ExecutionContext& ctx, Functor* functor) {
    auto attrs = functor->GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = ctx.Attr<float>(attr.first);
    }
  }

  template <typename Functor, typename... Args>
  void compute(const framework::ExecutionContext& ctx, Functor* functor,
               Args... args) {
    setAttrs(ctx, functor);
    (*functor)(args...);
  }

 public:
  template <typename Device, typename X, typename Y>
  void Compute(const framework::ExecutionContext& ctx, const int act_type,
               const Device& d, X x, Y y) {
    if (act_type == InplaceABNActivationType::identity) {
      y.device(d) = x;
    } else if (act_type == InplaceABNActivationType::leakyrelu) {
      LeakyReluFunctor<T> functor;
      compute(ctx, &functor, d, x, y);
    } else if (act_type == InplaceABNActivationType::elu) {
      ELUFunctor<T> functor;
      compute(ctx, &functor, d, x, y);
    } else {
      PADDLE_THROW(
          platform::errors::InvalidArgument("unsupported activation type"));
    }
  }

  template <typename Device, typename X, typename Y, typename DX, typename DY>
  void GradCompute(const framework::ExecutionContext& ctx, const int act_type,
                   const Device& d, X x, Y y, DX dx, DY dy) {
    const float alpha = ctx.Attr<float>("alpha");

    if (act_type == InplaceABNActivationType::identity) {
      x.device(d) = y;
      dx.device(d) = dy;
    } else if (act_type == InplaceABNActivationType::leakyrelu) {
      auto temp1 = (y < static_cast<T>(0)).template cast<T>().eval() /
                   static_cast<T>(alpha);
      auto temp2 = (y >= static_cast<T>(0)).template cast<T>().eval();
      x.device(d) = y * (temp1 + temp2).template cast<T>();

      LeakyReluGradFunctor<T> functor;
      compute(ctx, &functor, d, x, y, dy, dx);
    } else if (act_type == InplaceABNActivationType::elu) {
      auto temp1 = (y >= static_cast<T>(0)).template cast<T>().eval();
      auto temp = (y < static_cast<T>(0)).template cast<T>().eval();
      auto temp2 = (y * temp / static_cast<T>(alpha) + static_cast<T>(1)).log();
      x.device(d) = (y * temp1 + temp2).template cast<T>();

      ELUGradNegativeAlphaFunctor<T> functor;
      compute(ctx, &functor, d, x, y, dy, dx);
    } else {
      PADDLE_THROW(
          platform::errors::InvalidArgument("unsupported activation type"));
    }
  }
};

}  // namespace operators
}  // namespace paddle
