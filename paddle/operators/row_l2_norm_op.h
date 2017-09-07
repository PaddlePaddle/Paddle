/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include "paddle/operators/math/math_function.h"

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class RowL2NormKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<Tensor>("X");
    auto* out0 = context.Output<Tensor>("L2_Norm");
    out0->mutable_data<T>(context.GetPlace());
    auto* out1 = context.Output<Tensor>("Out");
    out1->mutable_data<T>(context.GetPlace());

    auto x = EigenMatrix<T>::From(*in);
    auto x_norm = EigenMatrix<T>::From(*out0);
    auto x_normalize = EigenMatrix<T>::From(*out1);
    auto place = context.GetEigenDevice<Place>();

    const float eps = 1E-6f;  // to avoid FPE
    x_norm.device(place) = x.square().sum(Eigen::array<int, 1>({{1}})).sqrt();
    x_normalize.device(place) =
        x / (x_norm + x_norm.constant(eps)).broadcast(x.dimensions());
  }
};

template <typename Place, typename T>
class RowL2NormGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in0 = context.Input<Tensor>("X");
    auto* in1 = context.Input<Tensor>("L2_Norm");
    auto* in2 = context.Input<Tensor>("Out");
    auto* in3 = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* out = context.Output<Tensor>(framework::GradVarName("X"));

    if (out != nullptr) {
      out->mutable_data<T>(context.GetPlace());
      auto x = EigenMatrix<T>::From(*in0);
      auto x_norm = EigenMatrix<T>::From(*in1);
      auto x_normalize = EigenMatrix<T>::From(*in2);
      auto x_normalize_grad = EigenMatrix<T>::From(*in3);
      auto x_grad = EigenMatrix<T>::From(*out);
      auto place = context.GetEigenDevice<Place>();

      const float eps = 1E-6f;  // consistent with forward
      auto x_norm1 = x_norm + x_norm.constant(eps);
      // dX[ij] += dOut[ij] / X_Norm[i]
      // dX[ij] -= X[ij] * sum_{j}{dOut[ij] * Out[ij]} / square(X_Norm[i])
      auto tmp0 = x_normalize_grad / x_norm1.broadcast(x.dimensions());
      auto tmp1 =
          (x_normalize * x_normalize_grad).sum(Eigen::array<int, 1>({{1}})) /
          x_norm1.square();
      auto tmp2 =
          x * tmp1.reshape(x_norm.dimensions()).broadcast(x.dimensions());
      x_grad.device(place) = tmp0 - tmp2;
    }
  }
};

}  // namespace operators
}  // namespace paddle
