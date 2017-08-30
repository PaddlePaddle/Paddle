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

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;

template <typename Place, typename T>
class PadKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto paddings =
        context.op_.GetAttr<std::vector<std::pair<int, int>>>("paddings");
    T pad_value = context.op_.GetAttr<T>("pad_value");

    auto* X = context.Input<Tensor>("X");
    auto* Out = context.Output<Tensor>("Out");
    Out->mutable_data<T>(context.GetPlace());
    auto dims = X->dims();

    // Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor,
    // Eigen::DenseIndex>> X_tensor = EigenTensor<T, 2>::From(*X);
    // Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>>
    // Out_tensor = EigenTensor<T, 2>::From(*Out);
    EigenTensor<T, dims.size()>::ConstType X_tensor =
        EigenTensor<T, dims.size()>::From(*X);
    EigenTensor<T, dims.size()>::Type Out_tensor =
        EigenTensor<T, dims.size()>::From(*Out);
    Out_tensor = X_tensor.pad(paddings, pad_value);
  }
};

template <typename Place, typename T>
class PadGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    std::vector<std::pair<int, int>> paddings =
        context.op_.GetAttr<std::vector<std::pair<int, int>>>("paddings");
    for (int i = 0; i < paddings.size(); ++i) {
      paddings[0].first = -paddings[0].first;
      paddings[1].second = -paddings[1].second;
    }
    auto* dOut = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto dims = dOut->dims();

    auto* dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    dX->mutable_data<T>(ctx.GetPlace());

    EigenTensor<T, dims.size()>::Type dX_tensor =
        EigenTensor<T, dims.size()>::From(*dX);
    EigenTensor<T, dims.size()>::ConstType dOut_tensor =
        EigenTensor<T, dims.size()>::From(*dOut);
    dX_tensor = dOut_tensor.pad(paddings, 0);
  }
};

}  // namespace operators
}  // namespace paddle
