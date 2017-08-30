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
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename Place, typename T>
class SquaredL2DistanceKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input0 = context.Input<Tensor>("X");
    auto* input1 = context.Input<Tensor>("Y");
    auto* output0 = context.Output<Tensor>("sub_result");
    auto* output1 = context.Output<Tensor>("Out");

    output0->mutable_data<T>(context.GetPlace());
    output1->mutable_data<T>(context.GetPlace());

    auto X = EigenMatrix<T>::From(*input0);
    auto Y = EigenMatrix<T>::From(*input1);
    auto subResult = EigenMatrix<T>::From(*output0);
    auto Z = EigenMatrix<T>::From(*output1);

    auto place = context.GetEigenDevice<Place>();
    // buffer the substraction result
    subResult.device(place) = X - Y;
    const auto& inDims = X.dimensions();
    const auto& subResMat = subResult.reshape(Eigen::array<int, 2>(
        {static_cast<int>(inDims[0]), static_cast<int>(X.size() / inDims[0])}));
    Z.device(place) = subResMat.pow(2).sum(Eigen::array<int, 1>({1}));
  }
};

template <typename Place, typename T>
class SquaredL2DistanceGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input0 = context.Input<Tensor>("sub_result");
    auto* OG = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* IG = context.Output<Tensor>(framework::GradVarName("X"));

    IG->mutable_data<T>(context.GetPlace());

    auto subResult = EigenMatrix<T>::From(*input0);
    auto outGrad = EigenMatrix<T>::From(*OG);
    auto inGrad = EigenMatrix<T>::From(*IG);

    const auto& subResDims = subResult.dimensions();
    int firstDim = static_cast<int>(subResDims[0]);
    int cols = subResult.size() / firstDim;
    const auto subResMat =
        subResult.reshape(Eigen::array<int, 2>({firstDim, cols}));
    // create a matrix view for input gradient tensor
    auto inGradMat = inGrad.reshape(Eigen::array<int, 2>({firstDim, cols}));
    inGradMat.device(context.GetEigenDevice<Place>()) =
        2 * (outGrad.broadcast(Eigen::array<int, 2>({1, cols}))) * subResMat;
  }
};

}  // namespace operators
}  // namespace paddle
