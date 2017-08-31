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
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class SquaredL2DistanceKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input0 = context.Input<Tensor>("X");
    const int rank = framework::arity(input0->dims());
    switch (rank) {
      case 2:
        Operate<2>(context);
        break;
      case 3:
        Operate<3>(context);
        break;
      case 4:
        Operate<4>(context);
        break;
      case 5:
        Operate<5>(context);
        break;
      case 6:
        Operate<6>(context);
        break;
      default:
        // already asserted in SquaredL2DistanceOpMaker
        break;
    }
  }

 private:
  template <int Dims>
  void Operate(const framework::ExecutionContext& context) const {
    auto* input0 = context.Input<Tensor>("X");
    auto* input1 = context.Input<Tensor>("Y");
    auto* output0 = context.Output<Tensor>("sub_result");
    auto* output1 = context.Output<Tensor>("Out");

    output0->mutable_data<T>(context.GetPlace());
    output1->mutable_data<T>(context.GetPlace());

    auto X = EigenTensor<T, Dims>::From(*input0);
    auto Y = EigenTensor<T, Dims>::From(*input1);
    auto subResult = EigenTensor<T, Dims>::From(*output0);
    auto Z = EigenMatrix<T>::From(*output1);

    auto xDims = X.dimensions();
    auto yDims = Y.dimensions();

    auto place = context.GetEigenDevice<Place>();

    // buffer the substraction result
    if (yDims[0] == 1 && xDims[0] != yDims[0]) {
      auto yBroadcastDims = yDims;
      yBroadcastDims[0] = xDims[0];
      subResult.device(place) = X - Y.broadcast(yBroadcastDims);
    } else {
      subResult.device(place) = X - Y;
    }

    // create matrix view for substraction result
    const auto& subResMat = subResult.reshape(Eigen::array<int, 2>(
        {static_cast<int>(xDims[0]), static_cast<int>(X.size() / xDims[0])}));
    Z.device(place) = subResMat.pow(2).sum(Eigen::array<int, 1>({1}));
  }
};

template <typename Place, typename T>
class SquaredL2DistanceGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input0 = context.Input<Tensor>("sub_result");
    const int rank = framework::arity(input0->dims());
    switch (rank) {
      case 2:
        Operate<2>(context);
        break;
      case 3:
        Operate<3>(context);
        break;
      case 4:
        Operate<4>(context);
        break;
      case 5:
        Operate<5>(context);
        break;
      case 6:
        Operate<6>(context);
        break;
      default:
        // already asserted in SquaredL2DistanceOpMaker
        break;
    }
  }

 private:
  template <int Dims>
  void Operate(const framework::ExecutionContext& context) const {
    auto* input0 = context.Input<Tensor>("sub_result");
    auto* OG = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* XG = context.Output<Tensor>(framework::GradVarName("X"));
    auto* YG = context.Output<Tensor>(framework::GradVarName("Y"));

    auto subResult = EigenTensor<T, Dims>::From(*input0);
    auto outGrad = EigenMatrix<T>::From(*OG);

    auto subResDims = subResult.dimensions();
    int firstDim = static_cast<int>(subResDims[0]);
    int cols = subResult.size() / firstDim;
    const auto subResMat =
        subResult.reshape(Eigen::array<int, 2>({firstDim, cols}));

    // calculate gradient
    auto gradMat =
        2 * (outGrad.broadcast(Eigen::array<int, 2>({1, cols}))) * subResMat;

    // propagate back to input
    auto eigenPlace = context.GetEigenDevice<Place>();
    if (XG != nullptr) {
      XG->mutable_data<T>(context.GetPlace());
      auto xGrad = EigenTensor<T, Dims>::From(*XG);
      // dimensions are same with subResult
      auto xGradMat = xGrad.reshape(Eigen::array<int, 2>({firstDim, cols}));
      xGradMat.device(eigenPlace) = gradMat;
    }
    if (YG != nullptr) {
      YG->mutable_data<T>(context.GetPlace());
      auto yGrad = EigenTensor<T, Dims>::From(*YG);
      auto dimsYGrad = yGrad.dimensions();
      auto yGradMat = yGrad.reshape(Eigen::array<int, 2>(
          {static_cast<int>(dimsYGrad[0]),
           static_cast<int>(yGrad.size() / dimsYGrad[0])}));

      PADDLE_ENFORCE(dimsYGrad[0] <= firstDim,
                     "First dimension of gradient must be greater or "
                     "equal than first dimension of target");

      if (dimsYGrad[0] == firstDim) {
        yGradMat.device(eigenPlace) = -1 * gradMat;
      } else {
        yGradMat.device(eigenPlace) =
            -1 * (gradMat.sum(Eigen::array<int, 2>({0})));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
