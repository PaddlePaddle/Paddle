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

template <typename Place, typename T, size_t D>
void PadFunction(const framework::ExecutionContext& context) {
  auto pads = context.op_.GetAttr<std::vector<std::pair<int, int>>>("paddings");
  Eigen::array<std::pair<int, int>, D> paddings;
  for (int i = 0; i < pads.size(); ++i) {
    paddings[i] = pads[i];
  }
  T pad_value = context.op_.GetAttr<T>("pad_value");

  auto* X = context.Input<Tensor>("X");
  auto* Out = context.Output<Tensor>("Out");
  Out->mutable_data<T>(context.GetPlace());
  auto dims = X->dims();

  auto X_tensor = EigenTensor<T, D>::From(*X);
  auto Out_tensor = EigenTensor<T, D>::From(*Out);
  auto place = context.GetEigenDevice<Place>();
  Out_tensor.device(place) = X_tensor.pad(paddings, pad_value);
}

template <typename Place, typename T>
class PadKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    int dim = context.Input<Tensor>("X")->dims().size();
    switch (dim) {
      case 1:
        PadFunction<Place, T, 1>(context);
        break;
      case 2:
        PadFunction<Place, T, 2>(context);
        break;
      case 3:
        PadFunction<Place, T, 3>(context);
        break;
      case 4:
        PadFunction<Place, T, 4>(context);
        break;
      case 5:
        PadFunction<Place, T, 5>(context);
        break;
      case 6:
        PadFunction<Place, T, 6>(context);
        break;
      default:
        LOG(ERROR) << "Only ranks up to 6 supported.";
    }
  }
};

template <typename Place, typename T, size_t D>
void PadGradFunction(const framework::ExecutionContext& context) {
  auto pads = context.op_.GetAttr<std::vector<std::pair<int, int>>>("paddings");
  Eigen::array<std::pair<int, int>, D> paddings;
  for (int i = 0; i < pads.size(); ++i) {
    paddings[0].first = -paddings[0].first;
    paddings[1].second = -paddings[1].second;
  }
  auto* dOut = context.Input<Tensor>(framework::GradVarName("Out"));
  auto* dX = context.Output<Tensor>(framework::GradVarName("X"));
  dX->mutable_data<T>(context.GetPlace());

  auto dX_tensor = EigenTensor<T, D>::From(*dX);
  auto dOut_tensor = EigenTensor<T, D>::From(*dOut);
  auto place = context.GetEigenDevice<Place>();
  dX_tensor.device(place) = dOut_tensor.pad(paddings, 0);
}

template <typename Place, typename T>
class PadGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    size_t dim =
        context.Input<Tensor>(framework::GradVarName("Out"))->dims().size();
    switch (dim) {
      case 1:
        PadGradFunction<Place, T, 1>(context);
        break;
      case 2:
        PadGradFunction<Place, T, 2>(context);
        break;
      case 3:
        PadGradFunction<Place, T, 3>(context);
        break;
      case 4:
        PadGradFunction<Place, T, 4>(context);
        break;
      case 5:
        PadGradFunction<Place, T, 5>(context);
        break;
      case 6:
        PadGradFunction<Place, T, 6>(context);
        break;
      default:
        LOG(ERROR) << "Only ranks up to 6 supported.";
    }
  }
};

}  // namespace operators
}  // namespace paddle
