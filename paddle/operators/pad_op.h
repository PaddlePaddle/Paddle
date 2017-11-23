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

template <typename Place, typename T, size_t D>
void PadFunction(const framework::ExecutionContext& context) {
  auto pads = context.Attr<std::vector<int>>("paddings");
  Eigen::array<std::pair<int, int>, D> paddings;
  for (size_t i = 0; i < paddings.size(); ++i) {
    paddings[i].first = pads[i * 2];
    paddings[i].second = pads[i * 2 + 1];
  }
  T pad_value = context.Attr<T>("pad_value");

  auto* x = context.Input<Tensor>("X");
  auto* out = context.Output<Tensor>("Out");
  out->mutable_data<T>(context.GetPlace());

  auto x_tensor = EigenTensor<T, D>::From(*x);
  auto out_tensor = EigenTensor<T, D>::From(*out);
  auto place = context.GetEigenDevice<Place>();
  out_tensor.device(place) = x_tensor.pad(paddings, pad_value);
}

template <typename Place, typename T>
class PadKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    int rank = context.Input<Tensor>("X")->dims().size();
    switch (rank) {
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
        PADDLE_THROW(
            "PadOp only support tensors with no more than 6 dimensions.");
    }
  }
};

template <typename Place, typename T, size_t D>
void PadGradFunction(const framework::ExecutionContext& context) {
  auto pads = context.Attr<std::vector<int>>("paddings");
  Eigen::array<std::pair<int, int>, D> paddings;
  for (size_t i = 0; i < paddings.size(); ++i) {
    paddings[i].first = -pads[i * 2];
    paddings[i].second = -pads[i * 2 + 1];
  }
  auto* d_out = context.Input<Tensor>(framework::GradVarName("Out"));
  auto* d_x = context.Output<Tensor>(framework::GradVarName("X"));
  if (d_x != nullptr) {
    d_x->mutable_data<T>(context.GetPlace());
    auto d_x_tensor = EigenTensor<T, D>::From(*d_x);
    auto d_out_tensor = EigenTensor<T, D>::From(*d_out);
    auto place = context.GetEigenDevice<Place>();
    d_x_tensor.device(place) = d_out_tensor.pad(paddings, 0);
  }
}

template <typename Place, typename T>
class PadGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    size_t rank =
        context.Input<Tensor>(framework::GradVarName("Out"))->dims().size();
    switch (rank) {
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
        PADDLE_THROW(
            "PadOp only support tensors with no more than 6 dimensions.");
    }
  }
};

}  // namespace operators
}  // namespace paddle
