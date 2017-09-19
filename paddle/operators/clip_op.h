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

using framework::LoDTensor;

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;

template <typename Place, typename T, size_t D>
void ClipFunction(const framework::ExecutionContext& context) {
  auto max = context.op().Attr<float>("max");
  auto min = context.op().Attr<float>("min");
  auto* x = context.Input<LoDTensor>("X");
  auto* out = context.Output<LoDTensor>("Out");
  out->mutable_data<T>(context.GetPlace());
  auto x_tensor = EigenTensor<T, D>::From(*x);
  auto out_tensor = EigenTensor<T, D>::From(*out);
  auto place = context.GetEigenDevice<Place>();
  out_tensor.device(place) = x_tensor.cwiseMin(max).cwiseMax(min);
}

template <typename Place, typename T>
class ClipKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    int rank = context.Input<LoDTensor>("X")->dims().size();
    switch (rank) {
      case 1:
        ClipFunction<Place, T, 1>(context);
        break;
      case 2:
        ClipFunction<Place, T, 2>(context);
        break;
      case 3:
        ClipFunction<Place, T, 3>(context);
        break;
      case 4:
        ClipFunction<Place, T, 4>(context);
        break;
      case 5:
        ClipFunction<Place, T, 5>(context);
        break;
      case 6:
        ClipFunction<Place, T, 6>(context);
        break;
      default:
        PADDLE_THROW(
            "PadOp only support tensors with no more than 6 dimensions.");
    }
  }
};

template <typename T>
class ClipGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto max = context.op().Attr<float>("max");
    auto min = context.op().Attr<float>("min");
    auto* d_out = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* d_x = context.Output<LoDTensor>(framework::GradVarName("X"));
    if (d_x != nullptr) {
      auto* x = context.Input<LoDTensor>("X");
      auto dims = d_x->dims();
      int64_t count = d_out->numel();
      auto d_x_data = d_x->mutable_data<T>(context.GetPlace());
      auto d_out_data = d_out->data<T>();
      auto x_data = x->data<T>();
      for (int i = 0; i < count; ++i) {
        if (x_data[i] > min && x_data[i] < max) {
          d_x_data[i] = d_out_data[i];
        } else {
          d_x_data[i] = 0;
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
