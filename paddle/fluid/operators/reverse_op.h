// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T, int Rank>
struct ReverseFunctor {
  void operator()(const DeviceContext& context, const framework::LoDTensor& in,
                  framework::LoDTensor* out, const std::vector<int>& axis) {
    Eigen::DSizes<bool, Rank> reverse_axis;
    for (int i = 0; i < Rank; ++i) {
      reverse_axis[i] = false;
    }
    for (int a : axis) {
      if (a >= 0) {
        reverse_axis[a] = true;
      } else {
        reverse_axis[Rank + a] = true;
      }
    }

    auto in_eigen = framework::EigenTensor<T, Rank>::From(in);
    auto out_eigen = framework::EigenTensor<T, Rank>::From(*out);
    auto& dev = *context.eigen_device();

    EigenReverse<std::decay_t<decltype(dev)>, T, Rank>::Eval(
        dev, out_eigen, in_eigen, reverse_axis);
  }
};

template <typename DeviceContext, typename T>
class ReverseKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x_var = context.InputVar("X");
    const auto& axis = context.Attr<std::vector<int>>("axis");
    if (x_var->IsType<framework::LoDTensorArray>()) {
      auto& x_array = x_var->Get<framework::LoDTensorArray>();
      auto* out_array = context.Output<framework::LoDTensorArray>("Out");

      out_array->resize(x_array.size());
      for (size_t offset = 0; offset < x_array.size(); offset++) {
        auto& x_tensor = x_array.at(offset);
        PADDLE_ENFORCE_GT(
            x_tensor.memory_size(), 0,
            platform::errors::PreconditionNotMet(
                "The input LoDTensorArray X[%d] holds no memory.", offset));
        auto out_offset = x_array.size() - offset - 1;
        auto* out_tensor = &out_array->at(out_offset);

        out_tensor->set_lod(x_tensor.lod());
        paddle::framework::TensorCopy(x_tensor, context.GetPlace(), out_tensor);
      }
      return;
    }
    auto* x = context.Input<framework::LoDTensor>("X");
    auto* out = context.Output<framework::LoDTensor>("Out");
    out->mutable_data<T>(context.GetPlace());
    int rank = x->dims().size();
    auto& dev_ctx = context.template device_context<DeviceContext>();

    switch (rank) {
      case 1:
        ReverseFunctor<DeviceContext, T, 1> functor1;
        functor1(dev_ctx, *x, out, axis);
        break;
      case 2:
        ReverseFunctor<DeviceContext, T, 2> functor2;
        functor2(dev_ctx, *x, out, axis);
        break;
      case 3:
        ReverseFunctor<DeviceContext, T, 3> functor3;
        functor3(dev_ctx, *x, out, axis);
        break;
      case 4:
        ReverseFunctor<DeviceContext, T, 4> functor4;
        functor4(dev_ctx, *x, out, axis);
        break;
      case 5:
        ReverseFunctor<DeviceContext, T, 5> functor5;
        functor5(dev_ctx, *x, out, axis);
        break;
      case 6:
        ReverseFunctor<DeviceContext, T, 6> functor6;
        functor6(dev_ctx, *x, out, axis);
        break;
      default:
        PADDLE_THROW(paddle::platform::errors::OutOfRange(
            "The reserve operator does not support input tensors"
            "whose ranks are greater than 6."));
    }
  }
};

}  // namespace operators
}  // namespace paddle
