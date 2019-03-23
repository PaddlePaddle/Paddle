/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class FSPOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Input<Tensor>("Y");
    auto* output = context.Output<Tensor>("Out");
    output->mutable_data<T>(context.GetPlace());
    auto x_dims = x->dims();
    auto y_dims = y->dims();

    auto batch_size = x_dims[0];
    auto x_channel = x_dims[1];
    auto y_channel = y_dims[1];
    auto height = x_dims[2];
    auto width = x_dims[3];

    auto blas = math::GetBlas<DeviceContext, T>(context);

    for (int64_t batch = 0; batch < batch_size; ++batch) {
      auto x_mat =
          x->Slice(batch, batch + 1).Resize({x_channel, height * width});
      auto y_mat =
          y->Slice(batch, batch + 1).Resize({y_channel, height * width});
      auto out_mat =
          output->Slice(batch, batch + 1).Resize({x_channel, y_channel});

      blas.MatMul(x_mat, false, y_mat, true,
                  static_cast<T>(1.0 / (height * width)), &out_mat,
                  static_cast<T>(0.0));
    }
  }
};

template <typename DeviceContext, typename T>
class FSPGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* d_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* d_y = context.Output<Tensor>(framework::GradVarName("Y"));
    if (d_x == nullptr && d_y == nullptr) {
      return;
    }
    auto* d_out = context.Input<Tensor>(framework::GradVarName("Out"));
    auto d_out_dims = d_out->dims();
    auto batch_size = d_out_dims[0];
    auto x_channel = d_out_dims[1];
    auto y_channel = d_out_dims[2];
    int64_t h = 0;
    int64_t w = 0;

    auto blas = math::GetBlas<DeviceContext, T>(context);
    math::SetConstant<DeviceContext, T> set_zero;
    if (d_x != nullptr) {
      d_x->mutable_data<T>(context.GetPlace());
      set_zero(context.template device_context<DeviceContext>(), d_x,
               static_cast<T>(0));
      auto* y = context.Input<Tensor>("Y");
      auto y_dims = y->dims();
      h = y_dims[2];
      w = y_dims[3];
      for (int64_t batch = 0; batch < batch_size; ++batch) {
        auto d_x_mat = d_x->Slice(batch, batch + 1).Resize({x_channel, h * w});
        auto y_mat = y->Slice(batch, batch + 1).Resize({y_channel, h * w});
        auto d_out_mat =
            d_out->Slice(batch, batch + 1).Resize({x_channel, y_channel});
        blas.MatMul(d_out_mat, false, y_mat, false,
                    static_cast<T>(1.0 / (h * w)), &d_x_mat,
                    static_cast<T>(0.0));
      }
    }
    if (d_y != nullptr) {
      d_y->mutable_data<T>(context.GetPlace());
      set_zero(context.template device_context<DeviceContext>(), d_y,
               static_cast<T>(0));
      auto* x = context.Input<Tensor>("X");
      auto x_dims = x->dims();
      h = x_dims[2];
      w = x_dims[3];

      for (int64_t batch = 0; batch < batch_size; ++batch) {
        auto d_y_mat = d_y->Slice(batch, batch + 1).Resize({y_channel, h * w});
        auto x_mat = x->Slice(batch, batch + 1).Resize({x_channel, h * w});
        auto d_out_mat =
            d_out->Slice(batch, batch + 1).Resize({x_channel, y_channel});
        blas.MatMul(d_out_mat, true, x_mat, false,
                    static_cast<T>(1.0 / (h * w)), &d_y_mat,
                    static_cast<T>(0.0));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
