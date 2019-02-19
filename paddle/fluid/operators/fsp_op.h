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
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class CPUFSPOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(context.GetPlace()),
                   "It must use CPUPlace.");
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Input<Tensor>("Y");
    auto* output = context.Output<Tensor>("Out");
    const T* x_data = x->data<T>();
    const T* y_data = y->data<T>();
    T* output_data = output->mutable_data<T>(context.GetPlace());
    auto x_dims = x->dims();
    auto y_dims = y->dims();

    const size_t batch_size = x_dims[0];
    const size_t x_channel = x_dims[1];
    const size_t y_channel = y_dims[1];
    const size_t height = x_dims[2];
    const size_t width = x_dims[3];

    for (size_t batch = 0; batch < batch_size; ++batch) {
      for (size_t out_h = 0; out_h < x_channel; ++out_h) {
        for (size_t out_w = 0; out_w < y_channel; ++out_w) {
          const T* x_data_ptr = x_data + batch * (x_channel * height * width) +
                                out_h * (height * width);
          const T* y_data_ptr = y_data + batch * (y_channel * height * width) +
                                out_w * (height * width);
          T sum = 0;
          size_t count = height * width;
          for (size_t i = 0; i < count; ++i) {
            sum += (x_data_ptr[i] * y_data_ptr[i]);
          }
          output_data[batch * x_channel * y_channel + out_h * y_channel +
                      out_w] = sum / count;
        }
      }
    }
  }
};

template <typename T>
class CPUFSPGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(context.GetPlace()),
                   "It must use CPUPlace.");

    auto* d_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* d_y = context.Output<Tensor>(framework::GradVarName("Y"));
    if (d_x == nullptr && d_y == nullptr) {
      return;
    }
    auto* d_out = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* d_output_data = d_out->data<T>();
    auto d_out_dims = d_out->dims();
    const T* x_data = nullptr;
    const T* y_data = nullptr;
    T* d_x_data = nullptr;
    T* d_y_data = nullptr;
    size_t h = 0;
    size_t w = 0;

    math::SetConstant<platform::CPUDeviceContext, T> set_zero;
    if (d_x != nullptr) {
      d_x_data = d_x->mutable_data<T>(context.GetPlace());
      set_zero(context.template device_context<platform::CPUDeviceContext>(),
               d_x, static_cast<T>(0));
      auto* y = context.Input<Tensor>("Y");
      y_data = y->data<T>();
      auto d_x_dims = d_x->dims();
      h = d_x_dims[2];
      w = d_x_dims[3];
    }
    if (d_y != nullptr) {
      d_y_data = d_y->mutable_data<T>(context.GetPlace());
      set_zero(context.template device_context<platform::CPUDeviceContext>(),
               d_y, static_cast<T>(0));
      auto* x = context.Input<Tensor>("X");
      x_data = x->data<T>();
      auto d_y_dims = d_y->dims();
      h = d_y_dims[2];
      w = d_y_dims[3];
    }

    const size_t batch_size = d_out_dims[0];
    const size_t x_channel = d_out_dims[1];
    const size_t y_channel = d_out_dims[2];
    const size_t height = h;
    const size_t width = w;

    for (size_t batch = 0; batch < batch_size; ++batch) {
      for (size_t out_h = 0; out_h < x_channel; ++out_h) {
        for (size_t out_w = 0; out_w < y_channel; ++out_w) {
          int x_offset =
              batch * (x_channel * height * width) + out_h * (height * width);
          int y_offset =
              batch * (y_channel * height * width) + out_w * (height * width);
          T d_output = d_output_data[batch * x_channel * y_channel +
                                     out_h * y_channel + out_w];
          size_t count = height * width;
          for (size_t i = 0; i < count; ++i) {
            if (d_x_data != nullptr) {
              d_x_data[x_offset + i] += d_output * y_data[y_offset + i] / count;
            }
            if (d_y_data != nullptr) {
              d_y_data[y_offset + i] += d_output * x_data[x_offset + i] / count;
            }
          }
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
