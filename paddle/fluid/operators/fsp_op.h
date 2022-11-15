/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class FSPOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<phi::DenseTensor>("X");
    auto* y = context.Input<phi::DenseTensor>("Y");
    auto* output = context.Output<phi::DenseTensor>("Out");
    output->mutable_data<T>(context.GetPlace());
    auto x_dims = x->dims();
    auto y_dims = y->dims();

    auto batch_size = x_dims[0];
    auto x_channel = x_dims[1];
    auto y_channel = y_dims[1];
    auto height = x_dims[2];
    auto width = x_dims[3];

    auto blas = phi::funcs::GetBlas<DeviceContext, T>(context);

    phi::funcs::MatDescriptor x_mat_desc;
    x_mat_desc.height_ = x_channel;
    x_mat_desc.width_ = height * width;
    x_mat_desc.batch_size_ = batch_size;
    x_mat_desc.stride_ = x_channel * height * width;
    x_mat_desc.trans_ = false;

    phi::funcs::MatDescriptor y_mat_desc;
    y_mat_desc.height_ = height * width;
    y_mat_desc.width_ = y_channel;
    y_mat_desc.batch_size_ = batch_size;
    y_mat_desc.stride_ = y_channel * height * width;
    y_mat_desc.trans_ = true;

    blas.MatMul(*x,
                x_mat_desc,
                *y,
                y_mat_desc,
                static_cast<T>(1.0 / (height * width)),
                output,
                static_cast<T>(0.0));
  }
};

template <typename DeviceContext, typename T>
class FSPGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* d_x = context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* d_y = context.Output<phi::DenseTensor>(framework::GradVarName("Y"));
    if (d_x == nullptr && d_y == nullptr) {
      return;
    }
    auto* d_out =
        context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto d_out_dims = d_out->dims();
    auto batch_size = d_out_dims[0];
    auto x_channel = d_out_dims[1];
    auto y_channel = d_out_dims[2];
    int64_t h = 0;
    int64_t w = 0;

    auto blas = phi::funcs::GetBlas<DeviceContext, T>(context);
    phi::funcs::SetConstant<DeviceContext, T> set_zero;
    if (d_x != nullptr) {
      d_x->mutable_data<T>(context.GetPlace());
      set_zero(context.template device_context<DeviceContext>(),
               d_x,
               static_cast<T>(0));
      auto* y = context.Input<phi::DenseTensor>("Y");
      auto y_dims = y->dims();
      h = y_dims[2];
      w = y_dims[3];

      phi::funcs::MatDescriptor d_out_mat_desc;
      d_out_mat_desc.height_ = x_channel;
      d_out_mat_desc.width_ = y_channel;
      d_out_mat_desc.batch_size_ = batch_size;
      d_out_mat_desc.stride_ = x_channel * y_channel;
      d_out_mat_desc.trans_ = false;

      phi::funcs::MatDescriptor y_mat_desc;
      y_mat_desc.height_ = y_channel;
      y_mat_desc.width_ = h * w;
      y_mat_desc.batch_size_ = batch_size;
      y_mat_desc.stride_ = y_channel * h * w;
      y_mat_desc.trans_ = false;

      blas.MatMul(*d_out,
                  d_out_mat_desc,
                  *y,
                  y_mat_desc,
                  static_cast<T>(1.0 / (h * w)),
                  d_x,
                  static_cast<T>(0.0));
    }

    if (d_y != nullptr) {
      d_y->mutable_data<T>(context.GetPlace());
      set_zero(context.template device_context<DeviceContext>(),
               d_y,
               static_cast<T>(0));
      auto* x = context.Input<phi::DenseTensor>("X");
      auto x_dims = x->dims();
      h = x_dims[2];
      w = x_dims[3];

      phi::funcs::MatDescriptor d_out_mat_desc;
      d_out_mat_desc.height_ = y_channel;
      d_out_mat_desc.width_ = x_channel;
      d_out_mat_desc.batch_size_ = batch_size;
      d_out_mat_desc.stride_ = x_channel * y_channel;
      d_out_mat_desc.trans_ = true;

      phi::funcs::MatDescriptor x_mat_desc;
      x_mat_desc.height_ = x_channel;
      x_mat_desc.width_ = h * w;
      x_mat_desc.batch_size_ = batch_size;
      x_mat_desc.stride_ = x_channel * h * w;
      x_mat_desc.trans_ = false;

      blas.MatMul(*d_out,
                  d_out_mat_desc,
                  *x,
                  x_mat_desc,
                  static_cast<T>(1.0 / (h * w)),
                  d_y,
                  static_cast<T>(0.0));
    }
  }
};

}  // namespace operators
}  // namespace paddle
