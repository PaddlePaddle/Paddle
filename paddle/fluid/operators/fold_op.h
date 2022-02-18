/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <memory>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/im2col.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class FoldOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* input = ctx.Input<Tensor>("X");
    const int batch_size = static_cast<int>(input->dims()[0]);
    Tensor* output = ctx.Output<Tensor>("Y");
    output->mutable_data<T>(ctx.GetPlace());

    std::vector<int> output_sizes = ctx.Attr<std::vector<int>>("output_sizes");
    std::vector<int> kernel_sizes = ctx.Attr<std::vector<int>>("kernel_sizes");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");

    math::Col2ImFunctor<math::ColFormat::kCFO, DeviceContext, T> col2im;

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    auto input_dims = input->dims();

    int output_height = (output_sizes[0] + 2 * paddings[0] -
                         (dilations[0] * (kernel_sizes[0] - 1) + 1)) /
                            strides[0] +
                        1;
    int output_width = (output_sizes[1] + 2 * paddings[1] -
                        (dilations[1] * (kernel_sizes[1] - 1) + 1)) /
                           strides[1] +
                       1;

    int n_input_plane = input_dims[1];
    int n_output_plane = n_input_plane / (kernel_sizes[0] * kernel_sizes[1]);

    framework::DDim output_shape(
        {n_output_plane, output_sizes[0], output_sizes[1]});

    framework::DDim input_matrix_shape({input_dims[0], kernel_sizes[0],
                                        kernel_sizes[1], output_height,
                                        output_width});
    pten::funcs::SetConstant<DeviceContext, T> set_zero;
    set_zero(dev_ctx, output, static_cast<T>(0));

    for (int i = 0; i < batch_size; i++) {
      Tensor out_batch =
          output->Slice(i, i + 1).Resize(output_shape);  // im size=3
      Tensor in_batch =
          input->Slice(i, i + 1).Resize(input_matrix_shape);  // col size=5
      col2im(dev_ctx, in_batch, dilations, strides, paddings, &out_batch);
    }
  }
};

template <typename DeviceContext, typename T>
class FoldGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* output_grad = ctx.Input<Tensor>(framework::GradVarName("Y"));
    Tensor* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    input_grad->mutable_data<T>(ctx.GetPlace());

    if ((!output_grad) || (!input_grad)) return;

    std::vector<int> output_sizes = ctx.Attr<std::vector<int>>("output_sizes");
    std::vector<int> kernel_sizes = ctx.Attr<std::vector<int>>("kernel_sizes");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");

    const int batch_size = static_cast<int>(input_grad->dims()[0]);

    auto input_dims = input_grad->dims();

    int output_height = (output_sizes[0] + 2 * paddings[0] -
                         (dilations[0] * (kernel_sizes[0] - 1) + 1)) /
                            strides[0] +
                        1;
    int output_width = (output_sizes[1] + 2 * paddings[1] -
                        (dilations[1] * (kernel_sizes[1] - 1) + 1)) /
                           strides[1] +
                       1;

    int n_input_plane = input_dims[1];
    int n_output_plane = n_input_plane / (kernel_sizes[0] * kernel_sizes[1]);

    framework::DDim output_shape(
        {n_output_plane, output_sizes[0], output_sizes[1]});
    framework::DDim input_matrix_shape({input_dims[0], kernel_sizes[0],
                                        kernel_sizes[1], output_height,
                                        output_width});

    math::Im2ColFunctor<math::ColFormat::kCFO, DeviceContext, T> im2col;
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    for (int i = 0; i < batch_size; i++) {
      Tensor out_grad_batch = output_grad->Slice(i, i + 1).Resize(output_shape);
      Tensor in_grad_batch =
          input_grad->Slice(i, i + 1).Resize(input_matrix_shape);
      im2col(dev_ctx, out_grad_batch, dilations, strides, paddings,
             &in_grad_batch);
    }
  }
};
}  // namespace operators
}  // namespace paddle
