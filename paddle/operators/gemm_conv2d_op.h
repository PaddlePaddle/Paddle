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
#include "paddle/operators/math/im2col.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename Place, typename T>
class GemmConv2DKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    // The filter will be reshaped in the calculations,
    // so here use an assignment operation,
    // that avoids modifying the variable in the Scope.
    Tensor filter = *context.Input<Tensor>("Filter");
    Tensor* output = context.Output<Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());

    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    int groups = context.Attr<int>("groups");

    int batch_size = input->dims()[0];
    int input_channels = input->dims()[1];
    int filter_height = filter.dims()[filter.dims().size() - 2];
    int filter_width = filter.dims()[filter.dims().size() - 1];
    int output_channels = output->dims()[1];
    int output_height = output->dims()[2];
    int output_width = output->dims()[3];

    paddle::operators::math::Im2ColFunctor<
        paddle::operators::math::ColFormat::kCFO, Place, T>
        im2col;
    // use col_shape in the im2col calculation
    framework::DDim col_shape = {input_channels / groups, filter_height,
                                 filter_width, output_height, output_width};
    // use col_matrix_shape in the gemm calculation
    framework::DDim col_matrix_shape = {
        input_channels / groups * filter_height * filter_width,
        output_height * output_width};
    Tensor col;
    col.mutable_data<T>(col_shape, context.GetPlace());
    // col_matrix shares the same piece of data with col,
    // but will be reshaped into a two-dimensional matrix shape
    // to call the matrix multiplication interface.
    Tensor col_matrix = col;
    col_matrix.Resize(col_matrix_shape);

    framework::DDim input_shape = {input->dims()[1], input->dims()[2],
                                   input->dims()[3]};
    framework::DDim filter_matrix_shape = {filter.dims()[0],
                                           filter.numel() / filter.dims()[0]};
    filter.Resize(filter_matrix_shape);

    framework::DDim output_matrix_shape = {output_channels,
                                           output_height * output_width};

    // convolution operator: im2col + gemm
    int in_step = input_channels / groups;
    int out_step = output_channels / groups;
    for (int i = 0; i < batch_size; i++) {
      Tensor in_batch = input->Slice<T>(i, i + 1).Resize(input_shape);
      Tensor out_batch = output->Slice<T>(i, i + 1).Resize(output_matrix_shape);
      for (int g = 0; g < groups; g++) {
        // im2col
        Tensor in_slice = in_batch.Slice<T>(g * in_step, (g + 1) * in_step);
        im2col(context.device_context(), in_slice, col, strides[0], strides[1],
               paddings[0], paddings[1]);

        // gemm
        Tensor out_slice = out_batch.Slice<T>(g * out_step, (g + 1) * out_step);
        Tensor filter_slice = filter.Slice<T>(g * out_step, (g + 1) * out_step);
        math::matmul<Place, T>(context.device_context(), filter_slice, false,
                               col_matrix, false, T(1.0), &out_slice, T(0.0));
      }
    }
  }
};

template <typename Place, typename T>
class GemmConvGrad2DKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    const Tensor* output_grad =
        context.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad =
        context.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad =
        context.Output<Tensor>(framework::GradVarName("Filter"));

    // The filter and filter_grad will be reshaped in the calculations,
    // so here use an assignment operation,
    // that avoids modifying the variable in the Scope.
    Tensor filter = *context.Input<Tensor>("Filter");

    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    int groups = context.Attr<int>("groups");

    int batch_size = input->dims()[0];
    int input_channels = input->dims()[1];
    int filter_height = filter.dims()[filter.dims().size() - 2];
    int filter_width = filter.dims()[filter.dims().size() - 1];
    int output_channels = output_grad->dims()[1];
    int output_height = output_grad->dims()[2];
    int output_width = output_grad->dims()[3];

    paddle::operators::math::Col2ImFunctor<
        paddle::operators::math::ColFormat::kCFO, Place, T>
        col2im;
    paddle::operators::math::Im2ColFunctor<
        paddle::operators::math::ColFormat::kCFO, Place, T>
        im2col;
    // use col_shape in the im2col and col2im calculation
    framework::DDim col_shape = {input_channels / groups, filter_height,
                                 filter_width, output_height, output_width};
    // use col_matrix_shape in the gemm calculation
    framework::DDim col_matrix_shape = {
        input_channels / groups * filter_height * filter_width,
        output_height * output_width};
    Tensor col;
    col.mutable_data<T>(col_shape, context.GetPlace());
    // col_matrix shares the same piece of data with col,
    // but will be reshaped into a two-dimensional matrix shape
    // to call the matrix multiplication interface.
    Tensor col_matrix = col;
    col_matrix.Resize(col_matrix_shape);

    framework::DDim input_shape = {input->dims()[1], input->dims()[2],
                                   input->dims()[3]};
    framework::DDim output_matrix_shape = {
        output_grad->dims()[1],
        output_grad->dims()[2] * output_grad->dims()[3]};

    framework::DDim filter_matrix_shape = {filter.dims()[0],
                                           filter.numel() / filter.dims()[0]};
    filter.Resize(filter_matrix_shape);

    // convolution backward input operator:  gemm + col2im
    // convolution backward weight operator: im2col + gemm
    int in_step = input_channels / groups;
    int out_step = output_channels / groups;

    if (input_grad) {
      input_grad->mutable_data<T>(context.GetPlace());
      auto t = framework::EigenVector<T>::Flatten(*input_grad);
      t.device(context.GetEigenDevice<Place>()) = t.constant(static_cast<T>(0));

      for (int i = 0; i < batch_size; i++) {
        Tensor out_grad_batch =
            output_grad->Slice<T>(i, i + 1).Resize(output_matrix_shape);
        Tensor in_grad_batch =
            input_grad->Slice<T>(i, i + 1).Resize(input_shape);
        for (int g = 0; g < groups; g++) {
          // gemm
          Tensor out_grad_slice =
              out_grad_batch.Slice<T>(g * out_step, (g + 1) * out_step);
          Tensor filter_slice =
              filter.Slice<T>(g * out_step, (g + 1) * out_step);
          math::matmul<Place, T>(context.device_context(), filter_slice, true,
                                 out_grad_slice, false, T(1.0), &col_matrix,
                                 T(0.0));

          // col2im
          Tensor in_grad_slice =
              in_grad_batch.Slice<T>(g * in_step, (g + 1) * in_step);
          col2im(context.device_context(), in_grad_slice, col, strides[0],
                 strides[1], paddings[0], paddings[1]);
        }
      }
    }

    if (filter_grad) {
      filter_grad->mutable_data<T>(context.GetPlace());
      Tensor filter_grad_ = *filter_grad;
      filter_grad_.Resize(filter_matrix_shape);
      auto t = framework::EigenVector<T>::Flatten(filter_grad_);
      t.device(context.GetEigenDevice<Place>()) = t.constant(static_cast<T>(0));

      for (int i = 0; i < batch_size; i++) {
        Tensor out_grad_batch =
            output_grad->Slice<T>(i, i + 1).Resize(output_matrix_shape);
        Tensor in_batch = input->Slice<T>(i, i + 1).Resize(input_shape);
        for (int g = 0; g < groups; g++) {
          // im2col
          Tensor out_grad_slice =
              out_grad_batch.Slice<T>(g * out_step, (g + 1) * out_step);
          Tensor in_slice = in_batch.Slice<T>(g * in_step, (g + 1) * in_step);
          im2col(context.device_context(), in_slice, col, strides[0],
                 strides[1], paddings[0], paddings[1]);

          // gemm
          Tensor filter_grad_slice =
              filter_grad_.Slice<T>(g * out_step, (g + 1) * out_step);
          math::matmul<Place, T>(context.device_context(), out_grad_slice,
                                 false, col_matrix, true, T(1.0),
                                 &filter_grad_slice, T(1.0));
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
