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

#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/im2col.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename Place, typename T>
class GemmConvKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    Tensor* filter = const_cast<Tensor*>(context.Input<Tensor>("Filter"));
    Tensor* output = context.Output<Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());
    paddle::framework::Tensor col;
    paddle::framework::Tensor in_slice;
    paddle::framework::Tensor out_slice;

    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");

    int batch_size = input->dims()[0];
    int input_channels = input->dims()[1];
    int filter_height = filter->dims()[filter->dims().size() - 2];
    int filter_width = filter->dims()[filter->dims().size() - 1];
    int output_height = output->dims()[2];
    int output_width = output->dims()[3];

    paddle::operators::math::Im2ColFunctor<
        paddle::operators::math::ColFormat::kCFO, Place, T>
        im2col;
    framework::DDim col_shape = {input_channels, filter_height, filter_width,
                                 output_height, output_width};
    col.mutable_data<float>(col_shape, context.GetPlace());

    auto* device_context =
        const_cast<platform::DeviceContext*>(context.device_context_);

    framework::DDim input_shape = {input->dims()[1], input->dims()[2],
                                   input->dims()[3]};
    framework::DDim filter_matrix_shape = {
        filter->dims()[0],
        filter->dims()[1] * filter->dims()[2] * filter->dims()[3]};
    framework::DDim col_matrix_shape = {
        input_channels * filter_height * filter_width,
        output_height * output_width};
    framework::DDim output_matrix_shape = {
        output->dims()[1], output->dims()[2] * output->dims()[3]};
    filter->Resize(filter_matrix_shape);

    // convolution opperator: im2col + gemm
    for (int i = 0; i < batch_size; i++) {
      // im2col
      in_slice = input->Slice<T>(i, i + 1);
      in_slice.Resize(input_shape);
      col.Resize(col_shape);
      im2col(in_slice, col, strides[0], strides[1], paddings[0], paddings[1],
             device_context);

      // gemm
      out_slice = output->Slice<T>(i, i + 1);
      out_slice.Resize(output_matrix_shape);
      col.Resize(col_matrix_shape);
      math::matmul<Place, T>(*filter, false, col, false, T(1.0), &out_slice,
                             T(0.0), device_context);
    }
  }
};

template <typename Place, typename T>
class GemmConvGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
#if 0
    auto input = context.Input<Tensor>("Input");
    auto filter = context.Input<Tensor>("Filter");
    auto output = context.Output<Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());
#endif
  }
};

}  // namespace operators
}  // namespace paddle
