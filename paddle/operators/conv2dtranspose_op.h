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
using DDim = framework::DDim;

// Define Op classes in .h file so that other conv transpose
// operator implementations can reuse the code.
class Conv2DTransposeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  Conv2DTransposeOpMaker(framework::OpProto* proto,
                         framework::OpAttrChecker* op_checker);
};

class Conv2DTransposeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override;
};

class Conv2DTransposeOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override;
};

template <typename Place, typename T>
class GemmConv2DTransposeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    // The filter will be reshaped, so it should not be constant pointer
    Tensor filter = *context.Input<Tensor>("Filter");

    Tensor* output = context.Output<Tensor>("Output");

    std::vector<int> strides = context.Attr<std::vector<int>>("strides");

    // TODO(Zhuoyuan): Paddings can be added in future.
    // groups will alway be disabled in conv2dtranspose.

    const int batch_size = input->dims()[0];
    const int m = input->dims()[1];
    const int h = input->dims()[2];
    const int w = input->dims()[3];

    const int k_h = filter.dims()[2];
    const int k_w = filter.dims()[3];

    const int c = output->dims()[1];  // output channels
    const int o_h = output->dims()[2];
    const int o_w = output->dims()[3];

    paddle::operators::math::Col2ImFunctor<
        paddle::operators::math::ColFormat::kCFO, Place, T>
        col2im;

    // use col_shape in the im2col and col2im calculation
    DDim col_shape = {c, k_h, k_w, h, w};

    // use col_matrix_shape in the gemm calculation
    DDim col_matrix_shape = {c * k_h * k_w, h * w};

    Tensor col;
    col.mutable_data<T>(col_shape, context.GetPlace());
    // col_matrix shares the same piece of data with col,
    // but will be reshaped into a two-dimensional matrix shape
    // to call the matrix multiplication interface.
    Tensor col_matrix;
    col_matrix.ShareDataWith(col);
    col_matrix.Resize(col_matrix_shape);

    DDim output_shape = {c, o_h, o_w};
    DDim input_matrix_shape = {m, h * w};

    DDim filter_matrix_shape = {m, c * k_h * k_w};
    filter.Resize(filter_matrix_shape);

    // convolution transpose: gemm + col2im (similar to conv-backward on input)

    output->mutable_data<T>(context.GetPlace());
    auto t = framework::EigenVector<T>::Flatten(*output);
    t.device(context.GetEigenDevice<Place>()) = t.constant(static_cast<T>(0));

    for (int i = 0; i < batch_size; i++) {
      // batch with size (M, h * w)
      Tensor input_batch = input->Slice(i, i + 1).Resize(input_matrix_shape);
      // filter size: (M, c * k_h * k_w)

      // output size: (c, o_h, o_w)
      Tensor output_batch = output->Slice(i, i + 1).Resize(output_shape);

      // col_matrix = filter * input_batch
      // of shape (c * k_h * k_w, h * w)
      math::matmul<Place, T>(context.device_context(), filter, true,
                             input_batch, false, T(1.0), &col_matrix, T(0.0));
      col2im(context.device_context(), output_batch, col, strides[0],
             strides[1], 0, 0, 0, 0);
    }
  }
};

template <typename Place, typename T>
class GemmConv2DTransposeGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    const Tensor* output_grad =
        context.Input<Tensor>(framework::GradVarName("Output"));

    // For filter, we do not use const pointer b/c we will do reshape,
    // but we should avoid modifying its value.
    Tensor filter = *context.Input<Tensor>("Filter");

    Tensor* input_grad =
        context.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad =
        context.Output<Tensor>(framework::GradVarName("Filter"));

    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    // Actually, no paddings and groups allowed in conv transpose.
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");

    const int batch_size = input->dims()[0];
    const int m = input->dims()[1];
    const int h = input->dims()[2];
    const int w = input->dims()[3];

    const int k_h = filter.dims()[2];
    const int k_w = filter.dims()[3];

    const int c = output_grad->dims()[1];  // output channels
    const int o_h = output_grad->dims()[2];
    const int o_w = output_grad->dims()[3];

    // Only im2col functor required for bp to get to the right shape
    paddle::operators::math::Im2ColFunctor<
        paddle::operators::math::ColFormat::kCFO, Place, T>
        im2col;

    // use col_shape in the im2col and col2im calculation
    DDim col_shape = {c, k_h, k_w, h, w};

    // use col_matrix_shape in the gemm calculation
    DDim col_matrix_shape_f = {c * h * w, k_h * k_w};

    Tensor col;
    col.mutable_data<T>(col_shape, context.GetPlace());
    // col_matrix shares the same piece of data with col,
    // but will be reshaped into a two-dimensional matrix shape
    // to call the matrix multiplication interface.

    DDim output_shape = {c, o_h, o_w};
    DDim input_matrix_shape = {m, h * w};

    DDim filter_matrix_shape = {m, c * k_h * k_w};
    filter.Resize(filter_matrix_shape);

    // convolution transpose grad on input:
    // im2col + gemm (similar to conv-forward)
    // input need to compute gradient
    if (input_grad) {
      Tensor col_matrix;
      col_matrix.ShareDataWith(col);
      DDim col_matrix_shape = {c * k_h * k_w, h * w};
      col_matrix.Resize(col_matrix_shape);

      input_grad->mutable_data<T>(context.GetPlace());
      auto t = framework::EigenVector<T>::Flatten(*input_grad);
      t.device(context.GetEigenDevice<Place>()) = t.constant(static_cast<T>(0));

      for (int i = 0; i < batch_size; i++) {
        // batch with size (c, o_h * o_w)
        Tensor output_grad_batch =
            output_grad->Slice(i, i + 1).Resize(output_shape);
        // filter of size (m, c * k_h * k_w)

        // batch with size (m, h, w)
        Tensor input_grad_batch =
            input_grad->Slice(i, i + 1).Resize(input_matrix_shape);

        // im2col: dy from (c, o_h, o_w) -> (c * k_h * k_w, h * w)
        im2col(context.device_context(), output_grad_batch, col, strides[0],
               strides[1], paddings[0], paddings[0], paddings[1], paddings[1]);

        // gemm: dx = filter * dy
        // (m, c * k_h * k_w) * (c * k_h * k_w, h * w) -> (m, c, h)
        math::matmul<Place, T>(context.device_context(), filter, false,
                               col_matrix, false, T(1.0), &input_grad_batch,
                               T(0.0));
      }
    }

    // filter gradient required
    if (filter_grad) {
      Tensor col_matrix_f;
      col_matrix_f.ShareDataWith(col);
      DDim col_matrix_shape_f = {c * h * w, k_h * k_w};
      col_matrix_f.Resize(col_matrix_shape_f);

      filter_grad->mutable_data<T>(context.GetPlace());
      Tensor filter_grad_ = *filter_grad;
      filter_grad_.Resize(filter_matrix_shape);
      auto t = framework::EigenVector<T>::Flatten(filter_grad_);
      t.device(context.GetEigenDevice<Place>()) = t.constant(static_cast<T>(0));

      for (int i = 0; i < batch_size; ++i) {
        // batch with size (c, o_h, o_w)
        Tensor output_grad_batch =
            output_grad->Slice(i, i + 1).Resize(output_shape);
        // input batch
        Tensor in_batch = input->Slice(i, i + 1).Resize(input_matrix_shape);

        // im2col: (c * h * w, k_h * k_w)
        im2col(context.device_context(), output_grad_batch, col, strides[0],
               strides[1], paddings[0], paddings[0], paddings[1], paddings[1]);

        // gemm: d_filter = x * y_grad^T
        // (m, c * h * w) * (k_h * k_w, c * h * w) -> (m, c, h)
        math::matmul<Place, T>(context.device_context(), in_batch, false,
                               col_matrix_f, true, T(1.0), &filter_grad_,
                               T(1.0));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
