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
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/im2col.h"
#include "paddle/fluid/operators/math/vol2col.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

// Define Op classes in .h file so that other conv transpose
// operator implementations can reuse the code.
class Conv2DTransposeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override;
};

class Conv3DTransposeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override;
};

class ConvTransposeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;
};

class ConvTransposeOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;
};

template <typename DeviceContext, typename T>
class GemmConvTransposeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    // The filter will be reshaped, so it should not be constant pointer
    Tensor filter = *context.Input<Tensor>("Filter");
    Tensor* output = context.Output<Tensor>("Output");

    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    int groups = context.Attr<int>("groups");

    const int batch_size = static_cast<int>(input->dims()[0]);

    // input_shape_vec: {n, c, h, w} or {n, c, d, h, w}
    std::vector<int64_t> input_shape_vec = framework::vectorize(input->dims());
    // filter_shape_vec: {k_o, k_c, k_h, k_w} or {k_o, k_c, k_d, k_h, k_w}
    std::vector<int64_t> filter_shape_vec = framework::vectorize(filter.dims());

    // use col_shape in the im2col and col2im (or vol2col and col2vol)
    // calculation
    // col_shape_vec: {c/g, k_h, k_w, h, w} or {c/g, k_d, k_h, k_w, d, h, w}
    size_t data_dim = filter_shape_vec.size() - 2;
    std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
    col_shape_vec[0] = output->dims()[1] / groups;
    for (size_t j = 0; j < data_dim; ++j) {
      col_shape_vec[j + 1] = filter_shape_vec[j + 2];
      col_shape_vec[j + 1 + data_dim] = input_shape_vec[j + 2];
    }
    DDim col_shape(framework::make_ddim(col_shape_vec));

    // use col_matrix_shape in the gemm calculation
    // size: (c/g * k_h * k_w, h * w) or (c/g * k_d * k_h * k_w, d * h * w)
    DDim col_matrix_shape = framework::flatten_to_2d(col_shape, data_dim + 1);

    Tensor col;
    col.mutable_data<T>(col_shape, context.GetPlace());
    // col_matrix shares the same piece of data with col,
    // but will be reshaped into a two-dimensional matrix shape
    // to call the matrix multiplication interface.
    Tensor col_matrix;
    col_matrix.ShareDataWith(col);
    col_matrix.Resize(col_matrix_shape);

    // output size: (c, o_h, o_w) or (c, o_d, o_h, o_w)
    DDim output_shape =
        framework::slice_ddim(output->dims(), 1, output->dims().size());

    // input matrix size: (m, h * w) or (m, d * h * w)
    DDim input_matrix_shape = {input->dims()[1], col_matrix_shape[1]};

    // filter size: (m, c/g * k_h * k_w) or (m, c/g * k_d * k_h * k_w)
    DDim filter_matrix_shape = {input->dims()[1], col_matrix_shape[0]};
    filter.Resize(filter_matrix_shape);

    output->mutable_data<T>(context.GetPlace());
    math::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    set_zero(dev_ctx, output, static_cast<T>(0));

    int in_step = static_cast<int>(input->dims()[1]) / groups;
    int out_step = static_cast<int>(output->dims()[1]) / groups;
    math::Col2ImFunctor<math::ColFormat::kCFO, DeviceContext, T> col2im;
    math::Col2VolFunctor<DeviceContext, T> col2vol;

    // convolution transpose: gemm + col2im or col2vol (similar to conv-backward
    // on input)
    for (int i = 0; i < batch_size; i++) {
      // batch with size (m, h * w) or (m, d * h * w)
      Tensor input_batch = input->Slice(i, i + 1).Resize(input_matrix_shape);

      // output size: (c, o_h, o_w) or (c, o_d, o_h, o_w)
      Tensor output_batch = output->Slice(i, i + 1).Resize(output_shape);

      for (int g = 0; g < groups; g++) {
        Tensor in_slice = input_batch.Slice(g * in_step, (g + 1) * in_step);
        Tensor filter_slice = filter.Slice(g * in_step, (g + 1) * in_step);
        Tensor out_slice = output_batch.Slice(g * out_step, (g + 1) * out_step);

        // col_matrix = filter_slice * input_slice
        // of shape (c/g * k_h * k_w, h * w)
        // or (c/g * k_d * k_h * k_w, d * h * w)
        blas.MatMul(filter_slice, true, in_slice, false, static_cast<T>(1.0),
                    &col_matrix, static_cast<T>(0.0));

        if (data_dim == 2U) {
          // col2im: col_matrix -> dy
          // from (c/g * k_h * k_w, h * w) to (c/g, o_h, o_w)
          col2im(dev_ctx, col, dilations, strides,
                 std::vector<int>{paddings[0], paddings[1], paddings[0],
                                  paddings[1]},
                 &out_slice);
        } else if (data_dim == 3U) {
          // col2vol: col_matrix -> dy
          // from (c/g * k_d * k_h * k_w, d * h * w) to (c/g, o_d, o_h, o_w)
          col2vol(dev_ctx, col, dilations, strides, paddings, &out_slice);
        }
      }
    }
  }
};

template <typename DeviceContext, typename T>
class GemmConvTransposeGradKernel : public framework::OpKernel<T> {
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

    if ((!input_grad) && (!filter_grad)) return;

    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    int groups = context.Attr<int>("groups");

    const int batch_size = static_cast<int>(input->dims()[0]);

    // input_shape_vec: {n, c, h, w} or {n, c, d, h, w}
    std::vector<int64_t> input_shape_vec = framework::vectorize(input->dims());
    // filter_shape_vec: {k_o, k_c, k_h, k_w} or {k_o, k_c, k_d, k_h, k_w}
    std::vector<int64_t> filter_shape_vec = framework::vectorize(filter.dims());

    // use col_shape in the im2col and col2im (or vol2col and col2vol)
    // calculation
    // col_shape_vec: {c, k_h, k_w, h, w} or {c, k_d, k_h, k_w, d, h, w}
    size_t data_dim = filter_shape_vec.size() - 2;
    std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
    col_shape_vec[0] = output_grad->dims()[1];
    for (size_t j = 0; j < data_dim; ++j) {
      col_shape_vec[j + 1] = filter_shape_vec[j + 2];
      col_shape_vec[j + 1 + data_dim] = input_shape_vec[j + 2];
    }
    DDim col_shape(framework::make_ddim(col_shape_vec));

    // use col_matrix_shape in the gemm calculation
    // size: (c * k_h * k_w, h * w) or (c * k_d * k_h * k_w, d * h * w)
    DDim col_matrix_shape = framework::flatten_to_2d(col_shape, data_dim + 1);

    // output size: (c, o_h, o_w) or (c, o_d, o_h, o_w)
    DDim output_shape = framework::slice_ddim(output_grad->dims(), 1,
                                              output_grad->dims().size());

    // input matrix size: (m, h * w) or (m, d * h * w)
    DDim input_matrix_shape = {input->dims()[1], col_matrix_shape[1]};

    // filter size: (m, c/g * k_h * k_w) or (m, c/g * k_d * k_h * k_w)
    DDim filter_matrix_shape = {input->dims()[1], col_matrix_shape[0] / groups};
    filter.Resize(filter_matrix_shape);
    int in_step = static_cast<int>(input->dims()[1]) / groups;
    int col_step = static_cast<int>(col_matrix_shape[0]) / groups;

    // convolution transpose grad on input:
    // im2col + gemm (similar to conv-forward)
    // input need to compute gradient
    auto& dev_ctx = context.template device_context<DeviceContext>();
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    if (input_grad || filter_grad) {
      Tensor col;
      col.mutable_data<T>(col_shape, context.GetPlace());
      // col_matrix shares the same piece of data with col,
      // but will be reshaped into a two-dimensional matrix shape
      // to call the matrix multiplication interface.
      Tensor col_matrix;
      col_matrix.ShareDataWith(col);
      col_matrix.Resize(col_matrix_shape);

      Tensor filter_grad_;
      math::SetConstant<DeviceContext, T> set_zero;

      math::Im2ColFunctor<math::ColFormat::kCFO, DeviceContext, T> im2col;
      math::Vol2ColFunctor<DeviceContext, T> vol2col;

      if (input_grad) {
        input_grad->mutable_data<T>(context.GetPlace());
      }
      if (filter_grad) {  // filter size (m, c/g, k_h, k_w)
        filter_grad->mutable_data<T>(context.GetPlace());
        set_zero(dev_ctx, filter_grad, static_cast<T>(0));
        filter_grad_ = *filter_grad;
        filter_grad_.Resize(filter_matrix_shape);
      }

      for (int i = 0; i < batch_size; i++) {
        // batch with size (c, o_h * o_w)
        Tensor output_grad_batch =
            output_grad->Slice(i, i + 1).Resize(output_shape);

        if (data_dim == 2U) {
          // im2col: dy -> col matrix
          // from (c, o_h, o_w) to (c * k_h * k_w, h * w)
          im2col(dev_ctx, output_grad_batch, dilations, strides,
                 std::vector<int>{paddings[0], paddings[1], paddings[0],
                                  paddings[1]},
                 &col);
        } else if (data_dim == 3U) {
          // vol2col: dy -> col_matrix
          // from (c, o_d, o_h, o_w) to (c * k_d * k_h * k_w, d * h * w)
          vol2col(dev_ctx, output_grad_batch, dilations, strides, paddings,
                  &col);
        }

        if (input_grad) {
          // batch with size (m, h, w)
          Tensor input_grad_batch =
              input_grad->Slice(i, i + 1).Resize(input_matrix_shape);
          // gemm: dx = filter * dy
          // (m, c * k_h * k_w) * (c * k_h * k_w, h * w) -> (m, h * w)
          // or
          // (m, c * k_d * k_h * k_w) * (c * k_d * k_h * k_w, d * h * w) -> (m,
          // d, h, w)
          for (int g = 0; g < groups; g++) {
            Tensor input_grad_slice =
                input_grad_batch.Slice(g * in_step, (g + 1) * in_step);
            Tensor filter_slice = filter.Slice(g * in_step, (g + 1) * in_step);
            Tensor col_matrix_slice =
                col_matrix.Slice(g * col_step, (g + 1) * col_step);

            blas.MatMul(filter_slice, false, col_matrix_slice, false,
                        static_cast<T>(1.0), &input_grad_slice,
                        static_cast<T>(0.0));
          }
        }
        if (filter_grad) {
          // input batch
          Tensor in_batch = input->Slice(i, i + 1).Resize(input_matrix_shape);
          // gemm: d_filter = x * dy^T
          // (m, c * h * w) * (k_h * k_w, c * h * w) -> (m, k_h * k_w)
          // or
          // (m, d * h * w) * (d * h * w, c * k_d * k_h * k_w) -> (m, c * k_d *
          // k_h * k_w)
          for (int g = 0; g < groups; g++) {
            Tensor in_batch_slice =
                in_batch.Slice(g * in_step, (g + 1) * in_step);
            Tensor filter_grad_slice =
                filter_grad_.Slice(g * in_step, (g + 1) * in_step);
            Tensor col_matrix_slice =
                col_matrix.Slice(g * col_step, (g + 1) * col_step);
            blas.MatMul(in_batch_slice, false, col_matrix_slice, true,
                        static_cast<T>(1.0), &filter_grad_slice,
                        static_cast<T>(1.0));
          }
        }
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle
