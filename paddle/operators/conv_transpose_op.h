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
#include "paddle/operators/math/vol2col.h"

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

class Conv3DTransposeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  Conv3DTransposeOpMaker(framework::OpProto* proto,
                         framework::OpAttrChecker* op_checker);
};

class ConvTransposeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override;
};

class ConvTransposeOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override;
};

template <typename Place, typename T>
class GemmConvTransposeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    // The filter will be reshaped, so it should not be constant pointer
    Tensor filter = *context.Input<Tensor>("Filter");
    Tensor* output = context.Output<Tensor>("Output");

    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    // TODO(Zhuoyuan): Paddings can be added in future.
    // groups will alway be disabled in conv2dtranspose.

    const int batch_size = static_cast<int>(input->dims()[0]);

    // input_shape_vec: {h, w} or {d, h, w}
    std::vector<int64_t> input_shape_vec = framework::vectorize(input->dims());
    input_shape_vec.erase(input_shape_vec.begin(), input_shape_vec.begin() + 2);

    // filter_shape_vec: {k_h, k_w} or {k_d, k_h, k_w}
    std::vector<int64_t> filter_shape_vec = framework::vectorize(filter.dims());
    filter_shape_vec.erase(filter_shape_vec.begin(),
                           filter_shape_vec.begin() + 2);

    // use col_shape in the im2col and col2im (or vol2col and col2vol)
    // calculation
    // col_shape_vec: {c, k_h, k_w, h, w} or {c, k_d, k_h, k_w, d, h, w}
    std::vector<int64_t> col_shape_vec;
    col_shape_vec.push_back(output->dims()[1]);
    col_shape_vec.insert(col_shape_vec.end(), filter_shape_vec.begin(),
                         filter_shape_vec.end());
    col_shape_vec.insert(col_shape_vec.end(), input_shape_vec.begin(),
                         input_shape_vec.end());
    DDim col_shape(framework::make_ddim(col_shape_vec));

    // use col_matrix_shape in the gemm calculation
    // size: (c * k_h * k_w, h * w) or (c * k_d * k_h * k_w, d * h * w)
    DDim col_matrix_shape =
        framework::flatten_to_2d(col_shape, filter_shape_vec.size() + 1);

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

    // filter size: (m, c * k_h * k_w) or (m, c * k_d * k_h * k_w)
    DDim filter_matrix_shape = {input->dims()[1], col_matrix_shape[0]};
    filter.Resize(filter_matrix_shape);

    output->mutable_data<T>(context.GetPlace());
    math::SetConstant<Place, T> set_zero;
    set_zero(context.device_context(), output, static_cast<T>(0));

    // convolution transpose: gemm + col2im or col2vol (similar to conv-backward
    // on input)
    for (int i = 0; i < batch_size; i++) {
      // batch with size (m, h * w) or (m, d * h * w)
      Tensor input_batch = input->Slice(i, i + 1).Resize(input_matrix_shape);

      // output size: (c, o_h, o_w) or (c, o_d, o_h, o_w)
      Tensor output_batch = output->Slice(i, i + 1).Resize(output_shape);

      // col_matrix = filter * input_batch
      // of shape (c * k_h * k_w, h * w) or (c * k_d * k_h * k_w, d * h * w)
      math::matmul<Place, T>(context.device_context(), filter, true,
                             input_batch, false, static_cast<T>(1.0),
                             &col_matrix, static_cast<T>(0.0));

      if (filter_shape_vec.size() == 2) {
        // col2im: col_matrix -> dy
        // from (c * k_h * k_w, h * w) to (c, o_h, o_w)
        math::Col2ImFunctor<math::ColFormat::kCFO, Place, T> col2im;

        col2im(context.device_context(), output_batch, col, strides[0],
               strides[1], 0, 0, 0, 0);
      } else if (filter_shape_vec.size() == 3) {
        // col2vol: col_matrix -> dy
        // from (c * k_d * k_h * k_w, d * h * w) to (c, o_d, o_h, o_w)
        math::Col2VolFunctor<Place, T> col2vol;
        col2vol(context.device_context(), output_batch, col, strides[0],
                strides[1], strides[2], 0, 0, 0);
      }
    }
  }
};

template <typename Place, typename T>
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
    // Actually, no paddings and groups allowed in conv transpose.
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");

    const int batch_size = static_cast<int>(input->dims()[0]);

    // input_shape_vec: {h, w} or {d, h, w}
    std::vector<int64_t> input_shape_vec = framework::vectorize(input->dims());
    input_shape_vec.erase(input_shape_vec.begin(), input_shape_vec.begin() + 2);

    // filter_shape_vec: {k_h, k_w} or {k_d, k_h, k_w}
    std::vector<int64_t> filter_shape_vec = framework::vectorize(filter.dims());
    filter_shape_vec.erase(filter_shape_vec.begin(),
                           filter_shape_vec.begin() + 2);

    // use col_shape in the im2col and col2im (or vol2col and col2vol)
    // calculation
    // col_shape_vec: {c, k_h, k_w, h, w} or {c, k_d, k_h, k_w, d, h, w}
    std::vector<int64_t> col_shape_vec;
    col_shape_vec.push_back(output_grad->dims()[1]);
    col_shape_vec.insert(col_shape_vec.end(), filter_shape_vec.begin(),
                         filter_shape_vec.end());
    col_shape_vec.insert(col_shape_vec.end(), input_shape_vec.begin(),
                         input_shape_vec.end());
    DDim col_shape(framework::make_ddim(col_shape_vec));

    // use col_matrix_shape in the gemm calculation
    // size: (c * k_h * k_w, h * w) or (c * k_d * k_h * k_w, d * h * w)
    DDim col_matrix_shape =
        framework::flatten_to_2d(col_shape, filter_shape_vec.size() + 1);

    // output size: (c, o_h, o_w) or (c, o_d, o_h, o_w)
    DDim output_shape = framework::slice_ddim(output_grad->dims(), 1,
                                              output_grad->dims().size());

    // input matrix size: (m, h * w) or (m, d * h * w)
    DDim input_matrix_shape = {input->dims()[1], col_matrix_shape[1]};

    // filter size: (m, c * k_h * k_w) or (m, c * k_d * k_h * k_w)
    DDim filter_matrix_shape = {input->dims()[1], col_matrix_shape[0]};
    filter.Resize(filter_matrix_shape);

    // convolution transpose grad on input:
    // im2col + gemm (similar to conv-forward)
    // input need to compute gradient
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
      math::SetConstant<Place, T> set_zero;

      if (input_grad) {
        input_grad->mutable_data<T>(context.GetPlace());
        set_zero(context.device_context(), input_grad, static_cast<T>(0));
      }
      if (filter_grad) {  // filter size (m, c, k_h, k_w)
        filter_grad->mutable_data<T>(context.GetPlace());
        set_zero(context.device_context(), filter_grad, static_cast<T>(0));
        filter_grad_ = *filter_grad;
        filter_grad_.Resize(filter_matrix_shape);
      }

      for (int i = 0; i < batch_size; i++) {
        // batch with size (c, o_h * o_w)
        Tensor output_grad_batch =
            output_grad->Slice(i, i + 1).Resize(output_shape);

        if (filter_shape_vec.size() == 2) {
          // im2col: dy -> col matrix
          // from (c, o_h, o_w) to (c * k_h * k_w, h * w)
          math::Im2ColFunctor<math::ColFormat::kCFO, Place, T> im2col;
          im2col(context.device_context(), output_grad_batch, col, strides[0],
                 strides[1], paddings[0], paddings[0], paddings[1],
                 paddings[1]);
        } else if (filter_shape_vec.size() == 3) {
          // vol2col: dy -> col_matrix
          // from (c, o_d, o_h, o_w) to (c * k_d * k_h * k_w, d * h * w)
          math::Vol2ColFunctor<Place, T> vol2col;
          vol2col(context.device_context(), output_grad_batch, col, strides[0],
                  strides[1], strides[2], paddings[0], paddings[1],
                  paddings[2]);
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
          math::matmul<Place, T>(context.device_context(), filter, false,
                                 col_matrix, false, static_cast<T>(1.0),
                                 &input_grad_batch, static_cast<T>(0.0));
        }
        if (filter_grad) {
          // input batch
          Tensor in_batch = input->Slice(i, i + 1).Resize(input_matrix_shape);
          // gemm: d_filter = x * dy^T
          // (m, c * h * w) * (k_h * k_w, c * h * w) -> (m, k_h * k_w)
          // or
          // (m, d * h * w) * (d * h * w, c * k_d * k_h * k_w) -> (m, c * k_d *
          // k_h * k_w)
          math::matmul<Place, T>(context.device_context(), in_batch, false,
                                 col_matrix, true, static_cast<T>(1.0),
                                 &filter_grad_, static_cast<T>(1.0));
        }
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle
