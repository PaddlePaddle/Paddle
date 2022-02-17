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
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/math/depthwise_conv.h"
#include "paddle/fluid/operators/math/im2col.h"
#include "paddle/fluid/operators/math/vol2col.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

template <typename DeviceContext, typename T, size_t D>
static void Slice(const framework::ExecutionContext& context,
                  const Tensor* input, Tensor* out,
                  const std::vector<int64_t>& begin_vec,
                  const std::vector<int64_t>& end_vec,
                  const std::vector<int64_t>& axes_vec) {
  auto& place =
      *context.template device_context<DeviceContext>().eigen_device();
  auto in_dims = input->dims();
  auto offsets = Eigen::DSizes<Eigen::DenseIndex, D>();
  auto extents = Eigen::DSizes<Eigen::DenseIndex, D>();
  for (size_t i = 0; i < D; ++i) {
    offsets[i] = 0;
    extents[i] = in_dims[i];
  }

  std::vector<int64_t> out_shape_vec = framework::vectorize(in_dims);
  for (size_t i = 0; i < axes_vec.size(); ++i) {
    offsets[axes_vec[i]] = begin_vec[i];
    extents[axes_vec[i]] = end_vec[i] - begin_vec[i];
    out_shape_vec[axes_vec[i]] = end_vec[i] - begin_vec[i];
  }

  framework::DDim out_dims(framework::make_ddim(out_shape_vec));
  out->mutable_data<T>(out_dims, context.GetPlace());

  auto in_t =
      framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
          *input);
  auto out_t =
      framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
          *out, out_dims);

  EigenSlice<std::decay_t<decltype(place)>, T, D>::Eval(place, out_t, in_t,
                                                        offsets, extents);
  out->Resize(out_dims);
}

template <typename DeviceContext, typename T, size_t D>
static void Slice(const framework::ExecutionContext& context,
                  const Tensor* input, Tensor* out, int64_t begin_idx,
                  int64_t end_idx, int64_t axes) {
  std::vector<int64_t> begin_vec = {begin_idx};
  std::vector<int64_t> end_vec = {end_idx};
  std::vector<int64_t> axes_vec = {axes};
  Slice<DeviceContext, T, D>(context, input, out, begin_vec, end_vec, axes_vec);
}

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

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override;
};

class ConvTransposeOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;
};

class ConvTransposeOpDoubleGrad : public framework::OperatorWithKernel {
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
    const std::string data_layout_str =
        context.Attr<std::string>("data_format");
    const framework::DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    const Tensor* input = context.Input<Tensor>("Input");
    // The filter will be reshaped, so it should not be constant pointer
    Tensor filter = *context.Input<Tensor>("Filter");
    Tensor* output = context.Output<Tensor>("Output");

    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    int groups = context.Attr<int>("groups");
    std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");

    auto in_dims = input->dims();
    auto filter_dims = filter.dims();
    auto out_dims = output->dims();
    const int batch_size = static_cast<int>(input->dims()[0]);

    framework::DDim in_data_dims;
    if (data_layout != framework::DataLayout::kNHWC) {
      in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
    } else {
      in_data_dims = framework::slice_ddim(in_dims, 1, in_dims.size() - 1);
    }
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    // input_shape_vec: {n, c, h, w} or {n, c, d, h, w} for channel_first
    // input_shape_vec: {n, h, w, c} or {n, d, h, w, c} for channel_last
    std::vector<int64_t> input_shape_vec = framework::vectorize(input->dims());
    // filter_shape_vec: {k_o, k_i, k_h, k_w} or {k_o, k_i, k_d, k_h, k_w}
    std::vector<int64_t> filter_shape_vec = framework::vectorize(filter.dims());

    // use col_shape in the im2col and col2im (or vol2col and col2vol)
    // calculation
    // col_shape_vec: {o_c/g, k_h, k_w, h, w} or {o_c/g, k_d, k_h, k_w, d, h, w}
    size_t data_dim = filter_shape_vec.size() - 2;
    std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
    if (data_layout != framework::DataLayout::kNHWC) {
      col_shape_vec[0] = out_dims[1] / groups;
      for (size_t j = 0; j < data_dim; ++j) {
        col_shape_vec[j + 1] = filter_shape_vec[j + 2];
        col_shape_vec[j + 1 + data_dim] = input_shape_vec[j + 2];
      }
    } else {
      col_shape_vec[0] = out_dims[out_dims.size() - 1] / groups;
      for (size_t j = 0; j < data_dim; ++j) {
        col_shape_vec[j + 1] = filter_shape_vec[j + 2];
        col_shape_vec[j + 1 + data_dim] = input_shape_vec[j + 1];
      }
    }
    DDim col_shape(framework::make_ddim(col_shape_vec));

    // use col_matrix_shape in the gemm calculation
    // size: (o_c/g * k_h * k_w, h * w) or (o_c/g * k_d * k_h * k_w, d * h * w)
    DDim col_matrix_shape = framework::flatten_to_2d(col_shape, data_dim + 1);

    Tensor col;
    col.mutable_data<T>(col_shape, context.GetPlace());
    // col_matrix shares the same piece of data with col,
    // but will be reshaped into a two-dimensional matrix shape
    // to call the matrix multiplication interface.
    Tensor col_matrix;
    col_matrix.ShareDataWith(col);
    col_matrix.Resize(col_matrix_shape);

    // output size: (o_c, o_h, o_w) or (o_c, o_d, o_h, o_w) for channel_first
    // output size: (o_h, o_w, o_c) or (o_d, o_h, o_w, o_c) for channel_last
    DDim output_shape =
        framework::slice_ddim(output->dims(), 1, output->dims().size());

    // input matrix size: (i_c, h * w) or (i_c, d * h * w) for channel_first
    // input matrix size: (h * w, i_c) or (d * h * w, i_c) for channel_last
    DDim input_matrix_shape;
    if (data_layout != framework::DataLayout::kNHWC) {
      input_matrix_shape = {in_dims[1], col_matrix_shape[1]};
    } else {
      input_matrix_shape = {col_matrix_shape[1], in_dims[in_dims.size() - 1]};
    }

    // filter size: (i_c, o_c/g * k_h * k_w) or (i_c, o_c/g * k_d * k_h * k_w)
    DDim filter_matrix_shape;
    if (data_layout != framework::DataLayout::kNHWC) {
      filter_matrix_shape = {in_dims[1], col_matrix_shape[0]};
    } else {
      filter_matrix_shape = {in_dims[in_dims.size() - 1], col_matrix_shape[0]};
    }
    filter.Resize(filter_matrix_shape);

    output->mutable_data<T>(context.GetPlace());
    pten::funcs::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    set_zero(dev_ctx, output, static_cast<T>(0));

    int in_step =
        (data_layout != framework::DataLayout::kNHWC
             ? static_cast<int>(in_dims[1]) / groups
             : static_cast<int>(in_dims[in_dims.size() - 1]) / groups);

    int out_step =
        (data_layout != framework::DataLayout::kNHWC
             ? static_cast<int>(out_dims[1]) / groups
             : static_cast<int>(out_dims[out_dims.size() - 1]) / groups);
    math::Col2ImFunctor<math::ColFormat::kCFO, DeviceContext, T> col2im;
    math::Col2VolFunctor<DeviceContext, T> col2vol;
    math::ConcatFunctor<DeviceContext, T> concat_functor;

    // convolution transpose: gemm + col2im or col2vol (similar to conv-backward
    // on input)
    size_t D = input->dims().size();
    for (int i = 0; i < batch_size; i++) {
      // batch with size (i_c, h * w) or (i_c, d * h * w) for channel_first
      // batch with size (h * w, i_c) or (d * h * w, i_c) for channel_last
      Tensor input_batch = input->Slice(i, i + 1).Resize(input_matrix_shape);

      // output size: (o_c, o_h, o_w) or (o_c, o_d, o_h, o_w) for channel_first
      // output size: (o_h, o_w, o_c) or (o_d, o_h, o_w, o_c) for channel_last
      Tensor output_batch = output->Slice(i, i + 1).Resize(output_shape);

      std::vector<Tensor> output_batch_vec;
      for (int g = 0; g < groups; g++) {
        int64_t start = g * in_step;
        int64_t end = (g + 1) * in_step;
        int axes = (data_layout != framework::DataLayout::kNHWC ? 0 : 1);
        Tensor filter_slice = filter.Slice(g * in_step, (g + 1) * in_step);
        Tensor in_slice, out_slice;

        // col_matrix = filter_slice * input_slice
        // of shape (o_c/g * k_h * k_w, h * w)
        // or (o_c/g * k_d * k_h * k_w, d * h * w)
        if (data_layout != framework::DataLayout::kNHWC) {
          in_slice = input_batch.Slice(g * in_step, (g + 1) * in_step);
          out_slice = output_batch.Slice(g * out_step, (g + 1) * out_step);
          blas.MatMul(filter_slice, true, in_slice, false, static_cast<T>(1.0),
                      &col_matrix, static_cast<T>(0.0));
        } else {
          Slice<DeviceContext, T, 2>(context, &input_batch, &in_slice, start,
                                     end, axes);
          start = g * out_step;
          end = (g + 1) * out_step;
          axes = D - 2;
          if (D == 4U) {
            Slice<DeviceContext, T, 3>(context, &output_batch, &out_slice,
                                       start, end, axes);
          } else if (D == 5U) {
            Slice<DeviceContext, T, 4>(context, &output_batch, &out_slice,
                                       start, end, axes);
          }
          blas.MatMul(filter_slice, true, in_slice, true, static_cast<T>(1.0),
                      &col_matrix, static_cast<T>(0.0));
        }

        if (data_dim == 2U) {
          // col2im: col_matrix -> dy
          // from (o_c/g * k_h * k_w, h * w) to (o_c/g, o_h, o_w) or (o_h, o_w,
          // o_c/g)
          col2im(dev_ctx, col, dilations, strides,
                 std::vector<int>{paddings[0], paddings[2], paddings[1],
                                  paddings[3]},
                 &out_slice, data_layout);
        } else if (data_dim == 3U) {
          // col2vol: col_matrix -> dy
          // from (o_c/g * k_d * k_h * k_w, d * h * w) to (o_c/g, o_d, o_h, o_w)
          // or (o_d, o_h, o_w, o_c/g)
          col2vol(dev_ctx, col, dilations, strides, paddings, &out_slice,
                  data_layout);
        }
        if (data_layout == framework::DataLayout::kNHWC) {
          output_batch_vec.push_back(out_slice);
        }
      }
      if (data_layout == framework::DataLayout::kNHWC) {
        concat_functor(dev_ctx, output_batch_vec, static_cast<int>(D - 2),
                       &output_batch);
      }
    }
  }
};

template <typename DeviceContext, typename T>
class GemmConvTransposeGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const std::string data_layout_str =
        context.Attr<std::string>("data_format");
    const framework::DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
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
    std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");

    auto in_dims = input->dims();
    auto filter_dims = filter.dims();
    auto out_grad_dims = output_grad->dims();
    const int batch_size = static_cast<int>(input->dims()[0]);

    framework::DDim in_data_dims;
    if (data_layout != framework::DataLayout::kNHWC) {
      in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
    } else {
      in_data_dims = framework::slice_ddim(in_dims, 1, in_dims.size() - 1);
    }
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    // input_shape_vec: {n, c, h, w} or {n, c, d, h, w} for channel_first
    // input_shape_vec: {n, h, w, c} or {n, d, h, w, c} for channel_last
    std::vector<int64_t> input_shape_vec = framework::vectorize(input->dims());
    // filter_shape_vec: {i_c, o_c, k_h, k_w} or {i_c, o_c, k_d, k_h, k_w}
    std::vector<int64_t> filter_shape_vec = framework::vectorize(filter.dims());

    // use col_shape in the im2col and col2im (or vol2col and col2vol)
    // calculation
    // col_shape_vec: {o_c, k_h, k_w, h, w} or {o_c, k_d, k_h, k_w, d, h, w} for
    size_t data_dim = filter_shape_vec.size() - 2;
    std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
    if (data_layout != framework::DataLayout::kNHWC) {
      col_shape_vec[0] = out_grad_dims[1];
      for (size_t j = 0; j < data_dim; ++j) {
        col_shape_vec[j + 1] = filter_shape_vec[j + 2];
        col_shape_vec[j + 1 + data_dim] = input_shape_vec[j + 2];
      }
    } else {
      col_shape_vec[0] = out_grad_dims[out_grad_dims.size() - 1];
      for (size_t j = 0; j < data_dim; ++j) {
        col_shape_vec[j + 1] = filter_shape_vec[j + 2];
        col_shape_vec[j + 1 + data_dim] = input_shape_vec[j + 1];
      }
    }
    DDim col_shape(framework::make_ddim(col_shape_vec));

    // use col_matrix_shape in the gemm calculation
    // size: (o_c * k_h * k_w, h * w) or (o_c * k_d * k_h * k_w, d * h * w)
    DDim col_matrix_shape = framework::flatten_to_2d(col_shape, data_dim + 1);

    // output size: (o_c, o_h, o_w) or (o_c, o_d, o_h, o_w) for channel_first
    // output size: (o_h, o_w, o_c) or (o_d, o_h, o_w, o_c) for channel_last
    DDim output_shape = framework::slice_ddim(output_grad->dims(), 1,
                                              output_grad->dims().size());

    // input matrix size: (i_c, h * w) or (i_c, d * h * w) for channel_first
    // input matrix size: (h * w, i_c) or (d * h * w, i_c) for channel_last
    DDim input_matrix_shape;
    if (data_layout != framework::DataLayout::kNHWC) {
      input_matrix_shape = {in_dims[1], col_matrix_shape[1]};
    } else {
      input_matrix_shape = {col_matrix_shape[1], in_dims[in_dims.size() - 1]};
    }

    // filter size: (i_c, o_c/g * k_h * k_w) or (i_c, o_c/g * k_d * k_h * k_w)
    DDim filter_matrix_shape;
    if (data_layout != framework::DataLayout::kNHWC) {
      filter_matrix_shape = {in_dims[1], col_matrix_shape[0] / groups};
    } else {
      filter_matrix_shape = {in_dims[in_dims.size() - 1],
                             col_matrix_shape[0] / groups};
    }
    filter.Resize(filter_matrix_shape);

    int in_step =
        (data_layout != framework::DataLayout::kNHWC
             ? static_cast<int>(in_dims[1]) / groups
             : static_cast<int>(in_dims[in_dims.size() - 1]) / groups);
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
      pten::funcs::SetConstant<DeviceContext, T> set_zero;

      math::Im2ColFunctor<math::ColFormat::kCFO, DeviceContext, T> im2col;
      math::Vol2ColFunctor<DeviceContext, T> vol2col;
      math::ConcatFunctor<DeviceContext, T> concat_functor;

      if (input_grad) {
        input_grad->mutable_data<T>(context.GetPlace());
        set_zero(dev_ctx, input_grad, static_cast<T>(0));
      }
      if (filter_grad) {  // filter_grad_ size (i_c, o_c/g, k_h, k_w)
        filter_grad->mutable_data<T>(context.GetPlace());
        set_zero(dev_ctx, filter_grad, static_cast<T>(0));
        filter_grad_ = *filter_grad;
        filter_grad_.Resize(filter_matrix_shape);
      }

      size_t D = input->dims().size();
      for (int i = 0; i < batch_size; i++) {
        // batch with size (o_c, o_h, o_w) or (o_c, o_d, o_h, o_w) for
        // channel_first
        // batch with size (o_h, o_w, o_c) or (o_d, o_h, o_w, o_c) for
        // channel_last
        Tensor output_grad_batch =
            output_grad->Slice(i, i + 1).Resize(output_shape);

        if (data_dim == 2U) {
          // im2col: dy -> col matrix
          // from (o_c, o_h, o_w) to (o_c * k_h * k_w, i_h * i_w) for
          // channel_first
          // from (o_h, o_w, o_c) to (o_c * k_h * k_w, i_h * i_w) for
          // channel_last
          im2col(dev_ctx, output_grad_batch, dilations, strides,
                 std::vector<int>{paddings[0], paddings[2], paddings[1],
                                  paddings[3]},
                 &col, data_layout);
        } else if (data_dim == 3U) {
          // vol2col: dy -> col_matrix
          // from (o_c, o_d, o_h, o_w) to (o_c * k_d * k_h * k_w, i_d * i_h *
          // i_w) for channel_first
          // from (o_d, o_h, o_w, o_c) to (i_d * i_h * i_w, o_c * k_d * k_h *
          // k_w) for channel_last
          vol2col(dev_ctx, output_grad_batch, dilations, strides, paddings,
                  &col, data_layout);
        }

        if (input_grad) {
          // batch with size (i_c, i_h, i_w) or (i_h, i_w, i_c)
          Tensor input_grad_batch =
              input_grad->Slice(i, i + 1).Resize(input_matrix_shape);

          // gemm: dx = filter * dy
          // (i_c, o_c * k_h * k_w) * (o_c * k_h * k_w, i_h * i_w) -> (i_c, i_h
          // * i_w)
          // or
          // (i_c, o_c * k_d * k_h * k_w) * (o_c * k_d * k_h * k_w, i_d * i_h *
          // i_w) -> (i_c,
          // i_d, i_h, i_w)
          // gemm: dx = dy^T * filter^T for channel_last

          std::vector<Tensor> input_grad_batch_vec;
          for (int g = 0; g < groups; g++) {
            // input_grad_slice: (i_c/g, i_h * i_w) or (i_c/g, i_d * i_h * i_w)
            // for channel_first
            // input_grad_slice: (i_h * i_w, i_c/g) or (i_d * i_h * i_w, i_c/g)
            // for channel_last
            // filter_slice: (i_c/g, o_c/g * k_h * k_w)
            Tensor filter_slice = filter.Slice(g * in_step, (g + 1) * in_step);
            // col_matrix_slice: (o_c/g * k_h * k_w, h * w) or (o_c/g * k_d *
            // k_h * k_w, d * h * w)
            Tensor col_matrix_slice =
                col_matrix.Slice(g * col_step, (g + 1) * col_step);
            if (data_layout != framework::DataLayout::kNHWC) {
              Tensor input_grad_slice =
                  input_grad_batch.Slice(g * in_step, (g + 1) * in_step);
              blas.MatMul(filter_slice, false, col_matrix_slice, false,
                          static_cast<T>(1.0), &input_grad_slice,
                          static_cast<T>(0.0));
            } else {
              Tensor input_grad_slice;
              Slice<DeviceContext, T, 2>(context, &input_grad_batch,
                                         &input_grad_slice, g * in_step,
                                         (g + 1) * in_step, 1);
              blas.MatMul(col_matrix_slice, true, filter_slice, true,
                          static_cast<T>(1.0), &input_grad_slice,
                          static_cast<T>(0.0));
              DDim input_grad_slice_shape;
              if (data_dim == 2U) {
                input_grad_slice_shape = {in_dims[1], in_dims[2], in_step};
              } else {
                input_grad_slice_shape = {in_dims[1], in_dims[2], in_dims[3],
                                          in_step};
              }
              input_grad_slice =
                  input_grad_slice.Resize(input_grad_slice_shape);
              input_grad_batch_vec.push_back(input_grad_slice);
            }
          }
          if (data_layout == framework::DataLayout::kNHWC) {
            concat_functor(dev_ctx, input_grad_batch_vec,
                           static_cast<int>(D - 2), &input_grad_batch);
          }
        }
        if (filter_grad) {
          // input batch: (i_c, i_h * i_w) or (i_h, i_w * i_c)
          Tensor in_batch = input->Slice(i, i + 1).Resize(input_matrix_shape);
          // gemm: d_filter = x * dy^T
          // (i_c, i_h * i_w) * (i_h * i_w, o_c * k_h * k_w) -> (i_c, o_c * k_h
          // * k_w)
          // or
          // (i_c, i_d * i_h * i_w) * (i_d * i_h * i_w, o_c * k_d * k_h * k_w)
          // -> (i_c, o_c * k_d *
          // k_h * k_w)
          // gemm: d_filter = x^T * dy^T for channel_last

          for (int g = 0; g < groups; g++) {
            Tensor filter_grad_slice =
                filter_grad_.Slice(g * in_step, (g + 1) * in_step);
            Tensor col_matrix_slice =
                col_matrix.Slice(g * col_step, (g + 1) * col_step);
            if (data_layout != framework::DataLayout::kNHWC) {
              Tensor in_batch_slice =
                  in_batch.Slice(g * in_step, (g + 1) * in_step);
              blas.MatMul(in_batch_slice, false, col_matrix_slice, true,
                          static_cast<T>(1.0), &filter_grad_slice,
                          static_cast<T>(1.0));
            } else {
              Tensor in_batch_slice;
              Slice<DeviceContext, T, 2>(context, &in_batch, &in_batch_slice,
                                         g * in_step, (g + 1) * in_step, 1);
              blas.MatMul(in_batch_slice, true, col_matrix_slice, true,
                          static_cast<T>(1.0), &filter_grad_slice,
                          static_cast<T>(1.0));
            }
          }
        }
      }
    }
  }
};

template <typename DeviceContext, typename T>
class DepthwiseConvTransposeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const std::string data_layout_str =
        context.Attr<std::string>("data_format");
    const framework::DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    const Tensor* input = context.Input<Tensor>("Input");
    Tensor filter = *context.Input<Tensor>("Filter");
    Tensor* output = context.Output<Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());

    int groups = context.Attr<int>("groups");
    PADDLE_ENFORCE_EQ(
        groups, filter.dims()[0],
        platform::errors::InvalidArgument(
            "groups should be error to the 1st dimension of filter. But "
            "received groups is %d and filter dimension[0] is %d",
            groups, filter.dims()[0]));

    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");
    for (auto v : dilations) {
      PADDLE_ENFORCE_EQ(v, 1, platform::errors::InvalidArgument(
                                  "dilations should be 1 in depthwise conv. "
                                  "But received dilations is %d",
                                  v));
    }

    auto in_dims = input->dims();
    auto filter_dims = filter.dims();

    framework::DDim in_data_dims;
    if (data_layout != framework::DataLayout::kNHWC) {
      in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
    } else {
      in_data_dims = framework::slice_ddim(in_dims, 1, in_dims.size() - 1);
    }
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    output->mutable_data<T>(context.GetPlace());
    auto& dev_ctx = context.template device_context<DeviceContext>();
    pten::funcs::SetConstant<DeviceContext, T> set_zero;
    set_zero(dev_ctx, output, static_cast<T>(0));

    math::DepthwiseConvInputGradFunctor<DeviceContext, T>
        depthwiseConvInputGrad;
    depthwiseConvInputGrad(
        dev_ctx, *output, filter, *input, strides,
        std::vector<int>{paddings[0], paddings[2], paddings[1], paddings[3]},
        dilations, output, data_layout);
  }
};

template <typename DeviceContext, typename T>
class DepthwiseConvTransposeGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const std::string data_layout_str =
        context.Attr<std::string>("data_format");
    const framework::DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    const Tensor* input = context.Input<Tensor>("Input");
    const Tensor* output_grad =
        context.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad =
        context.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad =
        context.Output<Tensor>(framework::GradVarName("Filter"));
    Tensor filter = *context.Input<Tensor>("Filter");

    if (!input_grad && !filter_grad) return;

    auto& dev_ctx = context.template device_context<DeviceContext>();
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");

    auto in_dims = input->dims();
    auto filter_dims = filter.dims();

    framework::DDim in_data_dims;
    if (data_layout != framework::DataLayout::kNHWC) {
      in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
    } else {
      in_data_dims = framework::slice_ddim(in_dims, 1, in_dims.size() - 1);
    }
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    if (input_grad) {
      math::DepthwiseConvFunctor<DeviceContext, T> depthwiseConv;
      depthwiseConv(
          dev_ctx, *output_grad, filter, strides,
          std::vector<int>{paddings[0], paddings[2], paddings[1], paddings[3]},
          dilations, input_grad, data_layout);
    }

    if (filter_grad) {
      pten::funcs::SetConstant<DeviceContext, T> set_zero;
      filter_grad->mutable_data<T>(context.GetPlace());
      set_zero(dev_ctx, filter_grad, static_cast<T>(0));

      math::DepthwiseConvFilterGradFunctor<DeviceContext, T>
          depthwiseConvFilterGrad;
      depthwiseConvFilterGrad(
          dev_ctx, *output_grad, *input, strides,
          std::vector<int>{paddings[0], paddings[2], paddings[1], paddings[3]},
          dilations, filter_grad, data_layout);
    }
  }
};
}  // namespace operators
}  // namespace paddle
