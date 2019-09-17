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

#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/depthwise_conv.h"
#include "paddle/fluid/operators/math/im2col.h"
#include "paddle/fluid/operators/math/vol2col.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
constexpr int kConvMKLDNNFP32 = 1;
constexpr int kConvMKLDNNINT8 = 2;
constexpr int MaxKeyLength = 256;

// Base convolution operator definations for other conv
// like operators to reuse the implementation.
inline int ConvOutputSize(int input_size, int filter_size, int dilation,
                          int padding, int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + 2 * padding - dkernel) / stride + 1;
  PADDLE_ENFORCE(
      output_size > 0,
      "Due to the settings of padding(%d), filter_size(%d), dilation(%d) and "
      "stride(%d), the output size is less than 0, please check "
      "again. Input_size:%d",
      padding, filter_size, dilation, stride, input_size);

  return output_size;
}
inline bool IsExpand(const std::vector<int64_t>& filter_dim,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations) {
  bool filter_1 = true, strides_1 = true, padding_0 = true, dilation_1 = true;
  for (size_t j = 0; j < strides.size(); ++j) {
    filter_1 = filter_1 && (static_cast<int>(filter_dim[j + 2]) == 1);
    strides_1 = strides_1 && (strides[j] == 1);
    padding_0 = padding_0 && (paddings[j] == 0);
    dilation_1 = dilation_1 && (dilations[j] == 1);
  }
  return !(filter_1 && strides_1 && padding_0 && dilation_1);
}

// Define Op classes in .h file so that other conv
// operator implementations can reuse the code.
class Conv2DOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final;

 protected:
  virtual void Apply() {}
};

class Conv3DOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final;

 protected:
  virtual void Apply() {}
};

class ConvOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string> GetInputOutputWithSameType()
      const override {
    return std::unordered_map<std::string, std::string>{
        {"Input", /*->*/ "Output"}};
  }
};

class ConvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;
};

class ConvOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;
};

class ConvOpDoubleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;
};

template <typename DeviceContext, typename T>
class GemmConvKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    // The filter will be reshaped in the calculations,
    // so here use an assignment operation,
    // that avoids modifying the variable in the Scope.
    Tensor filter = *context.Input<Tensor>("Filter");
    Tensor* output = context.Output<Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());

    int groups = context.Attr<int>("groups");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");

    auto& dev_ctx = context.template device_context<DeviceContext>();

    const int batch_size = static_cast<int>(input->dims()[0]);

    // filter_shape_vec: {k_o, k_i, k_h, k_w} or {k_o, k_i, k_d, k_h, k_w}
    std::vector<int64_t> filter_shape_vec(framework::vectorize(filter.dims()));
    // output_shape_vec: {o_n, o_c, o_h, o_w} or {o_n, o_c, o_d, o_h, o_w}
    std::vector<int64_t> output_shape_vec(framework::vectorize(output->dims()));

    // use col_shape in the im2col calculation
    // col_shape_vec: {i_c/g, k_h, k_w, o_h, o_w} or {i_c/g, k_d, k_h, k_w, o_d,
    // o_h, o_w}
    size_t data_dim = filter_shape_vec.size() - 2;
    std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
    col_shape_vec[0] = input->dims()[1] / groups;
    for (size_t j = 0; j < data_dim; ++j) {
      col_shape_vec[j + 1] = filter_shape_vec[j + 2];
      col_shape_vec[j + 1 + data_dim] = output_shape_vec[j + 2];
    }
    framework::DDim col_shape(framework::make_ddim(col_shape_vec));

    // use col_matrix_shape in the gemm calculation
    // size: (i_c/g * k_h * k_w, o_h * o_w) or (i_c/g * k_d * k_h * k_w, o_d *
    // o_h * o_w)
    framework::DDim col_matrix_shape =
        framework::flatten_to_2d(col_shape, data_dim + 1);

    bool is_expand = IsExpand(filter_shape_vec, strides, paddings, dilations);
    Tensor col;
    // col_matrix shares the same piece of data with col,
    // but will be reshaped into a two-dimensional matrix shape
    // to call the matrix multiplication interface.
    Tensor col_matrix;
    if (is_expand) {
      col = context.AllocateTmpTensor<T, DeviceContext>(col_shape, dev_ctx);
      col_matrix.ShareDataWith(col);
      col_matrix.Resize(col_matrix_shape);
    }

    framework::DDim input_shape =
        framework::slice_ddim(input->dims(), 1, input->dims().size());

    framework::DDim filter_matrix_shape = {filter.dims()[0],
                                           filter.numel() / filter.dims()[0]};
    filter.Resize(filter_matrix_shape);

    framework::DDim output_matrix_shape = {
        output->dims()[1],
        output->numel() / (output->dims()[0] * output->dims()[1])};

    // convolution operator: im2col(or vol2col) + gemm
    int in_step = static_cast<int>(input->dims()[1]) / groups;
    int out_step = static_cast<int>(output->dims()[1]) / groups;

    math::Vol2ColFunctor<DeviceContext, T> vol2col;
    math::Im2ColFunctor<math::ColFormat::kCFO, DeviceContext, T> im2col;

    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    for (int i = 0; i < batch_size; i++) {
      Tensor in_batch = input->Slice(i, i + 1).Resize(input_shape);
      Tensor out_batch = output->Slice(i, i + 1).Resize(output_matrix_shape);

      for (int g = 0; g < groups; g++) {
        Tensor in_slice = in_batch.Slice(g * in_step, (g + 1) * in_step);

        if (!is_expand) {
          col.ShareDataWith(in_slice);
          col_matrix.ShareDataWith(col);
          col_matrix.Resize(col_matrix_shape);
        } else if (data_dim == 2U) {
          // im2col
          im2col(dev_ctx, in_slice, dilations, strides,
                 std::vector<int>{paddings[0], paddings[1], paddings[0],
                                  paddings[1]},
                 &col);
        } else if (data_dim == 3U) {
          // vol2col
          vol2col(dev_ctx, in_slice, dilations, strides, paddings, &col);
        }

        // gemm
        Tensor out_slice = out_batch.Slice(g * out_step, (g + 1) * out_step);
        Tensor filter_slice = filter.Slice(g * out_step, (g + 1) * out_step);
        blas.MatMul(filter_slice, false, col_matrix, false, T(1.0), &out_slice,
                    T(0.0));
      }
    }
  }
};

template <typename DeviceContext, typename T>
class GemmConvGradKernel : public framework::OpKernel<T> {
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

    if (!input_grad && !filter_grad) return;

    int groups = context.Attr<int>("groups");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");

    const int batch_size = static_cast<int>(input->dims()[0]);

    auto& dev_ctx = context.template device_context<DeviceContext>();

    // filter_shape_vec: {k_o, k_i, k_h, k_w} or {k_o, k_i, k_d, k_h, k_w}
    std::vector<int64_t> filter_shape_vec(framework::vectorize(filter.dims()));
    // output_shape_vec: {o_n, o_c, o_h, o_w} or {o_n, o_c, o_d, o_h, o_w}
    std::vector<int64_t> output_shape_vec(
        framework::vectorize(output_grad->dims()));

    // use col_shape in the im2col calculation
    // col_shape_vec: {i_c/g, k_h, k_w, o_h, o_w} or {i_c/g, k_d, k_h, k_w, o_d,
    // o_h, o_w}
    size_t data_dim = filter_shape_vec.size() - 2;
    std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
    col_shape_vec[0] = input->dims()[1] / groups;
    for (size_t j = 0; j < data_dim; ++j) {
      col_shape_vec[j + 1] = filter_shape_vec[j + 2];
      col_shape_vec[j + 1 + data_dim] = output_shape_vec[j + 2];
    }
    framework::DDim col_shape(framework::make_ddim(col_shape_vec));

    // use col_matrix_shape in the gemm calculation
    // size: (i_c/g * k_h * k_w, o_h * o_w)
    // or
    // (i_c/g * k_d * k_h * k_w, o_d * o_h * o_w)
    framework::DDim col_matrix_shape =
        framework::flatten_to_2d(col_shape, data_dim + 1);

    framework::DDim input_shape =
        framework::slice_ddim(input->dims(), 1, input->dims().size());

    framework::DDim filter_matrix_shape = {filter.dims()[0],
                                           filter.numel() / filter.dims()[0]};
    filter.Resize(filter_matrix_shape);

    framework::DDim output_matrix_shape = {
        output_grad->dims()[1],
        output_grad->numel() /
            (output_grad->dims()[0] * output_grad->dims()[1])};

    // convolution backward input operator:  gemm + col2im(or col2vol)
    // convolution backward weight operator: im2col(or vol2col) + gemm
    int in_step = static_cast<int>(input->dims()[1]) / groups;
    int out_step = static_cast<int>(output_grad->dims()[1]) / groups;

    bool is_expand = IsExpand(filter_shape_vec, strides, paddings, dilations);
    Tensor col;
    // col_matrix shares the same piece of data with col,
    // but will be reshaped into a two-dimensional matrix shape
    // to call the matrix multiplication interface.
    Tensor col_matrix;
    if (is_expand) {
      col = context.AllocateTmpTensor<T, DeviceContext>(col_shape, dev_ctx);
      col_matrix.ShareDataWith(col);
      col_matrix.Resize(col_matrix_shape);
    }

    math::SetConstant<DeviceContext, T> set_zero;
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);

    if (input_grad) {
      input_grad->mutable_data<T>(context.GetPlace());

      // if is_expand is false, the operation of set_zero is unnecessary,
      // because math::matmul will reset input_grad.
      if (is_expand) {
        set_zero(dev_ctx, input_grad, static_cast<T>(0));
      }
      math::Col2VolFunctor<DeviceContext, T> col2vol;
      math::Col2ImFunctor<math::ColFormat::kCFO, DeviceContext, T> col2im;

      for (int i = 0; i < batch_size; i++) {
        Tensor out_grad_batch =
            output_grad->Slice(i, i + 1).Resize(output_matrix_shape);
        Tensor in_grad_batch = input_grad->Slice(i, i + 1).Resize(input_shape);
        for (int g = 0; g < groups; g++) {
          // gemm
          Tensor out_grad_slice =
              out_grad_batch.Slice(g * out_step, (g + 1) * out_step);
          Tensor filter_slice = filter.Slice(g * out_step, (g + 1) * out_step);

          Tensor in_grad_slice =
              in_grad_batch.Slice(g * in_step, (g + 1) * in_step);

          if (!is_expand) {
            col_matrix.ShareDataWith(in_grad_slice);
            col_matrix.Resize(col_matrix_shape);
          }
          blas.MatMul(filter_slice, true, out_grad_slice, false, T(1.0),
                      &col_matrix, T(0.0));

          if (is_expand && data_dim == 2U) {
            col2im(dev_ctx, col, dilations, strides,
                   std::vector<int>{paddings[0], paddings[1], paddings[0],
                                    paddings[1]},
                   &in_grad_slice);
          } else if (is_expand && data_dim == 3U) {
            col2vol(dev_ctx, col, dilations, strides, paddings, &in_grad_slice);
          }
        }
      }
    }

    if (filter_grad) {
      filter_grad->mutable_data<T>(context.GetPlace());
      Tensor filter_grad_ = *filter_grad;
      filter_grad_.Resize(filter_matrix_shape);
      set_zero(dev_ctx, filter_grad, static_cast<T>(0));
      math::Im2ColFunctor<math::ColFormat::kCFO, DeviceContext, T> im2col;
      math::Vol2ColFunctor<DeviceContext, T> vol2col;
      for (int i = 0; i < batch_size; i++) {
        Tensor out_grad_batch =
            output_grad->Slice(i, i + 1).Resize(output_matrix_shape);
        Tensor in_batch = input->Slice(i, i + 1).Resize(input_shape);
        for (int g = 0; g < groups; g++) {
          // im2col
          Tensor out_grad_slice =
              out_grad_batch.Slice(g * out_step, (g + 1) * out_step);
          Tensor in_slice = in_batch.Slice(g * in_step, (g + 1) * in_step);

          if (!is_expand) {
            col.ShareDataWith(in_slice);
            col_matrix.ShareDataWith(col);
            col_matrix.Resize(col_matrix_shape);
          } else if (data_dim == 2U) {
            im2col(dev_ctx, in_slice, dilations, strides,
                   std::vector<int>{paddings[0], paddings[1], paddings[0],
                                    paddings[1]},
                   &col);
          } else if (data_dim == 3U) {
            vol2col(dev_ctx, in_slice, dilations, strides, paddings, &col);
          }

          // gemm
          Tensor filter_grad_slice =
              filter_grad_.Slice(g * out_step, (g + 1) * out_step);
          blas.MatMul(out_grad_slice, false, col_matrix, true, T(1.0),
                      &filter_grad_slice, T(1.0));
        }
      }
    }
  }
};

template <typename DeviceContext, typename T>
class GemmConvDoubleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      "It must use CPUPlace.");
    const Tensor* X = ctx.Input<Tensor>("Input");
    const Tensor* dY = ctx.Input<Tensor>("DOutput");
    const Tensor* ddX = ctx.Input<Tensor>("DDInput");
    const Tensor* ddW_in = ctx.Input<Tensor>("DDFilter");

    Tensor* ddY = ctx.Output<Tensor>("DDOutput");
    Tensor* dW = ctx.Output<Tensor>("DFilter");
    Tensor* dX = ctx.Output<Tensor>("DInput");
    Tensor W = detail::Ref(ctx.Input<Tensor>("Filter"),
                           "Cannot find input Filter(%s) in scope)",
                           ctx.Inputs("Filter")[0]);

    if (!ddY && !dW && !dX) return;
    int groups = ctx.Attr<int>("groups");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");

    const int batch_size = static_cast<int>(X->dims()[0]);
    std::vector<int64_t> filter_shape_vec(framework::vectorize(W.dims()));
    std::vector<int64_t> output_shape_vec(framework::vectorize(dY->dims()));

    size_t data_dim = filter_shape_vec.size() - 2;
    std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
    // col_shape [in_channel/group, kh, kw, oh, ow]
    col_shape_vec[0] = X->dims()[1] / groups;
    for (size_t j = 0; j < data_dim; ++j) {
      col_shape_vec[j + 1] = filter_shape_vec[j + 2];
      col_shape_vec[j + data_dim + 1] = output_shape_vec[j + 2];
    }
    framework::DDim col_shape(framework::make_ddim(col_shape_vec));
    // col_matrix_shape [in_channel/group * kh * kw, oh * ow]
    framework::DDim col_matrix_shape =
        framework::flatten_to_2d(col_shape, data_dim + 1);
    // input_shape [Cin, H, W]
    framework::DDim input_shape =
        framework::slice_ddim(X->dims(), 1, X->dims().size());
    // filter_matrix_shape [Cout, Cin * kh * kw]
    framework::DDim filter_matrix_shape = {W.dims()[0],
                                           W.numel() / W.dims()[0]};

    W.Resize(filter_matrix_shape);
    framework::DDim output_matrix_shape = {
        dY->dims()[1], dY->numel() / (dY->dims()[0] * dY->dims()[1])};
    int in_step = static_cast<int>(X->dims()[1]) / groups;
    int out_step = static_cast<int>(dY->dims()[1]) / groups;

    bool is_expand = IsExpand(filter_shape_vec, strides, paddings, dilations);
    Tensor col;
    Tensor col_matrix;
    if (is_expand) {
      col = ctx.AllocateTmpTensor<T, DeviceContext>(col_shape, dev_ctx);
      col_matrix.ShareDataWith(col);
      col_matrix.Resize(col_matrix_shape);
    }

    math::SetConstant<DeviceContext, T> set_zero;
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);

    // dx convolution double grad:  gemm + col2im(col2vol)
    // dx = ddw * dy  ==> dx(N, Cin, H, W), ddw(Cout, Cin, kh, kw), dy(N, Cout,
    // oH, oW)
    if (dX && ddW_in) {
      Tensor ddW;
      ddW.ShareDataWith(*ddW_in).Resize(filter_matrix_shape);

      dX->mutable_data<T>(ctx.GetPlace());
      // if is_expand is false, the operation of set_zero is unnecessary
      // because math::matmul will reset dx
      if (is_expand) {
        set_zero(dev_ctx, dX, static_cast<T>(0));
      }
      math::Col2VolFunctor<DeviceContext, T> col2vol;
      math::Col2ImFunctor<math::ColFormat::kCFO, DeviceContext, T> col2im;

      for (int i = 0; i < batch_size; i++) {
        Tensor dy_batch = dY->Slice(i, i + 1).Resize(output_matrix_shape);
        Tensor dx_batch = dX->Slice(i, i + 1).Resize(input_shape);
        for (int g = 0; g < groups; g++) {
          // gemm
          Tensor dy_slice = dy_batch.Slice(g * out_step, (g + 1) * out_step);
          Tensor ddw_slice = ddW.Slice(g * out_step, (g + 1) * out_step);
          Tensor dx_slice = dx_batch.Slice(g * in_step, (g + 1) * in_step);
          if (!is_expand) {
            col_matrix.ShareDataWith(dx_slice);
            col_matrix.Resize(col_matrix_shape);
          }
          blas.MatMul(ddw_slice, true, dy_slice, false, T(1.0), &col_matrix,
                      T(0.0));

          if (is_expand && data_dim == 2U) {
            col2im(dev_ctx, col, dilations, strides,
                   std::vector<int>{paddings[0], paddings[1], paddings[0],
                                    paddings[1]},
                   &dx_slice);
          } else if (is_expand && data_dim == 3U) {
            col2vol(dev_ctx, col, dilations, strides, paddings, &dx_slice);
          }
        }
      }
    }

    // dw = ddx * dy  ==> dw(Cout, Cin, kh, kw), ddx(N, Cin, H, W), dy(N, Cout,
    // oH, oW)
    // dw convolution double grad:  im2col(vol2col) + gemm
    if (dW) {
      dW->mutable_data<T>(ctx.GetPlace());
      set_zero(dev_ctx, dW, static_cast<T>(0));
      Tensor dW_arr = *dW;
      dW_arr.Resize(filter_matrix_shape);
      math::Im2ColFunctor<math::ColFormat::kCFO, DeviceContext, T> im2col;
      math::Vol2ColFunctor<DeviceContext, T> vol2col;
      for (int i = 0; i < batch_size; ++i) {
        Tensor dy_batch = dY->Slice(i, i + 1).Resize(output_matrix_shape);
        Tensor ddx_batch = ddX->Slice(i, i + 1).Resize(input_shape);
        for (int g = 0; g < groups; ++g) {
          // im2col
          Tensor dy_slice = dy_batch.Slice(g * out_step, (g + 1) * out_step);
          Tensor ddx_slice = ddx_batch.Slice(g * in_step, (g + 1) * in_step);
          if (!is_expand) {
            col.ShareDataWith(ddx_slice);
            col_matrix.ShareDataWith(col);
            col_matrix.Resize(col_matrix_shape);
          } else if (data_dim == 2U) {
            im2col(dev_ctx, ddx_slice, dilations, strides,
                   std::vector<int>{paddings[0], paddings[1], paddings[0],
                                    paddings[1]},
                   &col);
          } else if (data_dim == 3U) {
            vol2col(dev_ctx, ddx_slice, dilations, strides, paddings, &col);
          }

          Tensor dw_slice = dW_arr.Slice(g * out_step, (g + 1) * out_step);
          blas.MatMul(dy_slice, false, col_matrix, true, T(1.0), &dw_slice,
                      T(1.0));
        }
      }
    }

    // ddy = w * ddx + x * ddw ==> ddy(N, Cout, oH, oW), x/ddx(N, Cin, H, W),
    // w/ddw(Cout, Cin, kh, kw)
    // ddy convolution double grad: im2col(vol2col) + gemm
    if (ddY) {
      ddY->mutable_data<T>(ctx.GetPlace());
      set_zero(dev_ctx, ddY, static_cast<T>(0));
      math::Im2ColFunctor<math::ColFormat::kCFO, DeviceContext, T> im2col;
      math::Vol2ColFunctor<DeviceContext, T> vol2col;
      for (int i = 0; i < batch_size; ++i) {
        Tensor ddx_batch = ddX->Slice(i, i + 1).Resize(input_shape);
        Tensor x_batch = X->Slice(i, i + 1).Resize(input_shape);
        Tensor ddy_batch = ddY->Slice(i, i + 1).Resize(output_matrix_shape);
        for (int g = 0; g < groups; ++g) {
          Tensor x_slice = x_batch.Slice(g * in_step, (g + 1) * in_step);
          Tensor ddx_slice = ddx_batch.Slice(g * in_step, (g + 1) * in_step);
          if (!is_expand) {
            col.ShareDataWith(ddx_slice);
            col_matrix.ShareDataWith(col);
            col_matrix.Resize(col_matrix_shape);
          } else if (data_dim == 2U) {
            // im2col
            im2col(dev_ctx, ddx_slice, dilations, strides,
                   std::vector<int>{paddings[0], paddings[1], paddings[0],
                                    paddings[1]},
                   &col);
          } else if (data_dim == 3U) {
            // vol2col
            vol2col(dev_ctx, ddx_slice, dilations, strides, paddings, &col);
          }

          // gemm
          Tensor ddy_slice = ddy_batch.Slice(g * out_step, (g + 1) * out_step);
          Tensor w_slice = W.Slice(g * out_step, (g + 1) * out_step);
          blas.MatMul(w_slice, false, col_matrix, false, T(1.0), &ddy_slice,
                      T(0.0));

          if (ddW_in) {
            Tensor ddW;
            ddW.ShareDataWith(*ddW_in).Resize(filter_matrix_shape);

            if (!is_expand) {
              col.ShareDataWith(x_slice);
              col_matrix.ShareDataWith(col);
              col_matrix.Resize(col_matrix_shape);
            } else if (data_dim == 2U) {
              // im2col
              im2col(dev_ctx, x_slice, dilations, strides,
                     std::vector<int>{paddings[0], paddings[1], paddings[0],
                                      paddings[1]},
                     &col);
            } else if (data_dim == 3U) {
              // vol2col
              vol2col(dev_ctx, x_slice, dilations, strides, paddings, &col);
            }

            // gemm
            Tensor ddw_slice = ddW.Slice(g * out_step, (g + 1) * out_step);
            blas.MatMul(ddw_slice, false, col_matrix, false, T(1.0), &ddy_slice,
                        T(1.0));
          }
        }
      }
    }
  }
};

template <typename DeviceContext, typename T>
class DepthwiseConvKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    Tensor filter = *context.Input<Tensor>("Filter");
    Tensor* output = context.Output<Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());

    PADDLE_ENFORCE_EQ(
        output->dims()[1] % input->dims()[1], 0,
        "The output channels must be a multiple of the input channels");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    bool fuse_relu = context.Attr<bool>("fuse_relu_before_depthwise_conv");
    auto& dev_ctx = context.template device_context<DeviceContext>();

    if (fuse_relu) {
      math::DepthwiseConvFunctor<DeviceContext, T, true> depthwiseConv;
      depthwiseConv(dev_ctx, *input, filter, strides, paddings, dilations,
                    output);
    } else {
      math::DepthwiseConvFunctor<DeviceContext, T, false> depthwiseConv;
      depthwiseConv(dev_ctx, *input, filter, strides, paddings, dilations,
                    output);
    }
  }
};

template <typename DeviceContext, typename T>
class DepthwiseConvGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    const Tensor* output_grad =
        context.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad =
        context.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad =
        context.Output<Tensor>(framework::GradVarName("Filter"));
    Tensor filter = *context.Input<Tensor>("Filter");

    if (!input_grad && !filter_grad) return;

    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    bool fuse_relu = context.Attr<bool>("fuse_relu_before_depthwise_conv");

    math::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();

    if (input_grad) {
      input_grad->mutable_data<T>(context.GetPlace());
      set_zero(dev_ctx, input_grad, static_cast<T>(0));

      if (fuse_relu) {
        math::DepthwiseConvInputGradFunctor<DeviceContext, T, true>
            depthwiseConvInputGrad;
        depthwiseConvInputGrad(dev_ctx, *input, filter, *output_grad, strides,
                               paddings, dilations, input_grad);
      } else {
        math::DepthwiseConvInputGradFunctor<DeviceContext, T, false>
            depthwiseConvInputGrad;
        depthwiseConvInputGrad(dev_ctx, *input, filter, *output_grad, strides,
                               paddings, dilations, input_grad);
      }
    }

    if (filter_grad) {
      filter_grad->mutable_data<T>(context.GetPlace());
      set_zero(dev_ctx, filter_grad, static_cast<T>(0));
      if (fuse_relu) {
        math::DepthwiseConvFilterGradFunctor<DeviceContext, T, true>
            depthwiseConvFilterGrad;
        depthwiseConvFilterGrad(dev_ctx, *input, *output_grad, strides,
                                paddings, dilations, filter_grad);
      } else {
        math::DepthwiseConvFilterGradFunctor<DeviceContext, T, false>
            depthwiseConvFilterGrad;
        depthwiseConvFilterGrad(dev_ctx, *input, *output_grad, strides,
                                paddings, dilations, filter_grad);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
