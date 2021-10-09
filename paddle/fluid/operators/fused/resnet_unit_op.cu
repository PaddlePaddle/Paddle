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

#include "paddle/fluid/operators/fused/cudnn_bn_stats_finalize.cu.h"
#include "paddle/fluid/operators/fused/cudnn_norm_conv.cu.h"
#include "paddle/fluid/operators/fused/cudnn_scale_bias_add_relu.cu.h"
#include "paddle/fluid/operators/fused/resnet_unit_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class ResNetUnitKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("It must use CUDAPlace."));

    // input x
    const Tensor *input_x = ctx.Input<Tensor>("X");
    const Tensor *filter_x = ctx.Input<Tensor>("FilterX");
    const Tensor *scale_x = ctx.Input<Tensor>("ScaleX");
    const Tensor *bias_x = ctx.Input<Tensor>("BiasX");
    // norm conv
    Tensor *conv_out_x = ctx.Output<Tensor>("ConvX");
    // bn finalize
    Tensor *saved_mean_x = ctx.Output<Tensor>("SavedMeanX");
    Tensor *saved_invstd_x = ctx.Output<Tensor>("SavedInvstdX");
    Tensor *running_mean_x = ctx.Output<Tensor>("RunningMeanX");
    Tensor *running_var_x = ctx.Output<Tensor>("RunningVarX");
    // sbar
    Tensor *output = ctx.Output<Tensor>("Y");
    Tensor *bitmask = ctx.Output<Tensor>("BitMask");
    // attrs
    int pad = ctx.Attr<int>("pad");
    int stride = ctx.Attr<int>("stride");
    int stride_z = ctx.Attr<int>("stride_z");
    int dilate = ctx.Attr<int>("dilate");
    int group = ctx.Attr<int>("group");
    double eps = static_cast<double>(ctx.Attr<float>("epsilon"));
    double momentum = static_cast<double>(ctx.Attr<float>("momentum"));
    bool has_shortcut = ctx.Attr<bool>("has_shortcut");
    bool fused_add = ctx.Attr<bool>("fused_add");
    bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    std::string act_type = ctx.Attr<std::string>("act_type");

    auto input_x_shape = framework::vectorize<int>(input_x->dims());
    auto filter_x_shape = framework::vectorize<int>(filter_x->dims());
    auto param_dims = scale_x->dims();
    auto param_shape = framework::vectorize<int>(scale_x->dims());
    auto output_shape = framework::vectorize<int>(output->dims());
    auto bitmask_shape = framework::vectorize<int>(bitmask->dims());
    int output_channel = filter_x_shape[0];
    int64_t ele_count =
        std::accumulate(output_shape.begin(), output_shape.end(), 1,
                        std::multiplies<int>()) /
        output_channel;

    auto place = ctx.GetPlace();
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    // 1. Conv
    T *input_x_ptr = const_cast<T *>(input_x->data<T>());
    T *filter_x_ptr = const_cast<T *>(filter_x->data<T>());
    T *conv_out_x_ptr = conv_out_x->mutable_data<T>(place);

    Tensor sum_x;
    Tensor sum_of_squares_x;
    float *sum_x_ptr = sum_x.mutable_data<float>(param_dims, place);
    float *sum_of_squares_x_ptr =
        sum_of_squares_x.mutable_data<float>(param_dims, place);

    CudnnNormConvolution<T> conv_x_op(dev_ctx, input_x_shape, filter_x_shape,
                                      output_shape, pad, stride, dilate, group);
    conv_x_op.Forward(dev_ctx, input_x_ptr, filter_x_ptr, conv_out_x_ptr,
                      sum_x_ptr, sum_of_squares_x_ptr);

    // 2. BN
    float *scale_x_ptr = const_cast<float *>(scale_x->data<float>());
    float *bias_x_ptr = const_cast<float *>(bias_x->data<float>());
    float *saved_mean_x_ptr = saved_mean_x->mutable_data<float>(place);
    float *saved_invstd_x_ptr = saved_invstd_x->mutable_data<float>(place);
    float *running_mean_x_ptr = running_mean_x->mutable_data<float>(place);
    float *running_var_x_ptr = running_var_x->mutable_data<float>(place);

    Tensor equiv_scale_x;
    Tensor equiv_bias_x;
    T *equiv_scale_x_ptr = equiv_scale_x.mutable_data<T>(param_dims, place);
    T *equiv_bias_x_ptr = equiv_bias_x.mutable_data<T>(param_dims, place);

    CudnnBNStatsFinalize<T> bn_x_op(dev_ctx, param_shape);
    bn_x_op.Forward(dev_ctx, sum_x_ptr, sum_of_squares_x_ptr, scale_x_ptr,
                    bias_x_ptr, saved_mean_x_ptr, saved_invstd_x_ptr,
                    running_mean_x_ptr, running_var_x_ptr, equiv_scale_x_ptr,
                    equiv_bias_x_ptr, eps, momentum, ele_count,
                    !use_global_stats);

    // 3. scale + bias + add + relu
    T *output_ptr = output->mutable_data<T>(place);
    int32_t *bitmask_ptr = bitmask->mutable_data<int32_t>(place);

    CudnnScaleBiasAddRelu<T> sbar_op(dev_ctx, act_type, fused_add, has_shortcut,
                                     output_shape, param_shape, bitmask_shape);
    if (has_shortcut) {
      // input z
      const Tensor *input_z = ctx.Input<Tensor>("Z");
      const Tensor *filter_z = ctx.Input<Tensor>("FilterZ");
      const Tensor *scale_z = ctx.Input<Tensor>("ScaleZ");
      const Tensor *bias_z = ctx.Input<Tensor>("BiasZ");
      // norm conv
      Tensor *conv_out_z = ctx.Output<Tensor>("ConvZ");
      // bn finalize
      Tensor *saved_mean_z = ctx.Output<Tensor>("SavedMeanZ");
      Tensor *saved_invstd_z = ctx.Output<Tensor>("SavedInvstdZ");
      Tensor *running_mean_z = ctx.Output<Tensor>("RunningMeanZ");
      Tensor *running_var_z = ctx.Output<Tensor>("RunningVarZ");

      auto input_z_shape = framework::vectorize<int>(input_z->dims());
      auto filter_z_shape = framework::vectorize<int>(filter_z->dims());

      // 3.1 Conv for second input
      T *input_z_ptr = const_cast<T *>(input_z->data<T>());
      T *filter_z_ptr = const_cast<T *>(filter_z->data<T>());
      T *conv_out_z_ptr = conv_out_z->mutable_data<T>(place);

      Tensor sum_z;
      Tensor sum_of_squares_z;
      float *sum_z_ptr = sum_z.mutable_data<float>(param_dims, place);
      float *sum_of_squares_z_ptr =
          sum_of_squares_z.mutable_data<float>(param_dims, place);

      CudnnNormConvolution<T> conv_z_op(dev_ctx, input_z_shape, filter_z_shape,
                                        output_shape, pad, stride_z, dilate,
                                        group);
      conv_z_op.Forward(dev_ctx, input_z_ptr, filter_z_ptr, conv_out_z_ptr,
                        sum_z_ptr, sum_of_squares_z_ptr);

      // 3.2 BN for second input
      float *scale_z_ptr = const_cast<float *>(scale_z->data<float>());
      float *bias_z_ptr = const_cast<float *>(bias_z->data<float>());
      float *saved_mean_z_ptr = saved_mean_z->mutable_data<float>(place);
      float *saved_invstd_z_ptr = saved_invstd_z->mutable_data<float>(place);
      float *running_mean_z_ptr = running_mean_z->mutable_data<float>(place);
      float *running_var_z_ptr = running_var_z->mutable_data<float>(place);

      Tensor equiv_scale_z;
      Tensor equiv_bias_z;
      T *equiv_scale_z_ptr = equiv_scale_z.mutable_data<T>(param_dims, place);
      T *equiv_bias_z_ptr = equiv_bias_z.mutable_data<T>(param_dims, place);

      CudnnBNStatsFinalize<T> bn_z_op(dev_ctx, param_shape);
      bn_z_op.Forward(dev_ctx, sum_z_ptr, sum_of_squares_z_ptr, scale_z_ptr,
                      bias_z_ptr, saved_mean_z_ptr, saved_invstd_z_ptr,
                      running_mean_z_ptr, running_var_z_ptr, equiv_scale_z_ptr,
                      equiv_bias_z_ptr, eps, momentum, ele_count,
                      !use_global_stats);
      // 3.3 sbar
      sbar_op.Forward(dev_ctx, conv_out_x_ptr, equiv_scale_x_ptr,
                      equiv_bias_x_ptr, output_ptr, bitmask_ptr, conv_out_z_ptr,
                      equiv_scale_z_ptr, equiv_bias_z_ptr);
    } else {
      T *input_z_ptr = nullptr;
      if (fused_add) {
        // input z
        const Tensor *input_z = ctx.Input<Tensor>("Z");
        input_z_ptr = const_cast<T *>(input_z->data<T>());
      }
      sbar_op.Forward(dev_ctx, conv_out_x_ptr, equiv_scale_x_ptr,
                      equiv_bias_x_ptr, output_ptr, bitmask_ptr, input_z_ptr);
    }
  }
};

template <typename T>
class ResNetUnitGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("It must use CUDAPlace."));

    const Tensor *y_grad = ctx.Input<Tensor>(framework::GradVarName("Y"));

    const Tensor *x = ctx.Input<Tensor>("X");
    const Tensor *filter_x = ctx.Input<Tensor>("FilterX");
    const Tensor *scale_x = ctx.Input<Tensor>("ScaleX");
    const Tensor *bias_x = ctx.Input<Tensor>("BiasX");
    const Tensor *saved_mean_x = ctx.Input<Tensor>("SavedMeanX");
    const Tensor *saved_invstd_x = ctx.Input<Tensor>("SavedInvstdX");

    const Tensor *conv_out_x = ctx.Input<Tensor>("ConvX");
    const Tensor *output = ctx.Input<Tensor>("Y");
    const Tensor *bitmask = ctx.Input<Tensor>("BitMask");

    Tensor *x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    Tensor *filter_x_grad =
        ctx.Output<Tensor>(framework::GradVarName("FilterX"));
    Tensor *scale_x_grad = ctx.Output<Tensor>(framework::GradVarName("ScaleX"));
    Tensor *bias_x_grad = ctx.Output<Tensor>(framework::GradVarName("BiasX"));

    // attrs
    int pad = ctx.Attr<int>("pad");
    int stride = ctx.Attr<int>("stride");
    int stride_z = ctx.Attr<int>("stride_z");
    int dilate = ctx.Attr<int>("dilate");
    int group = ctx.Attr<int>("group");
    double eps = static_cast<double>(ctx.Attr<float>("epsilon"));
    double momentum = static_cast<double>(ctx.Attr<float>("momentum"));
    bool has_shortcut = ctx.Attr<bool>("has_shortcut");
    bool fused_add = ctx.Attr<bool>("fused_add");
    bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    std::string act_type = ctx.Attr<std::string>("act_type");

    auto x_shape = framework::vectorize<int>(x->dims());
    auto filter_x_shape = framework::vectorize<int>(filter_x->dims());
    auto param_shape = framework::vectorize<int>(scale_x->dims());
    auto output_shape = framework::vectorize<int>(output->dims());
    auto bitmask_shape = framework::vectorize<int>(bitmask->dims());

    auto place = ctx.GetPlace();
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    // 1. bn add relu backward, get conv_out_x_grad, scale_x_grad, bias_x_grad
    T *y_grad_ptr = const_cast<T *>(y_grad->data<T>());
    T *conv_out_x_ptr = const_cast<T *>(conv_out_x->data<T>());
    float *scale_x_ptr = const_cast<float *>(scale_x->data<float>());
    float *bias_x_ptr = const_cast<float *>(bias_x->data<float>());
    float *saved_mean_x_ptr = const_cast<float *>(saved_mean_x->data<float>());
    float *saved_invstd_x_ptr =
        const_cast<float *>(saved_invstd_x->data<float>());
    int32_t *bitmask_ptr = const_cast<int32_t *>(bitmask->data<int32_t>());
    float *scale_x_grad_ptr =
        scale_x_grad ? scale_x_grad->mutable_data<float>(place) : nullptr;
    float *bias_x_grad_ptr =
        bias_x_grad ? bias_x_grad->mutable_data<float>(place) : nullptr;

    Tensor conv_out_x_grad;
    T *conv_out_x_grad_ptr =
        conv_out_x_grad.mutable_data<T>(conv_out_x->dims(), place);

    CudnnScaleBiasAddRelu<T> sbar_x_op(dev_ctx, act_type, fused_add,
                                       has_shortcut, output_shape, param_shape,
                                       bitmask_shape);
    if (has_shortcut) {
      //       X                   Z
      //       |                   |
      //    NormConv            NormConv
      //       |                   |
      // BNStatsFinalize    BNStatsFinalize
      //       \                   /
      //          ScaleBiasAddRelu
      //                  |
      //                  Y
      const Tensor *z = ctx.Input<Tensor>("Z");
      const Tensor *filter_z = ctx.Input<Tensor>("FilterZ");
      const Tensor *scale_z = ctx.Input<Tensor>("ScaleZ");
      const Tensor *bias_z = ctx.Input<Tensor>("BiasZ");
      const Tensor *saved_mean_z = ctx.Input<Tensor>("SavedMeanZ");
      const Tensor *saved_invstd_z = ctx.Input<Tensor>("SavedInvstdZ");
      const Tensor *conv_out_z = ctx.Input<Tensor>("ConvZ");

      Tensor *z_grad = ctx.Output<Tensor>(framework::GradVarName("Z"));
      Tensor *filter_z_grad =
          ctx.Output<Tensor>(framework::GradVarName("FilterZ"));
      Tensor *scale_z_grad =
          ctx.Output<Tensor>(framework::GradVarName("ScaleZ"));
      Tensor *bias_z_grad = ctx.Output<Tensor>(framework::GradVarName("BiasZ"));

      // 1.1 Backward of BN + Add (+ Relu) backward for x, get conv_out_x_grad,
      // dscale_x, dbias_x and
      // temp grad for z
      Tensor z_grad_temp;
      T *z_grad_temp_ptr =
          z_grad_temp.mutable_data<T>(conv_out_z->dims(), place);
      sbar_x_op.Backward(dev_ctx, y_grad_ptr, conv_out_x_ptr, scale_x_ptr,
                         bias_x_ptr, saved_mean_x_ptr, saved_invstd_x_ptr,
                         bitmask_ptr, conv_out_x_grad_ptr, z_grad_temp_ptr,
                         scale_x_grad_ptr, bias_x_grad_ptr, eps);

      // 1.2 bn backward for z, get conv_out_z_grad, dscale_z, dbias_z
      T *conv_out_z_ptr = const_cast<T *>(conv_out_z->data<T>());
      float *scale_z_ptr = const_cast<float *>(scale_z->data<float>());
      float *bias_z_ptr = const_cast<float *>(bias_z->data<float>());
      float *saved_mean_z_ptr =
          const_cast<float *>(saved_mean_z->data<float>());
      float *saved_invstd_z_ptr =
          const_cast<float *>(saved_invstd_z->data<float>());
      float *scale_z_grad_ptr = scale_z_grad->mutable_data<float>(place);
      float *bias_z_grad_ptr = bias_z_grad->mutable_data<float>(place);

      Tensor conv_out_z_grad;
      T *conv_out_z_grad_ptr =
          conv_out_z_grad.mutable_data<T>(conv_out_z->dims(), place);

      CudnnScaleBiasAddRelu<T> sbar_z_op(
          dev_ctx, "", false, false, output_shape, param_shape, bitmask_shape);
      sbar_z_op.Backward(dev_ctx, z_grad_temp_ptr, conv_out_z_ptr, scale_z_ptr,
                         bias_z_ptr, saved_mean_z_ptr, saved_invstd_z_ptr,
                         nullptr, conv_out_z_grad_ptr, nullptr,
                         scale_z_grad_ptr, bias_z_grad_ptr, eps);

      // 1.3 conv backward for z, get dinput_z and filter_z_grad
      T *z_ptr = const_cast<T *>(z->data<T>());
      T *filter_z_ptr = const_cast<T *>(filter_z->data<T>());
      T *z_grad_ptr = z_grad ? z_grad->mutable_data<T>(place) : nullptr;
      T *filter_z_grad_ptr =
          filter_z_grad ? filter_z_grad->mutable_data<T>(place) : nullptr;

      auto z_shape = framework::vectorize<int>(z->dims());
      auto filter_z_shape = framework::vectorize<int>(filter_z->dims());
      CudnnNormConvolutionGrad<T> conv_z_op(dev_ctx, z_shape, filter_z_shape,
                                            output_shape, pad, stride_z, dilate,
                                            group);
      conv_z_op.Backward(dev_ctx, z_ptr, conv_out_z_grad_ptr, filter_z_ptr,
                         z_grad_ptr, filter_z_grad_ptr);
    } else {
      T *z_grad_ptr = nullptr;
      // 1.1 backward of BN (+ Add + Relu) for x, get conv_out_x_grad, dscale_x,
      // dbias_x
      // (and z_grad)
      if (fused_add) {
        Tensor *z_grad = ctx.Output<Tensor>(framework::GradVarName("Z"));
        z_grad_ptr = z_grad->mutable_data<T>(place);
      }
      sbar_x_op.Backward(dev_ctx, y_grad_ptr, conv_out_x_ptr, scale_x_ptr,
                         bias_x_ptr, saved_mean_x_ptr, saved_invstd_x_ptr,
                         bitmask_ptr, conv_out_x_grad_ptr, z_grad_ptr,
                         scale_x_grad_ptr, bias_x_grad_ptr, eps);
    }

    // 2. conv backward for x, get x_grad and filter_x_grad
    T *x_ptr = const_cast<T *>(x->data<T>());
    T *filter_x_ptr = const_cast<T *>(filter_x->data<T>());
    T *x_grad_ptr = x_grad ? x_grad->mutable_data<T>(place) : nullptr;
    T *filter_x_grad_ptr =
        filter_x_grad ? filter_x_grad->mutable_data<T>(place) : nullptr;

    bool use_addto = ctx.Attr<bool>("use_addto");
    CudnnNormConvolutionGrad<T> conv_x_op(dev_ctx, x_shape, filter_x_shape,
                                          output_shape, pad, stride, dilate,
                                          group);
    conv_x_op.Backward(dev_ctx, x_ptr, conv_out_x_grad_ptr, filter_x_ptr,
                       x_grad_ptr, filter_x_grad_ptr, use_addto);
  }
};

#undef MALLOC_AND_GET_PTR
}  // namespace operators
}  // namespace paddle

#if CUDNN_VERSION >= 8000
namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    resnet_unit, ops::ResNetUnitKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    resnet_unit_grad,
    ops::ResNetUnitGradKernel<plat::CUDADeviceContext, plat::float16>);
#endif
