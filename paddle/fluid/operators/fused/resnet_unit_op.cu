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
#define MALLOC_AND_GET_PTR(TR, Dtype, Place) \
  Dtype *TR##_ptr = TR->mutable_data<Dtype>(Place);

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
    Tensor *sum_x = ctx.Output<Tensor>("SumX");
    Tensor *sum_of_squares_x = ctx.Output<Tensor>("SqSumX");
    // bn finalize
    Tensor *saved_mean_x = ctx.Output<Tensor>("SavedMeanX");
    Tensor *saved_invstd_x = ctx.Output<Tensor>("SavedInvstdX");
    Tensor *running_mean_x = ctx.Output<Tensor>("RunningMeanX");
    Tensor *running_var_x = ctx.Output<Tensor>("RunningVarX");
    Tensor *equiv_scale_x = ctx.Output<Tensor>("EqScaleX");
    Tensor *equiv_bias_x = ctx.Output<Tensor>("EqBiasX");
    // sbar
    Tensor *output = ctx.Output<Tensor>("Y");
    Tensor *bitmask = ctx.Output<Tensor>("BitMask");
    // attrs
    int pad = ctx.Attr<int>("pad");
    int stride = ctx.Attr<int>("stride");
    int dilate = ctx.Attr<int>("dilate");
    int group = ctx.Attr<int>("group");
    double eps = static_cast<double>(ctx.Attr<float>("epsilon"));
    double momentum = static_cast<double>(ctx.Attr<float>("momentum"));
    bool has_shortcut = ctx.Attr<bool>("has_shortcut");
    bool fused_add = ctx.Attr<bool>("fused_add");
    bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    std::string act_type = ctx.Attr<std::string>("act_type");

    // tensor shape
    auto input_x_shape = framework::vectorize<int>(input_x->dims());
    auto filter_x_shape = framework::vectorize<int>(filter_x->dims());
    auto param_shape = framework::vectorize<int>(scale_x->dims());
    auto output_shape = framework::vectorize<int>(output->dims());
    auto bitmask_shape = framework::vectorize<int>(bitmask->dims());
    auto place = input_x->place();
    int output_channel = filter_x_shape[0];
    int64_t ele_count =
        std::accumulate(output_shape.begin(), output_shape.end(), 1,
                        std::multiplies<int>()) /
        output_channel;

    MALLOC_AND_GET_PTR(conv_out_x, T, place)
    MALLOC_AND_GET_PTR(sum_x, float, place)
    MALLOC_AND_GET_PTR(sum_of_squares_x, float, place)
    MALLOC_AND_GET_PTR(saved_mean_x, float, place)
    MALLOC_AND_GET_PTR(saved_invstd_x, float, place)
    MALLOC_AND_GET_PTR(running_mean_x, float, place)
    MALLOC_AND_GET_PTR(running_var_x, float, place)
    MALLOC_AND_GET_PTR(equiv_scale_x, T, place)
    MALLOC_AND_GET_PTR(equiv_bias_x, T, place)
    MALLOC_AND_GET_PTR(output, T, place)
    MALLOC_AND_GET_PTR(bitmask, int32_t, place)

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    // 1. Conv
    T *input_x_ptr = const_cast<T *>(input_x->data<T>());
    T *filter_x_ptr = const_cast<T *>(filter_x->data<T>());
    std::shared_ptr<CudnnNormConvolutionOp<T>> conv_x_op(
        new CudnnNormConvolutionOp<T>());
    conv_x_op->Init(dev_ctx, input_x_shape, filter_x_shape, output_shape, pad,
                    stride, dilate, group);
    conv_x_op->Forward(dev_ctx, input_x_ptr, filter_x_ptr, conv_out_x_ptr,
                       sum_x_ptr, sum_of_squares_x_ptr);
    // 2. BN
    float *scale_x_ptr = const_cast<float *>(scale_x->data<float>());
    float *bias_x_ptr = const_cast<float *>(bias_x->data<float>());
    std::shared_ptr<CudnnBNStatsFinalizeOp<T>> bn_x_op(
        new CudnnBNStatsFinalizeOp<T>());
    bn_x_op->Init(dev_ctx, param_shape);
    bn_x_op->Forward(dev_ctx, sum_x_ptr, sum_of_squares_x_ptr, scale_x_ptr,
                     bias_x_ptr, saved_mean_x_ptr, saved_invstd_x_ptr,
                     running_mean_x_ptr, running_var_x_ptr, equiv_scale_x_ptr,
                     equiv_bias_x_ptr, eps, momentum, ele_count,
                     use_global_stats);

    // 3. scale + bias + add + relu
    std::shared_ptr<CudnnScaleBiasAddReluOp<T>> sbar_op(
        new CudnnScaleBiasAddReluOp<T>(fused_add, has_shortcut));
    if (has_shortcut) {
      // input z
      const Tensor *input_z = ctx.Input<Tensor>("Z");
      const Tensor *filter_z = ctx.Input<Tensor>("FilterZ");
      const Tensor *scale_z = ctx.Input<Tensor>("ScaleZ");
      const Tensor *bias_z = ctx.Input<Tensor>("BiasZ");
      // norm conv
      Tensor *conv_out_z = ctx.Output<Tensor>("ConvZ");
      Tensor *sum_z = ctx.Output<Tensor>("SumZ");
      Tensor *sum_of_squares_z = ctx.Output<Tensor>("SqSumZ");
      // bn finalize
      Tensor *saved_mean_z = ctx.Output<Tensor>("SavedMeanZ");
      Tensor *saved_invstd_z = ctx.Output<Tensor>("SavedInvstdZ");
      Tensor *running_mean_z = ctx.Output<Tensor>("RunningMeanZ");
      Tensor *running_var_z = ctx.Output<Tensor>("RunningVarZ");
      Tensor *equiv_scale_z = ctx.Output<Tensor>("EqScaleZ");
      Tensor *equiv_bias_z = ctx.Output<Tensor>("EqBiasZ");
      MALLOC_AND_GET_PTR(conv_out_z, T, place)
      MALLOC_AND_GET_PTR(sum_z, float, place)
      MALLOC_AND_GET_PTR(sum_of_squares_z, float, place)
      MALLOC_AND_GET_PTR(saved_mean_z, float, place)
      MALLOC_AND_GET_PTR(saved_invstd_z, float, place)
      MALLOC_AND_GET_PTR(running_mean_z, float, place)
      MALLOC_AND_GET_PTR(running_var_z, float, place)
      MALLOC_AND_GET_PTR(equiv_scale_z, T, place)
      MALLOC_AND_GET_PTR(equiv_bias_z, T, place)
      auto input_z_shape = framework::vectorize<int>(input_z->dims());
      auto filter_z_shape = framework::vectorize<int>(filter_z->dims());
      // 3.1 Conv for second input
      T *input_z_ptr = const_cast<T *>(input_z->data<T>());
      T *filter_z_ptr = const_cast<T *>(filter_z->data<T>());
      std::shared_ptr<CudnnNormConvolutionOp<T>> conv_z_op(
          new CudnnNormConvolutionOp<T>());
      conv_z_op->Init(dev_ctx, input_z_shape, filter_z_shape, output_shape, pad,
                      stride, dilate, group);
      conv_z_op->Forward(dev_ctx, input_z_ptr, filter_z_ptr, conv_out_z_ptr,
                         sum_z_ptr, sum_of_squares_z_ptr);
      // 3.2 BN for second input
      float *scale_z_ptr = const_cast<float *>(scale_z->data<float>());
      float *bias_z_ptr = const_cast<float *>(bias_z->data<float>());
      std::shared_ptr<CudnnBNStatsFinalizeOp<T>> bn_z_op(
          new CudnnBNStatsFinalizeOp<T>());
      bn_z_op->Init(dev_ctx, param_shape);
      bn_z_op->Forward(dev_ctx, sum_z_ptr, sum_of_squares_z_ptr, scale_z_ptr,
                       bias_z_ptr, saved_mean_z_ptr, saved_invstd_z_ptr,
                       running_mean_z_ptr, running_var_z_ptr, equiv_scale_z_ptr,
                       equiv_bias_z_ptr, eps, momentum, ele_count,
                       use_global_stats);
      // 3.3 sbar
      sbar_op->Init(dev_ctx, act_type, output_shape, bitmask_shape,
                    output_shape, param_shape, output_shape);
      sbar_op->Forward(dev_ctx, conv_out_x_ptr, equiv_scale_x_ptr,
                       equiv_bias_x_ptr, output_ptr, bitmask_ptr,
                       conv_out_z_ptr, equiv_scale_z_ptr, equiv_bias_z_ptr);
    } else {
      if (fused_add) {
        // input z
        const Tensor *input_z = ctx.Input<Tensor>("Z");
        T *input_z_ptr = const_cast<T *>(input_z->data<T>());
        auto input_z_shape = framework::vectorize<int>(input_z->dims());
        sbar_op->Init(dev_ctx, act_type, output_shape, bitmask_shape,
                      output_shape, param_shape, input_z_shape);
        sbar_op->Forward(dev_ctx, conv_out_x_ptr, equiv_scale_x_ptr,
                         equiv_bias_x_ptr, output_ptr, bitmask_ptr,
                         const_cast<T *>(input_z_ptr));
      } else {
        sbar_op->Init(dev_ctx, act_type, output_shape, bitmask_shape,
                      output_shape, param_shape);
        sbar_op->Forward(dev_ctx, conv_out_x_ptr, equiv_scale_x_ptr,
                         equiv_bias_x_ptr, output_ptr, bitmask_ptr);
      }
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

    // backward input
    const Tensor *doutput = ctx.Input<Tensor>(framework::GradVarName("Y"));
    // forward input (backward input)
    const Tensor *input_x = ctx.Input<Tensor>("X");
    const Tensor *filter_x = ctx.Input<Tensor>("FilterX");
    const Tensor *scale_x = ctx.Input<Tensor>("ScaleX");
    const Tensor *bias_x = ctx.Input<Tensor>("BiasX");
    const Tensor *saved_mean_x = ctx.Input<Tensor>("SavedMeanX");
    const Tensor *saved_invstd_x = ctx.Input<Tensor>("SavedInvstdX");
    // forward output (backward input)
    const Tensor *conv_out_x = ctx.Input<Tensor>("ConvX");
    const Tensor *output = ctx.Input<Tensor>("Y");
    const Tensor *bitmask = ctx.Input<Tensor>("BitMask");

    // backward output
    Tensor *dinput_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    Tensor *dfilter_x = ctx.Output<Tensor>(framework::GradVarName("FilterX"));
    Tensor *dscale_x = ctx.Output<Tensor>(framework::GradVarName("ScaleX"));
    Tensor *dbias_x = ctx.Output<Tensor>(framework::GradVarName("BiasX"));
    Tensor *dconv_out_x = ctx.Output<Tensor>(framework::GradVarName("ConvX"));

    // attrs
    int pad = ctx.Attr<int>("pad");
    int stride = ctx.Attr<int>("stride");
    int dilate = ctx.Attr<int>("dilate");
    int group = ctx.Attr<int>("group");
    double eps = static_cast<double>(ctx.Attr<float>("epsilon"));
    double momentum = static_cast<double>(ctx.Attr<float>("momentum"));
    bool has_shortcut = ctx.Attr<bool>("has_shortcut");
    bool fused_add = ctx.Attr<bool>("fused_add");
    bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    std::string act_type = ctx.Attr<std::string>("act_type");

    // tensor shape
    auto input_x_shape = framework::vectorize<int>(input_x->dims());
    auto filter_x_shape = framework::vectorize<int>(filter_x->dims());
    auto param_shape = framework::vectorize<int>(scale_x->dims());
    auto output_shape = framework::vectorize<int>(output->dims());
    auto bitmask_shape = framework::vectorize<int>(bitmask->dims());

    auto place = input_x->place();
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    // get ptr
    // input ptr
    T *doutput_ptr = const_cast<T *>(doutput->data<T>());
    T *input_x_ptr = const_cast<T *>(input_x->data<T>());
    T *filter_x_ptr = const_cast<T *>(filter_x->data<T>());
    float *scale_x_ptr = const_cast<float *>(scale_x->data<float>());
    float *bias_x_ptr = const_cast<float *>(bias_x->data<float>());
    float *saved_mean_x_ptr = const_cast<float *>(saved_mean_x->data<float>());
    float *saved_invstd_x_ptr =
        const_cast<float *>(saved_invstd_x->data<float>());
    T *conv_out_x_ptr = const_cast<T *>(conv_out_x->data<T>());
    T *output_ptr = const_cast<T *>(output->data<T>());
    // output ptr
    MALLOC_AND_GET_PTR(dinput_x, T, place)
    MALLOC_AND_GET_PTR(dfilter_x, T, place)
    MALLOC_AND_GET_PTR(dscale_x, float, place)
    MALLOC_AND_GET_PTR(dbias_x, float, place)
    MALLOC_AND_GET_PTR(dconv_out_x, T, place)

    // 1. bn add relu backward, get dconv_out_x, dscale_x, dbias_x
    std::shared_ptr<CudnnScaleBiasAddReluOp<T>> sbar_x_op(
        new CudnnScaleBiasAddReluOp<T>(fused_add, has_shortcut));
    if (has_shortcut) {
      // get dconv_out_z, dz, dscale_z and dbias_z
      // forward input (backward input)
      const Tensor *input_z = ctx.Input<Tensor>("Z");
      const Tensor *filter_z = ctx.Input<Tensor>("FilterZ");
      const Tensor *scale_z = ctx.Input<Tensor>("ScaleZ");
      const Tensor *bias_z = ctx.Input<Tensor>("BiasZ");
      const Tensor *saved_mean_z = ctx.Input<Tensor>("SavedMeanZ");
      const Tensor *saved_invstd_z = ctx.Input<Tensor>("SavedInvstdZ");
      // forward output (backward input)
      const Tensor *conv_out_z = ctx.Input<Tensor>("ConvZ");
      // backward output
      Tensor *dinput_z = ctx.Output<Tensor>(framework::GradVarName("Z"));
      Tensor *dfilter_z = ctx.Output<Tensor>(framework::GradVarName("FilterZ"));
      Tensor *dscale_z = ctx.Output<Tensor>(framework::GradVarName("ScaleZ"));
      Tensor *dbias_z = ctx.Output<Tensor>(framework::GradVarName("BiasZ"));
      Tensor *dconv_out_z = ctx.Output<Tensor>(framework::GradVarName("ConvZ"));

      // get ptr
      // input ptr
      T *input_z_ptr = const_cast<T *>(input_z->data<T>());
      T *filter_z_ptr = const_cast<T *>(filter_z->data<T>());
      float *scale_z_ptr = const_cast<float *>(scale_z->data<float>());
      float *bias_z_ptr = const_cast<float *>(bias_z->data<float>());
      float *saved_mean_z_ptr =
          const_cast<float *>(saved_mean_z->data<float>());
      float *saved_invstd_z_ptr =
          const_cast<float *>(saved_invstd_z->data<float>());
      T *conv_out_z_ptr = const_cast<T *>(conv_out_z->data<T>());
      // output ptr
      MALLOC_AND_GET_PTR(dinput_z, T, place)
      MALLOC_AND_GET_PTR(dfilter_z, T, place)
      MALLOC_AND_GET_PTR(dscale_z, float, place)
      MALLOC_AND_GET_PTR(dbias_z, float, place)
      MALLOC_AND_GET_PTR(dconv_out_z, T, place)
      // 1.1 bn add relu backward for x, get dconv_out_x, dscale_x, dbias_x and
      // temp grad for z
      Tensor dz_temp;
      T *dz_temp_ptr = dz_temp.mutable_data<T>(conv_out_z->dims(), place);
      sbar_x_op->Init(dev_ctx, act_type, output_shape, bitmask_shape,
                      output_shape, param_shape, output_shape);
      sbar_x_op->Backward(dev_ctx, doutput_ptr, conv_out_x_ptr, output_ptr,
                          scale_x_ptr, bias_x_ptr, saved_mean_x_ptr,
                          saved_invstd_x_ptr, dconv_out_x_ptr, dz_temp_ptr,
                          dscale_x_ptr, dbias_x_ptr, eps);
      // 1.2 bn backward for z, get dconv_out_z, dscale_z, dbias_z
      std::shared_ptr<CudnnScaleBiasAddReluOp<T>> sbar_z_op(
          new CudnnScaleBiasAddReluOp<T>(false, false));
      sbar_z_op->Init(dev_ctx, "", output_shape, bitmask_shape, output_shape,
                      param_shape);
      sbar_z_op->Backward(dev_ctx, dz_temp_ptr, conv_out_z_ptr, nullptr,
                          scale_z_ptr, bias_z_ptr, saved_mean_z_ptr,
                          saved_invstd_z_ptr, dconv_out_z_ptr, nullptr,
                          dscale_z_ptr, dbias_z_ptr, eps);
      // 1.3 conv backward for z, get dinput_z and dfilter_z
      std::shared_ptr<CudnnNormConvolutionOp<T>> conv_z_op(
          new CudnnNormConvolutionOp<T>());

      auto input_z_shape = framework::vectorize<int>(input_z->dims());
      auto filter_z_shape = framework::vectorize<int>(filter_z->dims());
      conv_z_op->Init(dev_ctx, input_z_shape, filter_z_shape, output_shape, pad,
                      stride, dilate, group);
      conv_z_op->Backward(dev_ctx, input_z_ptr, dconv_out_z_ptr, filter_z_ptr,
                          dinput_z_ptr, dfilter_z_ptr);
    } else {
      if (fused_add) {
        // 1.1 bn add relu backward for x, get dconv_out_x, dscale_x, dbias_x
        // and dinput_z
        Tensor *dinput_z = ctx.Output<Tensor>(framework::GradVarName("Z"));
        MALLOC_AND_GET_PTR(dinput_z, T, place)
        sbar_x_op->Init(dev_ctx, act_type, output_shape, bitmask_shape,
                        output_shape, param_shape, output_shape);
        sbar_x_op->Backward(dev_ctx, doutput_ptr, conv_out_x_ptr, output_ptr,
                            scale_x_ptr, bias_x_ptr, saved_mean_x_ptr,
                            saved_invstd_x_ptr, dconv_out_x_ptr, dinput_z_ptr,
                            dscale_x_ptr, dbias_x_ptr, eps);
      } else {
        // 1.1 bn add relu backward for x, get dconv_out_x, dscale_x, dbias_x
        sbar_x_op->Init(dev_ctx, act_type, output_shape, bitmask_shape,
                        output_shape, param_shape);
        sbar_x_op->Backward(dev_ctx, doutput_ptr, conv_out_x_ptr, output_ptr,
                            scale_x_ptr, bias_x_ptr, saved_mean_x_ptr,
                            saved_invstd_x_ptr, dconv_out_x_ptr, nullptr,
                            dscale_x_ptr, dbias_x_ptr, eps);
      }
    }
    // 2. conv backward for x, get dinput_x and dfilter_x
    std::shared_ptr<CudnnNormConvolutionOp<T>> conv_x_op(
        new CudnnNormConvolutionOp<T>());
    conv_x_op->Init(dev_ctx, input_x_shape, filter_x_shape, output_shape, pad,
                    stride, dilate, group);
    conv_x_op->Backward(dev_ctx, input_x_ptr, dconv_out_x_ptr, filter_x_ptr,
                        dinput_x_ptr, dfilter_x_ptr);
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
