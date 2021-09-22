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

#define MALLOC_AND_GET_PTR(TR, Dtype, Place) \
  Dtype *TR##_ptr = TR->mutable_data<Dtype>(Place);

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
#undef MALLOC_AND_GET_PTR
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
  }
};

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
