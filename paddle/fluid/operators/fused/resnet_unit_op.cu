// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
    auto *input_x = ctx.Input<Tensor>("X");
    auto *filter_x = ctx.Input<Tensor>("FilterX");
    auto *scale_x = ctx.Input<Tensor>("ScaleX");
    auto *bias_x = ctx.Input<Tensor>("BiasX");
    // norm conv
    auto *conv_out_x = ctx.Output<Tensor>("ConvX");
    auto *sum_x = ctx.Output<Tensor>("SumX");
    auto *sum_of_squares_x = ctx.Output<Tensor>("SqSumX");
    // bn finalize
    auto *saved_mean_x = ctx.Output<Tensor>("SavedMeanX");
    auto *saved_invstd_x = ctx.Output<Tensor>("SavedInvstdX");
    auto *running_mean_x = ctx.Output<Tensor>("RunningMeanX");
    auto *running_var_x = ctx.Output<Tensor>("RunningVarX");
    auto *equiv_scale_x = ctx.Output<Tensor>("EqScaleX");
    auto *equiv_bias_x = ctx.Output<Tensor>("EqBiasX");
    // sbar
    auto *output = ctx.Output<Tensor>("Y");
    auto *bitmask = ctx.Output<Tensor>("BitMask");

    // tensor shape
    auto input_x_shape = framework::vectorize<int>(input_x->dims());
    auto filter_x_shape = framework::vectorize<int>(filter_x->dims());
    auto output_shape = framework::vectorize<int>(output->dims());
    auto bitmask_shape = framework::vectorize<int>(bitmask->dims());
    auto place = input_x->place();

#define MALLOC_AND_GET_PTR(TR, Dtype, Place) \
  auto TR##_ptr = TR->mutable_data<Dtype>(Place);

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

    // 1. Conv
    auto input_x_ptr = input_x->data<T>();
    auto filter_x_ptr = filter_x->data<T>();
    CuDNNNormConvolutionOp<T> *conv_x_op = new CuDNNNormConvolutionOp<T>();
    conv_x_op->Init(ctx, input_x_shape, filter_x_shape, output_shape);
    conv_x_op->Forward(ctx, input_x_ptr, filter_x_ptr, conv_out_x_ptr,
                       sum_x_ptr, sum_of_squares_x_ptr);
    // 2. BN
    auto scale_x_ptr = scale_x->data<float>();
    auto bias_x_ptr = bias_x->data<float>();
    CuDNNBNStatsFinalizeOp<T> *bn_x_op = new CuDNNBNStatsFinalizeOp<T>();
    bn_x_op->Init(ctx, filter_x_shape);
    bn_x_op->Forward(ctx, sum_x_ptr, sum_of_squares_x_ptr, scale_x_ptr,
                     bias_x_ptr, saved_mean_x_ptr, saved_invstd_x_ptr,
                     running_mean_x_ptr, running_var_x_ptr, equiv_scale_x_ptr,
                     equiv_bias_x_ptr);

    // 3. scale + bias + add + relu
    bool has_shortcut = ctx.Attr<bool>("has_shortcut");
    bool fused_add = ctx.Attr<bool>("fused_add");
    CuDNNScaleBiasAddReluOp<T> *sbar_op = new CuDNNScaleBiasAddReluOp<T>();
    if (has_shortcut) {
      // input z
      auto *input_z = ctx.Input<Tensor>("Z");
      auto *filter_z = ctx.Input<Tensor>("FilterZ");
      auto *scale_z = ctx.Input<Tensor>("ScaleZ");
      auto *bias_z = ctx.Input<Tensor>("BiasZ");
      // norm conv
      auto *conv_out_z = ctx.Output<Tensor>("ConvZ");
      auto *sum_z = ctx.Output<Tensor>("SumZ");
      auto *sum_of_squares_z = ctx.Output<Tensor>("SqSumZ");
      // bn finalize
      auto *saved_mean_z = ctx.Output<Tensor>("SavedMeanZ");
      auto *saved_invstd_z = ctx.Output<Tensor>("SavedInvstdZ");
      auto *running_mean_z = ctx.Output<Tensor>("RunningMeanZ");
      auto *running_var_z = ctx.Output<Tensor>("RunningVarZ");
      auto *equiv_scale_z = ctx.Output<Tensor>("EqScaleZ");
      auto *equiv_bias_z = ctx.Output<Tensor>("EqBiasZ");
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
      auto input_z_ptr = input_z->data<T>();
      auto filter_z_ptr = filter_z->data<T>();
      CuDNNNormConvolutionOp<T> *conv_z_op = new CuDNNNormConvolutionOp<T>();
      conv_z_op->Init(ctx, input_z_shape, filter_z_shape, output_shape);
      conv_z_op->Forward(ctx, input_z_ptr, filter_z_ptr, conv_out_z_ptr,
                         sum_z_ptr, sum_of_squares_z_ptr);
      // 3.2 BN for second input
      auto scale_z_ptr = scale_z->data<float>();
      auto bias_z_ptr = bias_z->data<float>();
      CuDNNBNStatsFinalizeOp<T> *bn_z_op = new CuDNNBNStatsFinalizeOp<T>();
      bn_z_op->Init(ctx, filter_z_shape);
      bn_z_op->Forward(ctx, sum_z_ptr, sum_of_squares_z_ptr, scale_z_ptr,
                       bias_z_ptr, saved_mean_z_ptr, saved_invstd_z_ptr,
                       running_mean_z_ptr, running_var_z_ptr, equiv_scale_z_ptr,
                       equiv_bias_z_ptr);
      // 3.3 sbar
      sbar_op->Init(ctx, output_shape, output_shape, bitmask_shape,
                    output_shape);
      sbar_op->Forward(ctx, conv_out_x_ptr, equiv_scale_x_ptr, equiv_bias_x_ptr,
                       output_ptr, bitmask_ptr, conv_out_z_ptr,
                       equiv_scale_z_ptr, equiv_bias_z_ptr);
    } else {
      if (fused_add) {
        // input z
        auto *input_z = ctx.Input<Tensor>("Z");
        auto input_z_ptr = input_z->data<T>();
        auto input_z_shape = framework::vectorize<int>(input_z->dims());
        sbar_op->Init(ctx, output_shape, output_shape, bitmask_shape,
                      input_z_shape);
        sbar_op->Forward(ctx, conv_out_x_ptr, equiv_scale_x_ptr,
                         equiv_bias_x_ptr, output_ptr, bitmask_ptr,
                         const_cast<T *>(input_z_ptr));
      } else {
        sbar_op->Init(ctx, output_shape, output_shape, bitmask_shape);
        sbar_op->Forward(ctx, conv_out_x_ptr, equiv_scale_x_ptr,
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
