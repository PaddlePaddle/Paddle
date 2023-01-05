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

#include "paddle/fluid/operators/fused/cudnn_fusion_helper.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

namespace paddle {
namespace operators {

namespace dynload = platform::dynload;
template <typename T>
using BatchNormParamType =
    typename platform::CudnnDataType<T>::BatchNormParamType;

#if CUDNN_VERSION >= 8000

template <typename T>
struct BNStatsFinalizeArgs {
  BNStatsFinalizeArgs() {
    dtype = platform::CudnnDataType<T>::type;
    param_dtype = platform::CudnnDataType<BatchNormParamType<T>>::type;
    format = CUDNN_TENSOR_NHWC;
  }

  void Set(const std::vector<int> &param_shape) {
    PADDLE_ENFORCE_EQ(
        param_shape.size(),
        4U,
        platform::errors::InvalidArgument(
            "The size of param_shape is expected to 4. But received "
            "param_shape's size is %d, param_shape is [%s].",
            param_shape.size(),
            phi::make_ddim(param_shape)));

    in_desc.set(param_shape, format, param_dtype);
    out_desc.set(param_shape, format, dtype);
  }

  cudnnDataType_t dtype;
  cudnnDataType_t param_dtype;
  cudnnTensorFormat_t format;

  phi::backends::gpu::TensorDescriptor in_desc;
  phi::backends::gpu::TensorDescriptor out_desc;
};

template <typename T>
class CudnnBNStatsFinalize {
 public:
  CudnnBNStatsFinalize(const phi::GPUContext &ctx,
                       const std::vector<int> &param_shape)
      : train_op_(CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING),
        inference_op_(CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE) {
    args_.Set(param_shape);
  }
  ~CudnnBNStatsFinalize() {}

  void Forward(const phi::GPUContext &ctx,
               const phi::DenseTensor &sum,
               const phi::DenseTensor &sum_of_squares,
               const phi::DenseTensor &scale,
               const phi::DenseTensor &bias,
               phi::DenseTensor *saved_mean,
               phi::DenseTensor *saved_invstd,
               phi::DenseTensor *running_mean,
               phi::DenseTensor *running_var,
               phi::DenseTensor *equiv_scale,
               phi::DenseTensor *equiv_bias,
               double eps,
               float momentum,
               int64_t ele_count,
               bool is_train) {
    if (is_train) {
      TrainInit(ctx);
    } else {
      InferenceInit(ctx);
    }
    auto &op = is_train ? train_op_ : inference_op_;

    // Set variant_param for both inference_op_ and train_op_
    float *sum_ptr = const_cast<float *>(sum.data<float>());
    float *sum_of_squares_ptr =
        const_cast<float *>(sum_of_squares.data<float>());
    float *scale_ptr = const_cast<float *>(scale.data<float>());
    float *bias_ptr = const_cast<float *>(bias.data<float>());
    float *saved_mean_ptr = ctx.template Alloc<float>(
        saved_mean, saved_mean->numel() * sizeof(float));
    float *saved_invstd_ptr = ctx.template Alloc<float>(
        saved_invstd, saved_invstd->numel() * sizeof(float));
    float *running_mean_ptr = ctx.template Alloc<float>(
        running_mean, running_mean->numel() * sizeof(float));
    float *running_var_ptr = ctx.template Alloc<float>(
        running_var, running_var->numel() * sizeof(float));
    T *equiv_scale_ptr =
        ctx.template Alloc<T>(equiv_scale, equiv_scale->numel() * sizeof(T));
    T *equiv_bias_ptr =
        ctx.template Alloc<T>(equiv_bias, equiv_bias->numel() * sizeof(T));
    op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_SCALE, scale_ptr);
    op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_BIAS, bias_ptr);
    op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_RUNNING_MEAN, running_mean_ptr);
    op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_RUNNING_VAR, running_var_ptr);
    op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQSCALE, equiv_scale_ptr);
    op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQBIAS, equiv_bias_ptr);
    op.SetOpVariantParamAttrPtr<double>(CUDNN_SCALAR_DOUBLE_BN_EPSILON, &eps);

    // Set extra variant_param only for train_op_:
    if (is_train) {
      op.SetOpVariantParamAttrPtr(CUDNN_PTR_YSUM, sum_ptr);
      op.SetOpVariantParamAttrPtr(CUDNN_PTR_YSQSUM, sum_of_squares_ptr);
      op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_SAVED_MEAN, saved_mean_ptr);
      op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_SAVED_INVSTD, saved_invstd_ptr);
      double avg_factor = 1.0 - momentum;
      op.SetOpVariantParamAttrPtr(CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT,
                                  &ele_count);
      op.SetOpVariantParamAttrPtr(CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR,
                                  &avg_factor);
    }
    // fused op execute
    auto handle = ctx.cudnn_handle();
    op.Execute(handle);
  }

 private:
  void TrainInit(const phi::GPUContext &ctx) {
    // Set constant_param for train op
    train_op_.SetOpConstParamAttr({CUDNN_PARAM_YSUM_PLACEHOLDER,
                                   CUDNN_PARAM_YSQSUM_PLACEHOLDER,
                                   CUDNN_PARAM_BN_SCALE_PLACEHOLDER,
                                   CUDNN_PARAM_BN_BIAS_PLACEHOLDER,
                                   CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER,
                                   CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER,
                                   CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER,
                                   CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER,
                                   CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
                                   CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER},
                                  CUDNN_PTR_16B_ALIGNED);
    // Set input and output desc for train op
    train_op_.SetOpConstParamDesc(
        {CUDNN_PARAM_YSTATS_DESC, CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC},
        args_.in_desc.desc());
    train_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC,
                                  args_.out_desc.desc());

    // Get workspace
    auto handle = ctx.cudnn_handle();
    train_op_.SetOpConstParamAttr(CUDNN_PARAM_BN_MODE,
                                  CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
    // Check workspace size, also creates plan.
    size_t workspace_size_bytes = train_op_.GetWorkspaceSizeInBytes(handle);
    PADDLE_ENFORCE_EQ(workspace_size_bytes,
                      0U,
                      platform::errors::InvalidArgument(
                          "Unexpected non-zero workspace size for "
                          "CudnnBNStatsFinalize."));
    train_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE,
                                       static_cast<void *>(nullptr));
    train_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE,
                                       &workspace_size_bytes);
  }

  void InferenceInit(const phi::GPUContext &ctx) {
    // Set constant_param for inference op
    inference_op_.SetOpConstParamAttr({CUDNN_PARAM_BN_SCALE_PLACEHOLDER,
                                       CUDNN_PARAM_BN_BIAS_PLACEHOLDER,
                                       CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER,
                                       CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER,
                                       CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
                                       CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER},
                                      CUDNN_PTR_16B_ALIGNED);
    // Set input and output desc for inference op
    inference_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC,
                                      args_.in_desc.desc());
    inference_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC,
                                      args_.out_desc.desc());

    // Get workspace
    auto handle = ctx.cudnn_handle();
    inference_op_.SetOpConstParamAttr(CUDNN_PARAM_BN_MODE,
                                      CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
    // Check workspace size, also creates plan.
    size_t workspace_size_bytes = inference_op_.GetWorkspaceSizeInBytes(handle);
    PADDLE_ENFORCE_EQ(workspace_size_bytes,
                      0U,
                      platform::errors::InvalidArgument(
                          "Unexpected non-zero workspace size for "
                          "CudnnBNStatsFinalize."));
    inference_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE,
                                           static_cast<void *>(nullptr));
    inference_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE,
                                           &workspace_size_bytes);
  }

  BNStatsFinalizeArgs<T> args_;
  CudnnFusionOp train_op_;
  CudnnFusionOp inference_op_;
};
#endif
}  // namespace operators
}  // namespace paddle
