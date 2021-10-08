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
#include "paddle/fluid/platform/cudnn_desc.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
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
        param_shape.size(), 4U,
        platform::errors::InvalidArgument(
            "The size of param_shape is expected to 4. But recieved "
            "param_shape's size is %d, param_shape is [%s].",
            param_shape.size(), framework::make_ddim(param_shape)));

    in_desc.set(param_shape, format, param_dtype);
    out_desc.set(param_shape, format, dtype);
  }

  cudnnDataType_t dtype;
  cudnnDataType_t param_dtype;
  cudnnTensorFormat_t format;

  platform::TensorDescriptor in_desc;
  platform::TensorDescriptor out_desc;
};

template <typename T>
class CudnnBNStatsFinalize {
 public:
  CudnnBNStatsFinalize(const platform::CUDADeviceContext &ctx,
                       const std::vector<int> &param_shape) {
    args_.Set(param_shape);
  }
  ~CudnnBNStatsFinalize() {}

  void Forward(const platform::CUDADeviceContext &ctx, float *sum_ptr,
               float *sum_of_squares_ptr, float *scale_ptr, float *bias_ptr,
               float *saved_mean_ptr, float *saved_invstd_ptr,
               float *running_mean_ptr, float *running_var_ptr,
               T *equiv_scale_ptr, T *equiv_bias_ptr, double eps,
               float momentum, int64_t ele_count, bool is_train) {
    CudnnFusionOp *op =
        is_train ? GetTrainForwardOp(ctx) : GetInferenceForwardOp(ctx);

    // Set variant_param for both inference_op_ and train_op_
    op->SetOpVariantParamAttrPtr(CUDNN_PTR_BN_SCALE, scale_ptr);
    op->SetOpVariantParamAttrPtr(CUDNN_PTR_BN_BIAS, bias_ptr);
    op->SetOpVariantParamAttrPtr(CUDNN_PTR_BN_RUNNING_MEAN, running_mean_ptr);
    op->SetOpVariantParamAttrPtr(CUDNN_PTR_BN_RUNNING_VAR, running_var_ptr);
    op->SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQSCALE, equiv_scale_ptr);
    op->SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQBIAS, equiv_bias_ptr);
    op->SetOpVariantParamAttrPtr<double>(CUDNN_SCALAR_DOUBLE_BN_EPSILON, &eps);

    // Set extra variant_param only for train_op_:
    if (is_train) {
      op->SetOpVariantParamAttrPtr(CUDNN_PTR_YSUM, sum_ptr);
      op->SetOpVariantParamAttrPtr(CUDNN_PTR_YSQSUM, sum_of_squares_ptr);
      op->SetOpVariantParamAttrPtr(CUDNN_PTR_BN_SAVED_MEAN, saved_mean_ptr);
      op->SetOpVariantParamAttrPtr(CUDNN_PTR_BN_SAVED_INVSTD, saved_invstd_ptr);
      double avg_factor = 1.0 - momentum;
      op->SetOpVariantParamAttrPtr(CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT,
                                   &ele_count);
      op->SetOpVariantParamAttrPtr(CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR,
                                   &avg_factor);
    }
    // fused op execute
    auto handle = ctx.cudnn_handle();
    op->Execute(handle);
  }

 private:
  CudnnFusionOp *GetTrainForwardOp(const platform::CUDADeviceContext &ctx) {
    CudnnFusionOp *train_op =
        new CudnnFusionOp(CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING);
    // Set constant_param for train op
    train_op->SetOpConstParamAttr(
        {CUDNN_PARAM_YSUM_PLACEHOLDER, CUDNN_PARAM_YSQSUM_PLACEHOLDER,
         CUDNN_PARAM_BN_SCALE_PLACEHOLDER, CUDNN_PARAM_BN_BIAS_PLACEHOLDER,
         CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER,
         CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER,
         CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER,
         CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER,
         CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER},
        CUDNN_PTR_16B_ALIGNED);
    // Set input and output desc for train op
    train_op->SetOpConstParamDesc(
        {CUDNN_PARAM_YSTATS_DESC, CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC},
        args_.in_desc.desc());
    train_op->SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC,
                                  args_.out_desc.desc());

    // Get workspace
    auto handle = ctx.cudnn_handle();
    train_op->SetOpConstParamAttr(CUDNN_PARAM_BN_MODE,
                                  CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
    // Check workspace size, also creates plan.
    size_t workspace_size_bytes = train_op->GetWorkspaceSizeInBytes(handle);
    PADDLE_ENFORCE_EQ(workspace_size_bytes, 0U,
                      platform::errors::InvalidArgument(
                          "Unexpected non-zero workspace size for "
                          "CudnnBNStatsFinalize."));
    train_op->SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE,
                                       static_cast<void *>(nullptr));
    train_op->SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE,
                                       &workspace_size_bytes);
    return train_op;
  }

  CudnnFusionOp *GetInferenceForwardOp(const platform::CUDADeviceContext &ctx) {
    CudnnFusionOp *inference_op =
        new CudnnFusionOp(CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE);
    // Set constant_param for inference op
    inference_op->SetOpConstParamAttr(
        {CUDNN_PARAM_BN_SCALE_PLACEHOLDER, CUDNN_PARAM_BN_BIAS_PLACEHOLDER,
         CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER,
         CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER,
         CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER},
        CUDNN_PTR_16B_ALIGNED);
    // Set input and output desc for inference op
    inference_op->SetOpConstParamDesc(CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC,
                                      args_.in_desc.desc());
    inference_op->SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC,
                                      args_.out_desc.desc());

    // Get workspace
    auto handle = ctx.cudnn_handle();
    inference_op->SetOpConstParamAttr(CUDNN_PARAM_BN_MODE,
                                      CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
    // Check workspace size, also creates plan.
    size_t workspace_size_bytes = inference_op->GetWorkspaceSizeInBytes(handle);
    PADDLE_ENFORCE_EQ(workspace_size_bytes, 0U,
                      platform::errors::InvalidArgument(
                          "Unexpected non-zero workspace size for "
                          "CudnnBNStatsFinalize."));
    inference_op->SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE,
                                           static_cast<void *>(nullptr));
    inference_op->SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE,
                                           &workspace_size_bytes);
    return inference_op;
  }

  BNStatsFinalizeArgs<T> args_;
};
#endif
}  // namespace operators
}  // namespace paddle
