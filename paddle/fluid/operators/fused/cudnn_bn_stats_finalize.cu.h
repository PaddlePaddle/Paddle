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

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
namespace dynload = platform::dynload;

#if CUDNN_VERSION >= 8000
template <typename T>
class CudnnBNStatsFinalizeOp {
 public:
  CudnnBNStatsFinalizeOp()
      : train_op_(CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING),
        inference_op_(CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE) {
    dtype_ = platform::CudnnDataType<T>::type;
    dtype_param_ = (dtype_ == CUDNN_DATA_HALF) ? CUDNN_DATA_FLOAT : dtype_;
  }

  ~CudnnBNStatsFinalizeOp() {}

  void Init(const platform::CUDADeviceContext &ctx,
            const std::vector<int> &param_shape) {
    InitDescriptors(ctx, param_shape);

    // Set constant_param for train op
    train_op_.SetOpConstParamAttr(
        {CUDNN_PARAM_YSUM_PLACEHOLDER, CUDNN_PARAM_YSQSUM_PLACEHOLDER,
         CUDNN_PARAM_BN_SCALE_PLACEHOLDER, CUDNN_PARAM_BN_BIAS_PLACEHOLDER,
         CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER,
         CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER,
         CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER,
         CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER,
         CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER},
        CUDNN_PTR_ELEM_ALIGNED);
    // Set input and output desc for train op
    train_op_.SetOpConstParamDesc(
        {CUDNN_PARAM_YSTATS_DESC, CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC},
        in_desc_.desc());
    train_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC,
                                  out_desc_.desc());

    // Set constant_param for inference op
    inference_op_.SetOpConstParamAttr(
        {CUDNN_PARAM_BN_SCALE_PLACEHOLDER, CUDNN_PARAM_BN_BIAS_PLACEHOLDER,
         CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER,
         CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER,
         CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER},
        CUDNN_PTR_ELEM_ALIGNED);
    // Set input and output desc for inference op
    inference_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC,
                                      in_desc_.desc());
    inference_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC,
                                      out_desc_.desc());

    // Get workspace
    auto handle = ctx.cudnn_handle();
    for (auto op : {&train_op_, &inference_op_}) {
      op->SetOpConstParamAttr(CUDNN_PARAM_BN_MODE, CUDNN_BATCHNORM_SPATIAL);
      // Check workspace size, also creates plan.
      size_t workspace_size_bytes = op->GetWorkspaceSizeInBytes(handle);
      PADDLE_ENFORCE_EQ(workspace_size_bytes, 0U,
                        platform::errors::InvalidArgument(
                            "Unexpected non-zero workspace size for "
                            "CudnnBNStatsFinalize op."));
      op->SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE,
                                   static_cast<void *>(nullptr));
      op->SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE, &workspace_size_bytes);
    }
  }

  void Forward(const platform::CUDADeviceContext &ctx, float *sum_ptr,
               float *sum_of_squares_ptr, float *scale_ptr, float *bias_ptr,
               float *saved_mean_ptr, float *saved_invstd_ptr,
               float *running_mean_ptr, float *running_var_ptr,
               T *equiv_scale_ptr, T *equiv_bias_ptr, double eps,
               float momentum, int64_t ele_count, bool is_train) {
    auto &op = is_train ? train_op_ : inference_op_;

    // Set variant_param for both inference_op_ and train_op_
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

  // TBD
  void Backward(const platform::CUDADeviceContext &ctx) {}

 private:
  void InitDescriptors(const platform::CUDADeviceContext &ctx,
                       const std::vector<int> &param_shape) {
    cudnnTensorFormat_t format = CUDNN_TENSOR_NHWC;
    in_desc_.set(param_shape, format, dtype_param_);
    out_desc_.set(param_shape, format, dtype_);
  }

  cudnnDataType_t dtype_;
  cudnnDataType_t dtype_param_;
  platform::TensorDescriptor in_desc_;
  platform::TensorDescriptor out_desc_;

  CudnnFusionOp train_op_;
  CudnnFusionOp inference_op_;
};
#endif
}  // namespace operators
}  // namespace paddle
