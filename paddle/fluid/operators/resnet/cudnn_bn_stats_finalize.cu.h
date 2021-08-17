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

#include "paddle/fluid/operators/resnet/cudnn_fusion_helper.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using dynload = platform::dynload;
template <typename T>
class CuDNNBNStatsFinalizeOp {
 public:
  CuDNNBNStatsFinalizeOp()
#if CUDNN_VERSION >= 8000
      : train_op_(CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING),
        inference_op_(CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE)
#endif
  {
    dtype_ = platform::CudnnDataType<T>::type;
    // For float16 input type beta, gamma, mean, and average are stored in
    // float32.
    // For other input types, these parameters have the same type as input
    dtype_param_ = (dtype_ == CUDNN_DATA_HALF) ? CUDNN_DATA_FLOAT : dtype_;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&out_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&in_desc_));
  }

  void Init(const framework::ExecutionContext &ctx,
            const std::vector<int> &output_shape) {
    InitDescriptors(ctx, output_shape);

#if CUDNN_VERSION >= 8000
    // Set up the 'Const Param Pack' for the BNForwardFinalizeStatisticsTraining
    // op
    // Describe pointer alignments
    train_op_.SetOpConstParamAttr(
        {CUDNN_PARAM_YSUM_PLACEHOLDER, CUDNN_PARAM_YSQSUM_PLACEHOLDER,
         CUDNN_PARAM_BN_SCALE_PLACEHOLDER, CUDNN_PARAM_BN_BIAS_PLACEHOLDER,
         CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER,
         CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER,
         CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER,
         CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER,
         CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER},
        CUDNN_PTR_ELEM_ALIGNED);
    train_op_.SetOpConstParamAttr(
        {CUDNN_PARAM_BN_SCALE_PLACEHOLDER, CUDNN_PARAM_BN_BIAS_PLACEHOLDER},
        CUDNN_PTR_NULL);
    // Set the I/O descriptors
    // sum and sum_squares input descriptor (typically fp32). Also
    // scale, bias, running_mean and running_var input descriptors, as well as
    // the
    // saved_mean and saved_inv_std output descriptor (typically fp32)
    train_op_.SetOpConstParamDesc(
        {CUDNN_PARAM_YSTATS_DESC, CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC},
        in_desc_);
    // equiv_scale and equiv_bias output descriptor (typically fp16)
    train_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC, out_desc_);

    // Set up the 'Const Param Pack' for the
    // BNForwardFinalizeStatisticsInference op
    inference_op_.SetOpConstParamAttr(
        {CUDNN_PARAM_BN_SCALE_PLACEHOLDER, CUDNN_PARAM_BN_BIAS_PLACEHOLDER,
         CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER,
         CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER,
         CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER},
        CUDNN_PTR_ELEM_ALIGNED);
    // Set the I/O descriptors
    // scale, bias, running_mean and running_var input descriptors, as well as
    // the
    // saved_mean and saved_inv_std output descriptor (typically fp32)
    inference_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC,
                                      in_desc_);
    // equiv_scale and equiv_bias output descriptor (typically fp16)
    inference_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC,
                                      out_desc_);

    // Perform some actions identically on both train and inference ops.
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    for (auto op : {&train_op_, &inference_op_}) {
      // Set the mode parameter in the ops, can't be
      // CUDNN_BATCHNORM_PER_ACTIVATION.
      op->SetOpConstParamAttr(CUDNN_PARAM_BN_MODE, CUDNN_BATCHNORM_SPATIAL);
      // Check workspace size, also creates 'plan'.
      size_t workspace_size_bytes = op->GetWorkspaceSizeInBytes(handle);
      PADDLE_ENFORCE_EQ(workspace_size_bytes, 0U,
                        platform::errors::InvalidArgument(
                            "Unexpected non-zero workspace size for "
                            "CuDNNBNStatsFinalize op."));
      op->SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE,
                                   static_cast<void *>(nullptr));
      op->SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE, &workspace_size_bytes);
    }
#endif
  }

  ~CuDNNBNStatsFinalizeOp() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(out_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(in_desc_));
  }

  void Forward(const framework::ExecutionContext &ctx, float *sum_ptr,
               float *sum_of_squares_ptr, float *saved_mean_ptr,
               float *saved_invstd_ptr, float *running_mean_ptr,
               float *running_var_ptr, T *equiv_scale_ptr, T *equiv_bias_ptr) {
#if CUDNN_VERSION < 8000
    LOG(FATAL) << "cuDNN version 8.0 or later is required.";
#else
    auto &op = true ? train_op_ : inference_op_;

    // The prep needed for the train_op_ is a superset of that needed for the
    // inference_op_.
    // Start here with the common prep needed for the inference_op_:
    // Set data pointers in the 'variant param pack'
    op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_RUNNING_MEAN, running_mean_ptr);
    op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_RUNNING_VAR, running_var_ptr);
    op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQSCALE, equiv_scale_ptr);
    op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQBIAS, equiv_bias_ptr);
    // Set some additional light-weight parameters in the 'variant param pack'
    op.SetOpVariantParamAttrPtr<double>(CUDNN_SCALAR_DOUBLE_BN_EPSILON, &eps);

    // Now add additional prep needed only for train_op_:
    if (true) {
      // Set data pointers in the 'variant param pack
      op.SetOpVariantParamAttrPtr(CUDNN_PTR_YSUM, sum_ptr);
      op.SetOpVariantParamAttrPtr(CUDNN_PTR_YSQSUM, sum_squares_ptr);
      op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_SAVED_MEAN, saved_mean_ptr);
      op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_SAVED_INVSTD, saved_invstd_ptr);
      // Set some additional light-weight parameters in the 'variant param pack'
      double avg_factor = 1.0 - momentum;
      int64_t ele_count = static_cast<int64_t>(ele_count);
      op.SetOpVariantParamAttrPtr(CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT,
                                  &elem_count);
      op.SetOpVariantParamAttrPtr(CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR,
                                  &avg_factor);
    }
    // Finally, launch op
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    op.Execute(handle);
#endif  // CUDNN_VERSION >= 7600
  }

  void Backward(const framework::ExecutionContext &ctx) {}

 private:
  void InitDescriptors(const framework::ExecutionContext &ctx,
                       const std::vector<int> &output_shape) {
    int c = output_shape.back();
    cudnnTensorFormat_t format = CUDNN_TENSOR_NHWC;
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensor4dDescriptor(
        out_desc_, format, dtype_, 1, c, 1, 1));
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensor4dDescriptor(
        in_desc_, format, dtype_param_, 1, c, 1, 1));
  }

  cudnnDataType_t dtype_;
  cudnnDataType_t dtype_param_;
  cudnnTensorDescriptor_t out_desc_, in_desc_;

#if CUDNN_VERSION >= 8000
  // New 'fused op' for BN stats finalize forward (training mode)
  CuDNNFusionOp train_op_;
  // New 'fused op' for BN stats finalize forward (inference mode)
  CuDNNFusionOp inference_op_;
#endif
};
}  // namespace operators
}  // namespace paddle
