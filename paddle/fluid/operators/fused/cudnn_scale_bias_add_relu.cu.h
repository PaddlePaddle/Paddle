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

#include <numeric>
#include "paddle/fluid/operators/fused/cudnn_fusion_helper.h"
#include "paddle/fluid/operators/fused/resnet_unit_op.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
namespace dynload = platform::dynload;
template <typename T>
class CuDNNScaleBiasAddReluOp {
 public:
  CuDNNScaleBiasAddReluOp()
#if CUDNN_VERSION >= 8000
      : fwd_op_(CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK)
#endif
  {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&in_x_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&in_x_bn_eq_bias_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&in_x_bn_eq_scale_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&equiv_x_scale_bias_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&in_z_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&in_z_bn_eq_bias_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&in_z_bn_eq_scale_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&equiv_z_scale_bias_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateActivationDescriptor(&activation_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&out_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&bitmask_desc_));
  }

  ~CuDNNScaleBiasAddReluOp() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(in_x_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(in_x_bn_eq_bias_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(in_x_bn_eq_scale_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(equiv_x_scale_bias_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(in_z_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(in_z_bn_eq_bias_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(in_z_bn_eq_scale_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(equiv_z_scale_bias_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyActivationDescriptor(activation_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(out_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(bitmask_desc_));
  }

  void Init(const framework::ExecutionContext &ctx,
            const std::vector<int> &out_shape, const std::vector<int> &x_shape,
            const std::vector<int> &bitmask_shape,
            std::vector<int> z_shape = {}) {
#if CUDNN_VERSION < 8000
    LOG(FATAL) << "cuDNN version 8.0 or later is required.";
#else
    has_shortcut_ = ctx.Attr<bool>("has_shortcut");
    fused_add_ = ctx.Attr<bool>("fused_add");
    dtype_ = platform::CudnnDataType<T>::type;
    format_ = CUDNN_TENSOR_NHWC;
    InitDescriptors(ctx, out_shape, bitmask_shape, x_shape, z_shape);
    GetTempSize(ctx);
#endif
  }

  void Forward(const framework::ExecutionContext &ctx, T *x_ptr, T *x_scale_ptr,
               T *x_bias_ptr, T *out_ptr, int32_t *bitmask_ptr,
               T *z_ptr = nullptr, T *z_scale_ptr = nullptr,
               T *z_bias_ptr = nullptr) {
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();
#if CUDNN_VERSION < 8000
    LOG(FATAL) << "cuDNN version 8.0 or later is required.";
#else
    // Set variant_param
    // input ptr
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_XDATA, x_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQSCALE, x_scale_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQBIAS, x_bias_ptr);
    if (has_shortcut_) {
      fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_ZDATA, z_ptr);
      fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_Z_EQSCALE, z_scale_ptr);
      fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_Z_EQBIAS, z_bias_ptr);
    } else {
      if (fused_add_) {
        fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_ZDATA, z_ptr);
      }
    }

    fwd_op_.SetOpVariantParamAttrPtr(
        CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES, &fwd_workspace_byte_);

    // output ptr
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_YDATA, out_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_ACTIVATION_BITMASK, bitmask_ptr);

    workspace_handle.RunFunc(
        [&](void *workspace_ptr) {
          // workspace ptr
          fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE, workspace_ptr);
          // workspace ptr
          fwd_op_.Execute(handle);
        },
        fwd_workspace_byte_);
#endif  // CUDNN_VERSION >= 8000
  }

  void Backward(const framework::ExecutionContext &ctx) {}

 private:
  void InitDescriptors(const framework::ExecutionContext &ctx,
                       const std::vector<int> &out_shape,
                       const std::vector<int> &bitmask_shape,
                       const std::vector<int> &x_shape,
                       std::vector<int> z_shape = {}) {
#if CUDNN_VERSION < 8000
    LOG(FATAL) << "cuDNN version 8.0 or later is required.";
#else
    int input_channel = x_shape[1];
    std::vector<int> equiv_scale_shape = {1, input_channel, 1, 1};

    // Set constant_param
    if (has_shortcut_) {
      fwd_op_.SetOpConstParamAttr(
          {CUDNN_PARAM_XDATA_PLACEHOLDER, CUDNN_PARAM_ZDATA_PLACEHOLDER,
           CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER,
           CUDNN_PARAM_YDATA_PLACEHOLDER},
          CUDNN_PTR_16B_ALIGNED);
      fwd_op_.SetOpConstParamAttr({CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
                                   CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER},
                                  CUDNN_PTR_16B_ALIGNED);
      fwd_op_.SetOpConstParamAttr({CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER,
                                   CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER},
                                  CUDNN_PTR_16B_ALIGNED);
    } else {
      if (fused_add_) {
        fwd_op_.SetOpConstParamAttr(
            {CUDNN_PARAM_XDATA_PLACEHOLDER, CUDNN_PARAM_ZDATA_PLACEHOLDER,
             CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER,
             CUDNN_PARAM_YDATA_PLACEHOLDER},
            CUDNN_PTR_16B_ALIGNED);
        fwd_op_.SetOpConstParamAttr({CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
                                     CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER},
                                    CUDNN_PTR_16B_ALIGNED);
      } else {
        fwd_op_.SetOpConstParamAttr({CUDNN_PARAM_XDATA_PLACEHOLDER,
                                     CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER,
                                     CUDNN_PARAM_YDATA_PLACEHOLDER},
                                    CUDNN_PTR_16B_ALIGNED);
        fwd_op_.SetOpConstParamAttr({CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
                                     CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER},
                                    CUDNN_PTR_16B_ALIGNED);
      }
    }

    // set input desc
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptorEx(
        in_x_desc_, format_, dtype_, x_shape.size(), x_shape.data()));
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_XDESC, in_x_desc_);
    if (has_shortcut_ || fused_add_) {
      PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptorEx(
          in_z_desc_, format_, dtype_, z_shape.size(), z_shape.data()));
      fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_ZDESC, in_z_desc_);
    }

    // set scale/bias desc
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptorEx(
        equiv_x_scale_bias_desc_, format_, dtype_, equiv_scale_shape.size(),
        equiv_scale_shape.data()));
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC,
                                equiv_x_scale_bias_desc_);
    if (has_shortcut_) {
      PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptorEx(
          equiv_z_scale_bias_desc_, format_, dtype_, equiv_scale_shape.size(),
          equiv_scale_shape.data()));
      fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC,
                                  equiv_z_scale_bias_desc_);
    }

    // set output desc
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptorEx(
        out_desc_, format_, dtype_, out_shape.size(), out_shape.data()));
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_YDESC, out_desc_);

    // set bitmask desc
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptorEx(
        bitmask_desc_, format_, CUDNN_DATA_INT32, bitmask_shape.size(),
        bitmask_shape.data()));
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_ACTIVATION_BITMASK_DESC,
                                bitmask_desc_);

    // set activation desc
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_IDENTITY;
    if (ctx.Attr<std::string>("act_type") != "") {
      PADDLE_ENFORCE_EQ(
          ctx.Attr<std::string>("act_type"), "relu",
          platform::errors::InvalidArgument(
              "Only relu activation supported in normalized convolution."));
      mode = CUDNN_ACTIVATION_RELU;
    }
    auto nan_prop = CUDNN_NOT_PROPAGATE_NAN;
    double dummy_clip = 0.0;
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetActivationDescriptor(
        activation_desc_, mode, nan_prop, dummy_clip));
    if (mode != CUDNN_ACTIVATION_IDENTITY) {
      fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_ACTIVATION_DESC,
                                  activation_desc_);
    }

    // others
    fwd_op_.SetOpConstParamAttr(CUDNN_PARAM_BN_MODE,
                                CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
#endif
  }

  void GetTempSize(const framework::ExecutionContext &ctx) {
#if CUDNN_VERSION < 8000
    LOG(FATAL) << "cuDNN version 8.0 or later is required.";
#else
    // Make op plan and get workspace size
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    fwd_workspace_byte_ = fwd_op_.GetWorkspaceSizeInBytes(handle);
#endif
  }

#if CUDNN_VERSION >= 8000
  bool has_shortcut_ = false;
  bool fused_add_ = false;
  size_t fwd_workspace_byte_;

  cudnnDataType_t dtype_;

  cudnnTensorDescriptor_t in_x_desc_;
  cudnnTensorDescriptor_t in_x_bn_eq_bias_;
  cudnnTensorDescriptor_t in_x_bn_eq_scale_;
  cudnnTensorDescriptor_t in_z_desc_;
  cudnnTensorDescriptor_t in_z_bn_eq_bias_;
  cudnnTensorDescriptor_t in_z_bn_eq_scale_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnTensorDescriptor_t bitmask_desc_;
  cudnnTensorDescriptor_t equiv_x_scale_bias_desc_;
  cudnnTensorDescriptor_t equiv_z_scale_bias_desc_;
  cudnnActivationDescriptor_t activation_desc_;

  cudnnTensorFormat_t format_;
  CuDNNFusionOp fwd_op_;
#endif
};
}  // namespace operators
}  // namespace paddle
