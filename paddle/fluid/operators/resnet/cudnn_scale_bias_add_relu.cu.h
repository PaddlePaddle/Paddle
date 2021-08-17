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
#include "paddle/fluid/operators/resnet/cudnn_fusion_helper.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using dynload = platform::dynload;
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
        dynload::cudnnCreateTensorDescriptor(&equiv_scale_bias_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&equiv_z_scale_bias_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&in_z_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&in_z_bn_eq_bias_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&in_z_bn_eq_scale_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateActivationDescriptor(&activation_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&out_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateTensorDescriptor(&out_relu_bitmask_desc_));
  }

  ~CuDNNScaleBiasAddReluOp() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(in_x_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(in_x_bn_eq_bias_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(in_x_bn_eq_scale_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(equiv_scale_bias_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(equiv_z_scale_bias_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(in_z_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(in_z_bn_eq_bias_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(in_z_bn_eq_scale_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyActivationDescriptor(activation_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(out_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyTensorDescriptor(out_relu_bitmask_desc_));
  }

  void Init(const framework::ExecutionContext &ctx,
            const std::vector<int> &out_shape, const std::vector<int> &x_shape,
            const std::vector<int> &bitmask_shape,
            const std::vector<int> &z_shape = {}) {
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
    if (fwd_workspace_byte_ > workspace_handle::WorkspaceSize()) {
      workspace_handle.ReallocWorkspace(fwd_workspace_byte_);
    }
    auto workspace_ptr = workspace_handle.allocation_;
#if CUDNN_VERSION < 8000
    LOG(FATAL) << "cuDNN version 8.0 or later is required.";
#else
    // This operator does not support output blending as specified by alpha or
    // beta.
    // Set data input pointers in op instance
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

    // Set workspace input pointer in op instance
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE, workspace_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(
        CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES, &fwd_workspace_byte_);

    // Set data output pointers in op instance
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_YDATA, out_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_ACTIVATION_BITMASK, bitmask_ptr);

    // Launch forward operation
    fwd_op_.Execute(handle);
#endif  // CUDNN_VERSION >= 8000
  }

  void Backward(const framework::ExecutionContext &ctx) {}

 private:
  void InitDescriptors(const framework::ExecutionContext &ctx,
                       const std::vector<int> &out_shape,
                       const std::vector<int> &bitmask_shape,
                       const std::vector<int> &x_shape,
                       const std::vector<int> &z_shape = {}) {
#if CUDNN_VERSION < 8000
    LOG(FATAL) << "cuDNN version 8.0 or later is required.";
#else
    auto bitmask_stride = GetStrides(bitmask_shape);
    auto x_stride = GetStrides(x_shape);
    auto z_stride = GetStrides(z_shape);
    auto out_stride = GetStrides(out_shape);

    auto dual_scale_bias_ptr_type = CUDNN_PTR_16B_ALIGNED;
    // Describe i/o tensor pointer alignment for forward fused op
    if (has_shortcut_) {
      fwd_op_.SetOpConstParamAttr(
          {CUDNN_PARAM_XDATA_PLACEHOLDER, CUDNN_PARAM_ZDATA_PLACEHOLDER,
           CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER,
           CUDNN_PARAM_YDATA_PLACEHOLDER},
          CUDNN_PTR_16B_ALIGNED);
      fwd_op_.SetOpConstParamAttr({CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
                                   CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER},
                                  dual_scale_bias_ptr_type);
      fwd_op_.SetOpConstParamAttr({CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER,
                                   CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER},
                                  dual_scale_bias_ptr_type);
    } else {
      if (fused_add_) {
        fwd_op_.SetOpConstParamAttr(
            {CUDNN_PARAM_XDATA_PLACEHOLDER, CUDNN_PARAM_ZDATA_PLACEHOLDER,
             CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER,
             CUDNN_PARAM_YDATA_PLACEHOLDER},
            CUDNN_PTR_16B_ALIGNED);
        fwd_op_.SetOpConstParamAttr({CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
                                     CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER},
                                    dual_scale_bias_ptr_type);
      } else {
        fwd_op_.SetOpConstParamAttr({CUDNN_PARAM_XDATA_PLACEHOLDER,
                                     CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER,
                                     CUDNN_PARAM_YDATA_PLACEHOLDER},
                                    CUDNN_PTR_16B_ALIGNED);
        fwd_op_.SetOpConstParamAttr({CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
                                     CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER},
                                    dual_scale_bias_ptr_type);
      }
    }

    // set input descriptor
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptor(
        in_x_desc_, dtype_, static_cast<int>(x_shape.size()), x_shape.data(),
        x_stride.data()));
    if (has_shortcut_ || fused_add_) {
      PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptor(
          in_z_desc_, dtype_, static_cast<int>(z_shape.size()), z_shape.data(),
          z_stride.data()));
    }

    // set output descriptor
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptor(
        out_desc_, dtype_, static_cast<int>(out_shape.size()), out_shape.data(),
        out_stride.data()));

    // Always set scale/bias descriptors
    int input_channel = x_shape.back();
    std::vector<int> equiv_scale_shape = {1, input_channel, 1, 1};
    std::vector<int> equiv_scale_stride = {input_channel, 1, 1, 1};

    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptor(
        equiv_scale_bias_desc_,
        dtype_,
        static_cast<int>(equiv_scale_shape.size()),
        equiv_scale_shape.data(),
        equiv_scale_stride.data());
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC,
     equiv_scale_bias_desc_);
    if (has_shortcut_) {
      PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptor(
          equiv_z_scale_bias_desc_,
          dtype_,
          static_cast<int>(equiv_scale_shape.size()),
          equiv_scale_shape.data(),
          equiv_scale_stride.data());
      fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC,
                                  equiv_z_scale_bias_desc_);
    }

    // int C = input_channel;
    // int64_t NHW = std::accmulate(out_shape.beigin(), out_shape.end() - 1, 1,
    //  std::multiplies<int>());
    // int32_t C_int32Elems = ((C + 63) & ~63) / 32;
    // int32_t NHW_int32Elems = (NHW + 31) & ~31;
    // std::vector<int> bitmask_shape = {NHW_int32Elems, C_int32Elems, 1};
    // std::vector<int> bitmask_stride = {C_int32Elems, 1, 1};
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptor(
                                out_relu_bitmask_desc_,
                                CUDNN_DATA_INT32,
                                3,
                                bitmask_shape.data(),
                                bitmask_stride.data()));

    // Set activation descriptor, default is no activation
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_IDENTITY;
    if (ctx.Attr<string>("act_type")) {
      PADDLE_ENFORCE_EQ(
          ctx.Attr<string>("act_type"), "relu",
          platform::errors::InvalidArgument(
              "Only relu activation supported in normalized convolution."));
      mode = CUDNN_ACTIVATION_RELU;
    }
    auto nan_prop = CUDNN_NOT_PROPAGATE_NAN;
    double dummy_clip = 0.0;
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetActivationDescriptor(
        activation_desc_, mode, nan_prop, dummy_clip));
    // Currently, the only way to turn off activation is to not set the
    // descriptor
    if (mode != CUDNN_ACTIVATION_IDENTITY) {
      fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_ACTIVATION_DESC,
                                  activation_desc_);
    }

    // Set desc pointers
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_XDESC, in_x_desc_);
    if (has_shortcut_ || fused_add_) {
      fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_ZDESC, in_z_desc_);
    }
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_ACTIVATION_BITMASK_DESC,
                                out_relu_bitmask_desc_);
    fwd_op_.SetOpConstParamAttr(CUDNN_PARAM_BN_MODE,
                                CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_YDESC, out_desc_);
#endif
  }

  void GetTempSize(const framework::ExecutionContext &ctx) {
#if CUDNN_VERSION < 8000
    LOG(FATAL) << "cuDNN version 8.0 or later is required.";
#else
    // Make op plan for forward op and set forward workspace size
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    fwd_workspace_byte_ = fwd_op_.GetWorkspaceSizeInBytes(handle);
#endif
  }

#if CUDNN_VERSION >= 8000
  bool has_shortcut_ = false;
  bool fused_add_ = false;
  // Temp workspace size in bytes needed for Forward() operation.
  size_t fwd_workspace_byte_;

  cudnnDataType_t dtype_;
  cudnnTensorDescriptor_t in_x_desc_;
  cudnnTensorDescriptor_t in_z_desc_;

  cudnnTensorDescriptor_t in_x_bn_eq_bias_;
  cudnnTensorDescriptor_t in_x_bn_eq_scale_;

  cudnnTensorDescriptor_t in_z_bn_eq_bias_;
  cudnnTensorDescriptor_t in_z_bn_eq_scale_;

  cudnnTensorDescriptor_t out_desc_;
  cudnnTensorDescriptor_t out_relu_bitmask_desc_;

  cudnnTensorFormat_t format_;

  cudnnTensorDescriptor_t equiv_scale_bias_desc_;
  cudnnTensorDescriptor_t equiv_z_scale_bias_desc_;

  // Specifies activation parameters: relu
  cudnnActivationDescriptor_t activation_desc_;
#endif
};
}  // namespace operators
}  // namespace paddle
