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

#include <numeric>

#include "paddle/fluid/operators/fused/cudnn_fusion_helper.h"
#include "paddle/fluid/operators/fused/resnet_unit_op.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
namespace dynload = platform::dynload;

#if CUDNN_VERSION >= 8000
template <typename T>
class CudnnScaleBiasAddReluOp {
 public:
  CudnnScaleBiasAddReluOp(bool fused_add, bool has_shortcut)
      : fused_add_(fused_add),
        has_shortcut_(has_shortcut),
        fwd_op_(CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK) {}

  ~CudnnScaleBiasAddReluOp() {}

  void Init(const platform::CUDADeviceContext &ctx, const std::string &act_type,
            const std::vector<int> &out_shape,
            const std::vector<int> &bitmask_shape,
            const std::vector<int> &x_shape,
            const std::vector<int> &param_shape,
            std::vector<int> z_shape = {}) {
    dtype_ = platform::CudnnDataType<T>::type;
    format_ = CUDNN_TENSOR_NHWC;
    InitDescriptors(ctx, act_type, out_shape, bitmask_shape, x_shape,
                    param_shape, z_shape);
    GetWorkspaceSize(ctx);
  }

  void Forward(const platform::CUDADeviceContext &ctx, T *x_ptr, T *x_scale_ptr,
               T *x_bias_ptr, T *out_ptr, int32_t *bitmask_ptr,
               T *z_ptr = nullptr, T *z_scale_ptr = nullptr,
               T *z_bias_ptr = nullptr) {
    auto handle = ctx.cudnn_handle();
    auto workspace_handle = ctx.cudnn_workspace_handle();
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
  }

  void Backward(const platform::CUDADeviceContext &ctx) {}

 private:
  void InitDescriptors(const platform::CUDADeviceContext &ctx,
                       const std::string &act_type,
                       const std::vector<int> &out_shape,
                       const std::vector<int> &bitmask_shape,
                       const std::vector<int> &x_shape,
                       const std::vector<int> &param_shape,
                       std::vector<int> z_shape = {}) {
    // Set constant_param
    if (has_shortcut_) {
      fwd_op_.SetOpConstParamAttr(
          {CUDNN_PARAM_XDATA_PLACEHOLDER, CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
           CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER, CUDNN_PARAM_YDATA_PLACEHOLDER,
           CUDNN_PARAM_ZDATA_PLACEHOLDER, CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER,
           CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER,
           CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER},
          CUDNN_PTR_16B_ALIGNED);
    } else {
      if (fused_add_) {
        fwd_op_.SetOpConstParamAttr(
            {CUDNN_PARAM_XDATA_PLACEHOLDER, CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
             CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER, CUDNN_PARAM_YDATA_PLACEHOLDER,
             CUDNN_PARAM_ZDATA_PLACEHOLDER,
             CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER},
            CUDNN_PTR_16B_ALIGNED);
      } else {
        fwd_op_.SetOpConstParamAttr(
            {CUDNN_PARAM_XDATA_PLACEHOLDER, CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
             CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER, CUDNN_PARAM_YDATA_PLACEHOLDER,
             CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER},
            CUDNN_PTR_16B_ALIGNED);
      }
    }

    // set input desc
    in_x_desc_.set(x_shape, format_, dtype_);
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_XDESC, in_x_desc_.desc());
    if (has_shortcut_ || fused_add_) {
      in_z_desc_.set(z_shape, format_, dtype_);
      fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_ZDESC, in_z_desc_.desc());
    }

    // set scale/bias desc
    equiv_x_scale_bias_desc_.set(param_shape, format_, dtype_);
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC,
                                equiv_x_scale_bias_desc_.desc());
    if (has_shortcut_) {
      equiv_z_scale_bias_desc_.set(param_shape, format_, dtype_);
      fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC,
                                  equiv_z_scale_bias_desc_.desc());
    }

    // set output desc
    out_desc_.set(out_shape, format_, dtype_);
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_YDESC, out_desc_.desc());

    // set bitmask desc
    bitmask_desc_.set(bitmask_shape, format_, CUDNN_DATA_INT32);
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_ACTIVATION_BITMASK_DESC,
                                bitmask_desc_.desc());

    // set activation desc
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_IDENTITY;
    if (act_type != "") {
      PADDLE_ENFORCE_EQ(
          act_type, "relu",
          platform::errors::InvalidArgument(
              "Only relu activation supported in normalized convolution."));
      mode = CUDNN_ACTIVATION_RELU;
    }
    double dummy_clip = 0.0;
    activation_desc_.set(mode, dummy_clip);
    if (mode != CUDNN_ACTIVATION_IDENTITY) {
      fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_ACTIVATION_DESC,
                                  activation_desc_.desc());
    }

    // others
    fwd_op_.SetOpConstParamAttr(CUDNN_PARAM_BN_MODE,
                                CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
  }

  void GetWorkspaceSize(const platform::CUDADeviceContext &ctx) {
    // Make op plan and get workspace size
    auto handle = ctx.cudnn_handle();
    fwd_workspace_byte_ = fwd_op_.GetWorkspaceSizeInBytes(handle);
  }

  bool fused_add_ = false;
  bool has_shortcut_ = false;
  size_t fwd_workspace_byte_;

  cudnnDataType_t dtype_;
  cudnnTensorFormat_t format_;

  platform::TensorDescriptor in_x_desc_;
  platform::TensorDescriptor in_z_desc_;
  platform::TensorDescriptor out_desc_;
  platform::TensorDescriptor bitmask_desc_;
  platform::TensorDescriptor equiv_x_scale_bias_desc_;
  platform::TensorDescriptor equiv_z_scale_bias_desc_;
  platform::ActivationDescriptor activation_desc_;

  CudnnFusionOp fwd_op_;
};
#endif
}  // namespace operators
}  // namespace paddle
