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
using Tensor = framework::Tensor;
template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;
namespace dynload = platform::dynload;
template <typename T>
using BatchNormParamType =
    typename platform::CudnnDataType<T>::BatchNormParamType;

#if CUDNN_VERSION >= 8000

template <typename T>
struct ScaleBiasAddReluArgs {
  ScaleBiasAddReluArgs() {
    dtype = platform::CudnnDataType<T>::type;
    param_dtype = platform::CudnnDataType<BatchNormParamType<T>>::type;
    format = CUDNN_TENSOR_NHWC;
  }

  void Set(const std::string &act_type, const std::vector<int> &data_shape,
           const std::vector<int> &param_shape,
           const std::vector<int> &bitmask_shape) {
    PADDLE_ENFORCE_EQ(
        data_shape.size(), 4U,
        platform::errors::InvalidArgument(
            "The size of data_shape is expected to 4. But recieved "
            "data_shape's size is %d, data_shape is [%s].",
            data_shape.size(), phi::make_ddim(data_shape)));
    PADDLE_ENFORCE_EQ(
        param_shape.size(), 4U,
        platform::errors::InvalidArgument(
            "The size of param_shape is expected to 4. But recieved "
            "param_shape's size is %d, param_shape is [%s].",
            param_shape.size(), phi::make_ddim(param_shape)));
    PADDLE_ENFORCE_EQ(
        bitmask_shape.size(), 3U,
        platform::errors::InvalidArgument(
            "The size of bitmask_shape is expected to 3. But recieved "
            "bitmask_shape's size is %d, bitmask_shape is [%s].",
            bitmask_shape.size(), phi::make_ddim(bitmask_shape)));

    in_desc.set(data_shape, format, dtype);
    out_desc.set(data_shape, format, dtype);
    equiv_scale_bias_desc.set(param_shape, format, dtype);
    scale_bias_mean_var_desc.set(param_shape, format, param_dtype);
    bitmask_desc.set(bitmask_shape, format, CUDNN_DATA_INT32);
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
    activation_desc.set(mode, dummy_clip);
  }

  cudnnDataType_t dtype;
  cudnnDataType_t param_dtype;
  cudnnTensorFormat_t format;

  platform::TensorDescriptor in_desc;
  platform::TensorDescriptor out_desc;
  platform::TensorDescriptor equiv_scale_bias_desc;
  platform::TensorDescriptor scale_bias_mean_var_desc;
  platform::TensorDescriptor bitmask_desc;
  platform::ActivationDescriptor activation_desc;
};

template <typename T>
class CudnnScaleBiasAddRelu {
 public:
  CudnnScaleBiasAddRelu(const platform::CUDADeviceContext &ctx,
                        const std::string &act_type, bool fuse_add,
                        bool has_shortcut, const std::vector<int> &data_shape,
                        const std::vector<int> &param_shape,
                        const std::vector<int> &bitmask_shape)
      : fwd_op_(CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK),
        bwd_op_(CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM) {
    fuse_add_ = fuse_add;
    has_shortcut_ = has_shortcut;
    args_.Set(act_type, data_shape, param_shape, bitmask_shape);
  }

  ~CudnnScaleBiasAddRelu() {}

  void Forward(const platform::CUDADeviceContext &ctx, const Tensor &x,
               const Tensor &x_scale, const Tensor &x_bias, const Tensor *z,
               const Tensor *z_scale, const Tensor *z_bias, Tensor *out,
               Tensor *bitmask) {
    ForwardInit(ctx);
    auto handle = ctx.cudnn_handle();
    auto place = ctx.GetPlace();
    auto workspace_handle = ctx.cudnn_workspace_handle();
    fwd_workspace_byte_ = fwd_op_.GetWorkspaceSizeInBytes(handle);
    // Set variant_param
    // input ptr
    T *x_ptr = const_cast<T *>(x.data<T>());
    T *x_scale_ptr = const_cast<T *>(x_scale.data<T>());
    T *x_bias_ptr = const_cast<T *>(x_bias.data<T>());
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_XDATA, x_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQSCALE, x_scale_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQBIAS, x_bias_ptr);
    if (has_shortcut_) {
      T *z_ptr = const_cast<T *>(z->data<T>());
      T *z_scale_ptr = const_cast<T *>(z_scale->data<T>());
      T *z_bias_ptr = const_cast<T *>(z_bias->data<T>());
      fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_ZDATA, z_ptr);
      fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_Z_EQSCALE, z_scale_ptr);
      fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_Z_EQBIAS, z_bias_ptr);
    } else {
      if (fuse_add_) {
        T *z_ptr = const_cast<T *>(z->data<T>());
        fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_ZDATA, z_ptr);
      }
    }

    fwd_op_.SetOpVariantParamAttrPtr(
        CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES, &fwd_workspace_byte_);

    // output ptr
    T *out_ptr = out->mutable_data<T>(place);
    int32_t *bitmask_ptr = bitmask->mutable_data<int32_t>(place);
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

  void Backward(const platform::CUDADeviceContext &ctx, const Tensor &dy,
                const Tensor &x, const Tensor &scale, const Tensor &bias,
                const Tensor &saved_mean, const Tensor &saved_invstd,
                const Tensor *bitmask, Tensor *dx, Tensor *dz, Tensor *dscale,
                Tensor *dbias, double eps) {
    BackwardInit(ctx);
    auto handle = ctx.cudnn_handle();
    auto place = ctx.GetPlace();
    auto workspace_handle = ctx.cudnn_workspace_handle();
    bwd_workspace_byte_ = bwd_op_.GetWorkspaceSizeInBytes(handle);
    // Set variant_param
    // input ptr
    T *dy_ptr = const_cast<T *>(dy.data<T>());
    T *x_ptr = const_cast<T *>(x.data<T>());
    float *scale_ptr = const_cast<float *>(scale.data<float>());
    float *bias_ptr = const_cast<float *>(bias.data<float>());
    float *saved_mean_ptr = const_cast<float *>(saved_mean.data<float>());
    float *saved_invstd_ptr = const_cast<float *>(saved_invstd.data<float>());
    int32_t *bitmask_ptr =
        bitmask ? const_cast<int32_t *>(bitmask->data<int32_t>()) : nullptr;
    T *dx_ptr = dx->mutable_data<T>(place);
    T *dz_ptr = dz ? dz->mutable_data<T>(place) : nullptr;
    float *dscale_ptr = dscale ? dscale->mutable_data<float>(place) : nullptr;
    float *dbias_ptr = dbias ? dbias->mutable_data<float>(place) : nullptr;

    bwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_XDATA, x_ptr);
    bwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_DYDATA, dy_ptr);
    bwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_SCALE, scale_ptr);
    bwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_BIAS, bias_ptr);
    bwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_SAVED_MEAN, saved_mean_ptr);
    bwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_SAVED_INVSTD,
                                     saved_invstd_ptr);
    bwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_ACTIVATION_BITMASK, bitmask_ptr);

    bwd_op_.SetOpVariantParamAttrPtr(
        CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES, &bwd_workspace_byte_);

    // output ptr
    bwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_DXDATA, dx_ptr);
    bwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_DSCALE, dscale_ptr);
    bwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_DBIAS, dbias_ptr);
    bwd_op_.SetOpVariantParamAttrPtr<double>(CUDNN_SCALAR_DOUBLE_BN_EPSILON,
                                             &eps);
    if (has_shortcut_ || fuse_add_) {
      bwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_DZDATA, dz_ptr);
    }

    workspace_handle.RunFunc(
        [&](void *workspace_ptr) {
          // workspace ptr
          bwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE, workspace_ptr);
          // workspace ptr
          bwd_op_.Execute(handle);
        },
        bwd_workspace_byte_);
  }

 private:
  void ForwardInit(const platform::CUDADeviceContext &ctx) {
    // Set constant_param
    fwd_op_.SetOpConstParamAttr(
        {CUDNN_PARAM_XDATA_PLACEHOLDER, CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
         CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER, CUDNN_PARAM_YDATA_PLACEHOLDER,
         CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER},
        CUDNN_PTR_16B_ALIGNED);
    if (has_shortcut_) {
      fwd_op_.SetOpConstParamAttr(
          {CUDNN_PARAM_ZDATA_PLACEHOLDER, CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER,
           CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER},
          CUDNN_PTR_16B_ALIGNED);
    } else if (fuse_add_) {
      fwd_op_.SetOpConstParamAttr(CUDNN_PARAM_ZDATA_PLACEHOLDER,
                                  CUDNN_PTR_16B_ALIGNED);
    }

    // input desc
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_XDESC, args_.in_desc.desc());
    if (has_shortcut_ || fuse_add_) {
      fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_ZDESC, args_.in_desc.desc());
    }

    // equiv scale/bias desc
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC,
                                args_.equiv_scale_bias_desc.desc());
    if (has_shortcut_) {
      fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC,
                                  args_.equiv_scale_bias_desc.desc());
    }

    // output desc
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_YDESC, args_.out_desc.desc());

    // bitmask desc
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_ACTIVATION_BITMASK_DESC,
                                args_.bitmask_desc.desc());

    // activation desc
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_ACTIVATION_DESC,
                                args_.activation_desc.desc());

    // others
    fwd_op_.SetOpConstParamAttr(CUDNN_PARAM_BN_MODE,
                                CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
  }

  void BackwardInit(const platform::CUDADeviceContext &ctx) {
    // Set constant_param
    bwd_op_.SetOpConstParamAttr(
        {CUDNN_PARAM_XDATA_PLACEHOLDER, CUDNN_PARAM_DYDATA_PLACEHOLDER,
         CUDNN_PARAM_DXDATA_PLACEHOLDER, CUDNN_PARAM_BN_SCALE_PLACEHOLDER,
         CUDNN_PARAM_BN_BIAS_PLACEHOLDER, CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER,
         CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER,
         CUDNN_PARAM_BN_DSCALE_PLACEHOLDER, CUDNN_PARAM_BN_DBIAS_PLACEHOLDER,
         CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER},
        CUDNN_PTR_16B_ALIGNED);
    if (has_shortcut_ || fuse_add_) {
      bwd_op_.SetOpConstParamAttr(CUDNN_PARAM_DZDATA_PLACEHOLDER,
                                  CUDNN_PTR_16B_ALIGNED);
    }

    // input desc
    bwd_op_.SetOpConstParamDesc(CUDNN_PARAM_XDESC, args_.in_desc.desc());
    bwd_op_.SetOpConstParamDesc(CUDNN_PARAM_DXDESC, args_.in_desc.desc());
    if (has_shortcut_ || fuse_add_) {
      bwd_op_.SetOpConstParamDesc(CUDNN_PARAM_DZDESC, args_.in_desc.desc());
    }

    // scale/bias/mean/var desc for backward
    bwd_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC,
                                args_.scale_bias_mean_var_desc.desc());

    // output desc
    bwd_op_.SetOpConstParamDesc(CUDNN_PARAM_DYDESC, args_.out_desc.desc());

    // bitmask desc
    bwd_op_.SetOpConstParamDesc(CUDNN_PARAM_ACTIVATION_BITMASK_DESC,
                                args_.bitmask_desc.desc());

    // activation desc
    bwd_op_.SetOpConstParamDesc(CUDNN_PARAM_ACTIVATION_DESC,
                                args_.activation_desc.desc());

    // others
    bwd_op_.SetOpConstParamAttr(CUDNN_PARAM_BN_MODE,
                                CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
  }

  bool fuse_add_ = false;
  bool has_shortcut_ = false;
  size_t fwd_workspace_byte_;
  size_t bwd_workspace_byte_;
  ScaleBiasAddReluArgs<T> args_;
  CudnnFusionOp fwd_op_;
  CudnnFusionOp bwd_op_;
};
#endif
}  // namespace operators
}  // namespace paddle
