// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <cfloat>
#include <string>
#include <vector>
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/scope_guard.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/fused/fused_bn_add_activation_op.h"
#include "paddle/fluid/operators/fused/fused_bn_kernel.h"
#include "paddle/fluid/operators/norm_utils.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/kernels/funcs/math_function.h"

DECLARE_bool(cudnn_batchnorm_spatial_persistent);

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;

template <typename T>
class FusedBatchNormAddActKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("It must use CUDAPlace."));
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    float momentum = ctx.Attr<float>("momentum");
    std::string act_type = ctx.Attr<std::string>("act_type");
    DataLayout data_layout =
        framework::StringToDataLayout(ctx.Attr<std::string>("data_layout"));

    if (epsilon <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
      LOG(ERROR) << "Provided epsilon is smaller than "
                 << "CUDNN_BN_MIN_EPSILON. Setting it to "
                 << "CUDNN_BN_MIN_EPSILON instead.";
    }
    epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);

    // Get the size for each dimension.
    // NHWC [batch_size, in_height, in_width, in_channels]
    const auto *x = ctx.Input<Tensor>("X");
    const auto *z = ctx.Input<Tensor>("Z");
    auto *y = ctx.Output<Tensor>("Y");
    int64_t numel = x->numel();

    const T *x_data = x->template data<T>();
    const T *z_data = z->template data<T>();
    T *y_data = y->template mutable_data<T>(ctx.GetPlace());

    const auto &in_dims = x->dims();

    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *scale_data = scale->template data<BatchNormParamType<T>>();
    const auto *bias = ctx.Input<Tensor>("Bias");
    const auto *bias_data = bias->template data<BatchNormParamType<T>>();

    auto *mean_out = ctx.Output<Tensor>("MeanOut");
    auto *variance_out = ctx.Output<Tensor>("VarianceOut");
    auto *mean_out_data =
        mean_out->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    auto *variance_out_data =
        variance_out->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());

    auto *saved_mean = ctx.Output<Tensor>("SavedMean");
    auto *saved_variance = ctx.Output<Tensor>("SavedVariance");
    auto *saved_mean_data =
        saved_mean->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    auto *saved_variance_data =
        saved_variance->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());

    int N, C, H, W, D;
    ExtractNCWHD(in_dims, data_layout, &N, &C, &H, &W, &D);

    // ------------------- cudnn descriptors ---------------------
    auto handle = dev_ctx.cudnn_handle();
    cudnnTensorDescriptor_t data_desc_;
    cudnnTensorDescriptor_t bn_param_desc_;
    cudnnBatchNormMode_t mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&data_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&bn_param_desc_));

    std::vector<int> dims, strides;
    if (data_layout == DataLayout::kNCHW) {
      dims = {N, C, H, W, D};
      strides = {C * H * W * D, H * W * D, W * D, D, 1};
    } else {
      dims = {N, C, H, W, D};
      strides = {H * W * D * C, 1, W * D * C, D * C, C};
    }

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        data_desc_, CudnnDataType<T>::type,
        in_dims.size() > 3 ? in_dims.size() : 4, dims.data(), strides.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnDeriveBNTensorDescriptor(
        bn_param_desc_, data_desc_, mode_));

    DEFINE_PADDLE_SCOPE_GUARD([=] {
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnDestroyTensorDescriptor(data_desc_));
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnDestroyTensorDescriptor(bn_param_desc_));
    });

    double this_factor = 1. - momentum;
    platform::ScopedActivationDescriptor scope_act_desc;
    cudnnBatchNormOps_t bnOps_;
    cudnnActivationDescriptor_t activation_desc_;
    bool fuse_add_act;
    if (data_layout == DataLayout::kNHWC &&
        std::is_same<T, platform::float16>::value) {
      bnOps_ = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
      activation_desc_ = scope_act_desc.descriptor<T>(act_type);
      fuse_add_act = true;
    } else {
      PADDLE_ENFORCE_EQ(
          act_type, "relu",
          phi::errors::InvalidArgument("Only bn + add + relu is supported."));
      bnOps_ = CUDNN_BATCHNORM_OPS_BN;
      activation_desc_ = nullptr;
      fuse_add_act = false;
    }

    // Create reserve space and workspace for batch norm.
    // Create tensor for each batchnorm op, it will be used in the
    // backward. Thus this tensor shouldn't be temp.
    size_t reserve_space_size = 0;
    auto *reserve_space = ctx.Output<Tensor>("ReserveSpace");
    PADDLE_ENFORCE_NOT_NULL(
        reserve_space,
        platform::errors::NotFound(
            "The argument ReserveSpace of batch_norm op is not found."));

    // -------------- cudnn batchnorm reserve space --------------
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
            /*handle=*/handle,
            /*mode=*/mode_,
            /*bnOps=*/bnOps_,
            /*activationDesc=*/activation_desc_,
            /*xDesc=*/data_desc_,
            /*sizeInBytes=*/&reserve_space_size));

    reserve_space->Resize({static_cast<int64_t>(numel)});
    void *reserve_space_ptr =
        reserve_space->mutable_data<uint8_t>(ctx.GetPlace());
    VLOG(1) << "ReserveSpaceSize = " << reserve_space_size;
    auto *mask = ctx.Output<framework::Tensor>("Mask");
    void *mask_ptr = nullptr;
    if (!fuse_add_act) {
      auto mask_size = GetBNMaskSpaceSize(N, C, H, W * D);
      mask->Resize({static_cast<int64_t>(mask_size)});
      mask_ptr = mask->mutable_data<uint8_t>(ctx.GetPlace());
    } else {
      mask->clear();
    }

    if (reserve_space_size == 0 && !fuse_add_act) {
      if (std::is_same<T, float>::value) {
        bool success = TryLaunchFusedNCHWFP32BNTrainingKernel(
            dev_ctx, reinterpret_cast<const float *>(x_data),
            reinterpret_cast<const float *>(z_data),
            reinterpret_cast<const float *>(scale_data),
            reinterpret_cast<const float *>(bias_data),
            reinterpret_cast<float *>(y_data),
            reinterpret_cast<float *>(saved_mean_data),
            reinterpret_cast<float *>(saved_variance_data),
            reinterpret_cast<float *>(mean_out_data),
            reinterpret_cast<float *>(variance_out_data), mask_ptr, N, C, H,
            W * D, this_factor, epsilon, true);
        if (success) {
          VLOG(1) << "Launch fast FP32 NCHW kernel";
          return;
        }
      }
    }

    size_t workspace_size = 0;
    void *workspace_ptr = nullptr;
    Tensor workspace_tensor;
    // --------------- cudnn batchnorm workspace ---------------
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::
            cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
                /*handle=*/handle,
                /*mode=*/mode_,
                /*bnOps=*/bnOps_,
                /*xDesc=*/data_desc_,
                /*zDesc=*/data_desc_,
                /*yDesc=*/data_desc_,
                /*bnScaleBiasMeanVarDesc=*/bn_param_desc_,
                /*activationDesc=*/activation_desc_,
                /*sizeInBytes=*/&workspace_size));
    workspace_ptr = workspace_tensor.mutable_data(ctx.GetPlace(), x->dtype(),
                                                  workspace_size);

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnBatchNormalizationForwardTrainingEx(
            handle, mode_, bnOps_, CudnnDataType<T>::kOne(),
            CudnnDataType<T>::kZero(), data_desc_, x_data, data_desc_, z_data,
            data_desc_, y_data, bn_param_desc_, scale_data, bias_data,
            this_factor, mean_out_data, variance_out_data, epsilon,
            saved_mean_data, saved_variance_data, activation_desc_,
            workspace_ptr, workspace_size, reserve_space_ptr,
            reserve_space_size));
    if (!fuse_add_act) {
      LaunchMaskedAddReluFwdKernel<T>(dev_ctx, y_data, z_data, y_data, mask_ptr,
                                      numel);
    }
  }
};

template <typename T>
class FusedBatchNormAddActGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("It must use CUDAPlace."));
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    std::string act_type = ctx.Attr<std::string>("act_type");
    DataLayout data_layout =
        framework::StringToDataLayout(ctx.Attr<std::string>("data_layout"));

    const auto *x = ctx.Input<Tensor>("X");
    const auto *x_data = x->template data<T>();
    const auto *y = ctx.Input<Tensor>("Y");
    const auto *y_data = y->template data<T>();

    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_z = ctx.Output<Tensor>(framework::GradVarName("Z"));

    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const T *dy_data = d_y->template data<T>();
    T *dx_data = d_x->mutable_data<T>(ctx.GetPlace());
    T *dz_data = d_z->mutable_data<T>(ctx.GetPlace());

    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *scale_data = scale->template data<BatchNormParamType<T>>();

    const auto *bias = ctx.Input<Tensor>("Bias");
    const auto *bias_data = bias->template data<BatchNormParamType<T>>();
    const auto *reserve_space = ctx.Input<Tensor>("ReserveSpace");
    void *reserve_space_ptr =
        const_cast<uint8_t *>(reserve_space->data<uint8_t>());

    const auto &in_dims = x->dims();
    int64_t numel = x->numel();

    int N, C, H, W, D;
    ExtractNCWHD(in_dims, data_layout, &N, &C, &H, &W, &D);

    // init output
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    PADDLE_ENFORCE_EQ(
        d_scale && d_bias, true,
        platform::errors::PreconditionNotMet(
            "Both the scale grad and the bias grad must not be null."));
    auto *dscale_data =
        d_scale->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    auto *dbias_data =
        d_bias->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    PADDLE_ENFORCE_EQ(scale->dims().size(), 1UL,
                      platform::errors::PreconditionNotMet(
                          "The scale only has one dimension."));
    PADDLE_ENFORCE_EQ(
        scale->dims()[0], C,
        platform::errors::PreconditionNotMet(
            "The size of scale is equal to the channel of Input(X)."));

    std::vector<int> dims, strides;
    if (data_layout == DataLayout::kNCHW) {
      dims = {N, C, H, W, D};
      strides = {C * H * W * D, H * W * D, W * D, D, 1};
    } else {
      dims = {N, C, H, W, D};
      strides = {H * W * D * C, 1, W * D * C, D * C, C};
    }
    // ------------------- cudnn descriptors ---------------------
    cudnnTensorDescriptor_t data_desc_;
    cudnnTensorDescriptor_t bn_param_desc_;
    cudnnBatchNormMode_t mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&data_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&bn_param_desc_));
    if (epsilon <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
      LOG(ERROR) << "Provided epsilon is smaller than "
                 << "CUDNN_BN_MIN_EPSILON. Setting it to "
                 << "CUDNN_BN_MIN_EPSILON instead.";
    }
    epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        data_desc_, CudnnDataType<T>::type,
        in_dims.size() > 3 ? in_dims.size() : 4, dims.data(), strides.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnDeriveBNTensorDescriptor(
        bn_param_desc_, data_desc_, mode_));

    const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
    const auto *saved_var = ctx.Input<Tensor>("SavedVariance");
    const auto *saved_mean_data =
        saved_mean->template data<BatchNormParamType<T>>();
    const auto *saved_var_data =
        saved_var->template data<BatchNormParamType<T>>();

    DEFINE_PADDLE_SCOPE_GUARD([=] {
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnDestroyTensorDescriptor(data_desc_));
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnDestroyTensorDescriptor(bn_param_desc_));
    });

    auto handle = dev_ctx.cudnn_handle();
    size_t workspace_size = 0;
    void *workspace_ptr = nullptr;
    Tensor workspace_tensor;
    auto reserve_space_size = reserve_space->memory_size();
    cudnnBatchNormOps_t bnOps_;
    platform::ScopedActivationDescriptor scope_act_desc;
    cudnnActivationDescriptor_t activation_desc_ =
        scope_act_desc.descriptor<T>(act_type);
    if (data_layout == DataLayout::kNHWC &&
        std::is_same<T, platform::float16>::value) {
      bnOps_ = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
    } else {
      auto *mask = ctx.Input<Tensor>("Mask");
      LaunchMaskedReluBwdKernel<T>(dev_ctx, dy_data, mask->data(), dz_data,
                                   numel);

      bnOps_ = CUDNN_BATCHNORM_OPS_BN;
      activation_desc_ = nullptr;
      dy_data = dz_data;
      dz_data = nullptr;
    }
    // --------------- cudnn batchnorm workspace ---------------
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetBatchNormalizationBackwardExWorkspaceSize(
            /*handle=*/handle,
            /*mode=*/mode_,
            /*bnOps=*/bnOps_,
            /*xDesc=*/data_desc_,
            /*yDesc=*/data_desc_,
            /*dyDesc=*/data_desc_,
            /*dzDesc=*/data_desc_,
            /*dxDesc=*/data_desc_,
            /*bnScaleBiasMeanVarDesc=*/bn_param_desc_,
            /*activationDesc=*/activation_desc_,
            /*sizeInBytes=*/&workspace_size));

    workspace_ptr = workspace_tensor.mutable_data(ctx.GetPlace(), x->dtype(),
                                                  workspace_size);
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnBatchNormalizationBackwardEx(
            /*handle=*/handle,
            /*mode=*/mode_,
            /*bnOps=*/bnOps_,
            /*alphaDataDiff=*/CudnnDataType<T>::kOne(),
            /*betaDataDiff=*/CudnnDataType<T>::kZero(),
            /*alphaParamDiff=*/CudnnDataType<T>::kOne(),
            /*betaParamDiff=*/CudnnDataType<T>::kZero(),
            /*xDesc=*/data_desc_,
            /*xData=*/x_data,
            /*yDesc=*/data_desc_,
            /*yData=*/y_data,
            /*dyDesc=*/data_desc_,
            /*dyData=*/dy_data,
            /*dzDesc=*/data_desc_,
            /*dzData=*/dz_data,
            /*dxDesc=*/data_desc_,
            /*dxData=*/dx_data,
            /*dBnScaleBiasDesc=*/bn_param_desc_,
            /*bnScaleData=*/scale_data,
            /*bnBiasData=*/bias_data,
            /*dBnScaleData=*/dscale_data,
            /*dBnBiasData=*/dbias_data,
            /*epsilon=*/epsilon,
            /*savedMean=*/saved_mean_data,
            /*savedInvVariance=*/saved_var_data,
            /*activationDesmc=*/activation_desc_,
            /*workspace=*/workspace_ptr,
            /*workSpaceSizeInBytes=*/workspace_size,
            /*reserveSpace=*/reserve_space_ptr,
            /*reserveSpaceSizeInBytes=*/reserve_space_size));
  }
};

}  // namespace operators
}  // namespace paddle

#if CUDNN_VERSION >= 7401
namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    fused_bn_add_activation,
    ops::FusedBatchNormAddActKernel<plat::CUDADeviceContext, plat::float16>,
    ops::FusedBatchNormAddActKernel<plat::CUDADeviceContext, float>);
REGISTER_OP_CUDA_KERNEL(
    fused_bn_add_activation_grad,
    ops::FusedBatchNormAddActGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::FusedBatchNormAddActGradKernel<plat::CUDADeviceContext, float>);
#endif
