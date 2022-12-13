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
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/fused/fused_bn_add_activation_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/norm_utils.h"

DECLARE_bool(cudnn_batchnorm_spatial_persistent);

namespace paddle {
namespace operators {
template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;

template <typename T>
class FusedBatchNormAddActKernel<phi::GPUContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()),
        true,
        platform::errors::PreconditionNotMet("It must use CUDAPlace."));
    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    float momentum = ctx.Attr<float>("momentum");
    std::string act_type = ctx.Attr<std::string>("act_type");

    if (epsilon <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
      LOG(ERROR) << "Provided epsilon is smaller than "
                 << "CUDNN_BN_MIN_EPSILON. Setting it to "
                 << "CUDNN_BN_MIN_EPSILON instead.";
    }
    epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);

    // Get the size for each dimension.
    // NHWC [batch_size, in_height, in_width, in_channels]
    const auto *x = ctx.Input<phi::DenseTensor>("X");
    const auto *z = ctx.Input<phi::DenseTensor>("Z");
    const auto &in_dims = x->dims();

    const auto *scale = ctx.Input<phi::DenseTensor>("Scale");
    const auto *bias = ctx.Input<phi::DenseTensor>("Bias");

    auto *mean_out = ctx.Output<phi::DenseTensor>("MeanOut");
    auto *variance_out = ctx.Output<phi::DenseTensor>("VarianceOut");
    dev_ctx.Alloc<BatchNormParamType<T>>(
        mean_out, mean_out->numel() * sizeof(BatchNormParamType<T>));
    dev_ctx.Alloc<BatchNormParamType<T>>(
        variance_out, variance_out->numel() * sizeof(BatchNormParamType<T>));

    auto *saved_mean = ctx.Output<phi::DenseTensor>("SavedMean");
    auto *saved_variance = ctx.Output<phi::DenseTensor>("SavedVariance");
    dev_ctx.Alloc<BatchNormParamType<T>>(
        saved_mean, saved_mean->numel() * sizeof(BatchNormParamType<T>));
    dev_ctx.Alloc<BatchNormParamType<T>>(
        saved_variance,
        saved_variance->numel() * sizeof(BatchNormParamType<T>));

    auto *y = ctx.Output<phi::DenseTensor>("Y");
    dev_ctx.Alloc<T>(y, y->numel() * sizeof(T));

    int N, C, H, W, D;
    const DataLayout data_layout = DataLayout::kNHWC;
    phi::funcs::ExtractNCWHD(in_dims, data_layout, &N, &C, &H, &W, &D);

    // ------------------- cudnn descriptors ---------------------
    auto handle = dev_ctx.cudnn_handle();
    cudnnTensorDescriptor_t data_desc_;
    cudnnTensorDescriptor_t bn_param_desc_;
    cudnnBatchNormMode_t mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&data_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&bn_param_desc_));

    std::vector<int> dims = {N, C, H, W, D};
    std::vector<int> strides = {H * W * D * C, 1, W * D * C, D * C, C};

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        data_desc_,
        CudnnDataType<T>::type,
        in_dims.size() > 3 ? in_dims.size() : 4,
        dims.data(),
        strides.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnDeriveBNTensorDescriptor(
        bn_param_desc_, data_desc_, mode_));

    double this_factor = 1. - momentum;
    cudnnBatchNormOps_t bnOps_ = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
    platform::ScopedActivationDescriptor scope_act_desc;
    cudnnActivationDescriptor_t activation_desc_ =
        scope_act_desc.descriptor<T>(act_type);
    size_t workspace_size = 0;
    size_t reserve_space_size = 0;
    void *reserve_space_ptr = nullptr;
    void *workspace_ptr = nullptr;
    phi::DenseTensor workspace_tensor;
    // Create reserve space and workspace for batch norm.
    // Create tensor for each batchnorm op, it will be used in the
    // backward. Thus this tensor shouldn't be temp.
    auto *reserve_space = ctx.Output<phi::DenseTensor>("ReserveSpace");
    PADDLE_ENFORCE_NOT_NULL(
        reserve_space,
        platform::errors::NotFound(
            "The argument ReserveSpace of batch_norm op is not found."));

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

    // -------------- cudnn batchnorm reserve space --------------
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
            /*handle=*/handle,
            /*mode=*/mode_,
            /*bnOps=*/bnOps_,
            /*activationDesc=*/activation_desc_,
            /*xDesc=*/data_desc_,
            /*sizeInBytes=*/&reserve_space_size));

    reserve_space->Resize({static_cast<int64_t>(
        (reserve_space_size + experimental::SizeOf(x->dtype()) - 1) /
        experimental::SizeOf(x->dtype()))});
    reserve_space_ptr =
        dev_ctx.Alloc<T>(reserve_space, reserve_space->numel() * sizeof(T));
    workspace_tensor.Resize({static_cast<int64_t>(
        (workspace_size + experimental::SizeOf(x->dtype()) - 1) /
        experimental::SizeOf(x->dtype()))});
    workspace_ptr = dev_ctx.Alloc<T>(&workspace_tensor,
                                     workspace_tensor.numel() * sizeof(T));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnBatchNormalizationForwardTrainingEx(
            handle,
            mode_,
            bnOps_,
            CudnnDataType<T>::kOne(),
            CudnnDataType<T>::kZero(),
            data_desc_,
            x->template data<T>(),
            data_desc_,
            z->template data<T>(),
            data_desc_,
            y->template data<T>(),
            bn_param_desc_,
            scale->template data<BatchNormParamType<T>>(),
            bias->template data<BatchNormParamType<T>>(),
            this_factor,
            dev_ctx.template Alloc<BatchNormParamType<T>>(
                mean_out, mean_out->numel() * sizeof(BatchNormParamType<T>)),
            dev_ctx.template Alloc<BatchNormParamType<T>>(
                variance_out,
                variance_out->numel() * sizeof(BatchNormParamType<T>)),
            epsilon,
            dev_ctx.template Alloc<BatchNormParamType<T>>(
                saved_mean,
                saved_mean->numel() * sizeof(BatchNormParamType<T>)),
            dev_ctx.template Alloc<BatchNormParamType<T>>(
                saved_variance,
                saved_variance->numel() * sizeof(BatchNormParamType<T>)),
            activation_desc_,
            workspace_ptr,
            workspace_size,
            reserve_space_ptr,
            reserve_space_size));

    // clean when exit.
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(data_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(bn_param_desc_));
  }
};

template <typename T>
class FusedBatchNormAddActGradKernel<phi::GPUContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()),
        true,
        platform::errors::PreconditionNotMet("It must use CUDAPlace."));
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    std::string act_type = ctx.Attr<std::string>("act_type");

    const auto *x = ctx.Input<phi::DenseTensor>("X");
    const auto *y = ctx.Input<phi::DenseTensor>("Y");
    const auto *d_y = ctx.Input<phi::DenseTensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<phi::DenseTensor>("Scale");
    const auto *bias = ctx.Input<phi::DenseTensor>("Bias");
    const auto *reserve_space = ctx.Input<phi::DenseTensor>("ReserveSpace");

    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
    const auto &in_dims = x->dims();

    int N, C, H, W, D;
    const DataLayout data_layout = DataLayout::kNHWC;
    phi::funcs::ExtractNCWHD(in_dims, data_layout, &N, &C, &H, &W, &D);

    // init output
    auto *d_x = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto *d_z = ctx.Output<phi::DenseTensor>(framework::GradVarName("Z"));
    auto *d_scale =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<phi::DenseTensor>(framework::GradVarName("Bias"));

    d_x->mutable_data<T>(ctx.GetPlace());
    d_z->mutable_data<T>(ctx.GetPlace());
    PADDLE_ENFORCE_EQ(
        d_scale && d_bias,
        true,
        platform::errors::PreconditionNotMet(
            "Both the scale grad and the bias grad must not be null."));
    d_scale->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    d_bias->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    PADDLE_ENFORCE_EQ(scale->dims().size(),
                      1UL,
                      platform::errors::PreconditionNotMet(
                          "The scale only has one dimension."));
    PADDLE_ENFORCE_EQ(
        scale->dims()[0],
        C,
        platform::errors::PreconditionNotMet(
            "The size of scale is equal to the channel of Input(X)."));

    std::vector<int> dims = {N, C, H, W, D};
    std::vector<int> strides = {H * W * C * D, 1, W * D * C, D * C, C};
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
        data_desc_,
        CudnnDataType<T>::type,
        in_dims.size() > 3 ? in_dims.size() : 4,
        dims.data(),
        strides.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnDeriveBNTensorDescriptor(
        bn_param_desc_, data_desc_, mode_));

    const auto *saved_mean = ctx.Input<phi::DenseTensor>("SavedMean");
    const auto *saved_var = ctx.Input<phi::DenseTensor>("SavedVariance");
    const auto *saved_mean_data =
        saved_mean->template data<BatchNormParamType<T>>();
    const auto *saved_var_data =
        saved_var->template data<BatchNormParamType<T>>();

    size_t workspace_size = 0;
    void *workspace_ptr = nullptr;
    phi::DenseTensor workspace_tensor;
    auto reserve_space_size = reserve_space->memory_size();
    cudnnBatchNormOps_t bnOps_ = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
    platform::ScopedActivationDescriptor scope_act_desc;
    cudnnActivationDescriptor_t activation_desc_ =
        scope_act_desc.descriptor<T>(act_type);
    // --------------- cudnn batchnorm workspace ---------------
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetBatchNormalizationBackwardExWorkspaceSize(
            /*handle=*/dev_ctx.cudnn_handle(),
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

    workspace_ptr = workspace_tensor.mutable_data(
        ctx.GetPlace(), x->dtype(), workspace_size);
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnBatchNormalizationBackwardEx(
            /*handle=*/dev_ctx.cudnn_handle(),
            /*mode=*/mode_,
            /*bnOps=*/bnOps_,
            /*alphaDataDiff=*/CudnnDataType<T>::kOne(),
            /*betaDataDiff=*/CudnnDataType<T>::kZero(),
            /*alphaParamDiff=*/CudnnDataType<T>::kOne(),
            /*betaParamDiff=*/CudnnDataType<T>::kZero(),
            /*xDesc=*/data_desc_,
            /*xData=*/x->template data<T>(),
            /*yDesc=*/data_desc_,
            /*yData=*/y->template data<T>(),
            /*dyDesc=*/data_desc_,
            /*dyData=*/d_y->template data<T>(),
            /*dzDesc=*/data_desc_,
            /*dzData=*/d_z->template data<T>(),
            /*dxDesc=*/data_desc_,
            /*dxData=*/d_x->template data<T>(),
            /*dBnScaleBiasDesc=*/bn_param_desc_,
            /*bnScaleData=*/scale->template data<BatchNormParamType<T>>(),
            /*bnBiasData=*/bias->template data<BatchNormParamType<T>>(),
            /*dBnScaleData=*/d_scale->template data<BatchNormParamType<T>>(),
            /*dBnBiasData=*/d_bias->template data<BatchNormParamType<T>>(),
            /*epsilon=*/epsilon,
            /*savedMean=*/saved_mean_data,
            /*savedInvVariance=*/saved_var_data,
            /*activationDesmc=*/activation_desc_,
            /*workspace=*/workspace_ptr,
            /*workSpaceSizeInBytes=*/workspace_size,
            /*reserveSpace=*/const_cast<T *>(reserve_space->template data<T>()),
            /*reserveSpaceSizeInBytes=*/reserve_space_size));

    // clean when exit.
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(data_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(bn_param_desc_));
  }
};

}  // namespace operators
}  // namespace paddle

#if CUDNN_VERSION >= 7401
namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    fused_bn_add_activation,
    ops::FusedBatchNormAddActKernel<phi::GPUContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    fused_bn_add_activation_grad,
    ops::FusedBatchNormAddActGradKernel<phi::GPUContext, plat::float16>);
#endif
