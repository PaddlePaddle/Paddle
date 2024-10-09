// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif

#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/norm_utils.h"
#include "paddle/phi/kernels/fused_bn_activation_grad_kernel.h"

COMMON_DECLARE_bool(cudnn_batchnorm_spatial_persistent);

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedBatchNormActGradKernel(const Context &dev_ctx,
                                 const DenseTensor &x,
                                 const DenseTensor &scale,
                                 const DenseTensor &bias,
                                 const DenseTensor &y,
                                 const DenseTensor &saved_mean,
                                 const DenseTensor &saved_variance,
                                 const DenseTensor &reserve_space,
                                 const DenseTensor &y_grad,
                                 float momentum,
                                 float epsilon,
                                 const std::string &act_type,
                                 DenseTensor *x_grad,
                                 DenseTensor *scale_grad,
                                 DenseTensor *bias_grad) {
// Note(andsonder): Fused bn activation only used in the gpu place.
#if defined(PADDLE_WITH_CUDA) and CUDNN_VERSION >= 7401
  using CudnnDataType = phi::backends::gpu::CudnnDataType<T>;
  using BatchNormParamType = typename CudnnDataType::BatchNormParamType;
  bool is_gpu_place = dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU;
  PADDLE_ENFORCE_EQ(
      is_gpu_place,
      true,
      common::errors::PreconditionNotMet("It must use CUDAPlace."));
  double epsilon1 = static_cast<double>(epsilon);

  const auto *d_y = &y_grad;

  const auto &x_dims = x.dims();

  PADDLE_ENFORCE_EQ(x_dims.size() >= 2 && x_dims.size() <= 5,
                    true,
                    common::errors::PreconditionNotMet(
                        "The Input dim size should be between 2 and 5"));
  int N, C, H, W, D;
  const phi::DataLayout data_layout = phi::DataLayout::kNHWC;
  phi::funcs::ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);

  // init output
  auto *d_x = x_grad;
  auto *d_scale = scale_grad;
  auto *d_bias = bias_grad;

  dev_ctx.template Alloc<T>(d_x);
  PADDLE_ENFORCE_EQ(
      d_scale && d_bias,
      true,
      common::errors::PreconditionNotMet(
          "Both the scale grad and the bias grad must not be null."));
  dev_ctx.template Alloc<BatchNormParamType>(d_scale);
  dev_ctx.template Alloc<BatchNormParamType>(d_bias);
  PADDLE_ENFORCE_EQ(
      scale.dims().size(),
      1UL,
      common::errors::PreconditionNotMet("The scale only has one dimension."));
  PADDLE_ENFORCE_EQ(
      scale.dims()[0],
      C,
      common::errors::PreconditionNotMet(
          "The size of scale is equal to the channel of Input(X)."));

  if ((N * H * W * D) == 1) {
    if (act_type == "relu") {
      auto x_v = phi::EigenVector<T>::Flatten(x);
      auto y_v = phi::EigenVector<T>::Flatten(y);
      auto dx_v = phi::EigenVector<T>::Flatten(*d_x);
      auto dy_v = phi::EigenVector<T>::Flatten(*d_y);
      auto &dev = *dev_ctx.eigen_device();
      phi::funcs::ReluGradFunctor<T>()(dev, x_v, y_v, dy_v, dx_v);
    } else {
      PADDLE_THROW(
          common::errors::Unimplemented("Unsupported activation type"));
    }
    phi::funcs::SetConstant<phi::GPUContext, BatchNormParamType> functor;
    functor(dev_ctx, d_scale, static_cast<BatchNormParamType>(0));
    functor(dev_ctx, d_bias, static_cast<BatchNormParamType>(0));
    return;
  }

  std::vector<int> dims = {N, C, H, W, D};
  std::vector<int> strides = {H * W * C * D, 1, W * D * C, D * C, C};
  // ------------------- cudnn descriptors ---------------------
  cudnnTensorDescriptor_t data_desc_;
  cudnnTensorDescriptor_t bn_param_desc_;
  cudnnBatchNormMode_t mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&data_desc_));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&bn_param_desc_));
  if (epsilon1 <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
    LOG(ERROR) << "Provided epsilon is smaller than "
               << "CUDNN_BN_MIN_EPSILON. Setting it to "
               << "CUDNN_BN_MIN_EPSILON instead.";
  }
  epsilon1 = std::max(epsilon1, CUDNN_BN_MIN_EPSILON);

  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
      data_desc_,
      CudnnDataType::type,
      x_dims.size() > 3 ? x_dims.size() : 4,
      dims.data(),
      strides.data()));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnDeriveBNTensorDescriptor(
      bn_param_desc_, data_desc_, mode_));

  const auto *saved_mean_data = saved_mean.template data<BatchNormParamType>();
  const auto *saved_var_data =
      saved_variance.template data<BatchNormParamType>();

  size_t workspace_size = 0;
  void *workspace_ptr = nullptr;
  phi::DenseTensor workspace_tensor;
  auto reserve_space_size = reserve_space.memory_size();
  cudnnBatchNormOps_t bnOps_ = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
  phi::backends::gpu::ScopedActivationDescriptor scope_act_desc;
  cudnnActivationDescriptor_t activation_desc_ =
      scope_act_desc.descriptor<T>(act_type);
  // --------------- cudnn batchnorm workspace ---------------
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnGetBatchNormalizationBackwardExWorkspaceSize(
          /*handle=*/dev_ctx.cudnn_handle(),
          /*mode=*/mode_,
          /*bnOps=*/bnOps_,
          /*xDesc=*/data_desc_,
          /*yDesc=*/data_desc_,
          /*dyDesc=*/data_desc_,
          /*dzDesc=*/nullptr,
          /*dxDesc=*/data_desc_,
          /*bnScaleBiasMeanVarDesc=*/bn_param_desc_,
          /*activationDesc=*/activation_desc_,
          /*sizeInBytes=*/&workspace_size));

  workspace_tensor.Resize({static_cast<int64_t>(
      (workspace_size + phi::SizeOf(x.dtype()) - 1) / phi::SizeOf(x.dtype()))});
  workspace_ptr = dev_ctx.template Alloc<T>(&workspace_tensor);

  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBatchNormalizationBackwardEx(
      /*handle=*/dev_ctx.cudnn_handle(),
      /*mode=*/mode_,
      /*bnOps=*/bnOps_,
      /*alphaDataDiff=*/CudnnDataType::kOne(),
      /*betaDataDiff=*/CudnnDataType::kZero(),
      /*alphaParamDiff=*/CudnnDataType::kOne(),
      /*betaParamDiff=*/CudnnDataType::kZero(),
      /*xDesc=*/data_desc_,
      /*xData=*/x.template data<T>(),
      /*yDesc=*/data_desc_,
      /*yData=*/y.template data<T>(),
      /*dyDesc=*/data_desc_,
      /*dyData=*/d_y->template data<T>(),
      /*dzDesc=*/nullptr,
      /*dzData=*/nullptr,
      /*dxDesc=*/data_desc_,
      /*dxData=*/
      dev_ctx.template Alloc<T>(d_x, d_x->numel() * sizeof(T)),
      /*dBnScaleBiasDesc=*/bn_param_desc_,
      /*bnScaleData=*/scale.template data<BatchNormParamType>(),
      /*bnBiasData=*/bias.template data<BatchNormParamType>(),
      /*dBnScaleData=*/
      dev_ctx.template Alloc<BatchNormParamType>(d_scale),
      /*dBnBiasData=*/
      dev_ctx.template Alloc<BatchNormParamType>(d_bias),
      /*epsilon=*/epsilon1,
      /*savedMean=*/saved_mean_data,
      /*savedInvVariance=*/saved_var_data,
      /*activationDesc=*/activation_desc_,
      /*workspace=*/workspace_ptr,
      /*workSpaceSizeInBytes=*/workspace_size,
      /*reserveSpace=*/const_cast<T *>(reserve_space.template data<T>()),
      /*reserveSpaceSizeInBytes=*/reserve_space_size));

  // clean when exit.
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(data_desc_));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(bn_param_desc_));
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "The fused_batch_norm_act operator is not supported on GPU "
      "when CUDNN version < 7.4.1"));
#endif
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_batch_norm_act_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedBatchNormActGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}
