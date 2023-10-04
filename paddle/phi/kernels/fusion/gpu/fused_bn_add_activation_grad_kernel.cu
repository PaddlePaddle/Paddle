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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/norm_utils.h"
#include "paddle/phi/kernels/fused_bn_add_activation_grad_kernel.h"

PHI_DECLARE_bool(cudnn_batchnorm_spatial_persistent);

namespace phi {
namespace fusion {

template <typename T>
using CudnnDataType = phi::backends::gpu::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;

template <typename T, typename Context>
void FusedBatchNormAddActGradKernel(const Context &dev_ctx,
                                    const DenseTensor &x,
                                    const DenseTensor &y,
                                    const DenseTensor &y_grad,
                                    const DenseTensor &scale,
                                    const DenseTensor &bias,
                                    const DenseTensor &saved_mean,
                                    const DenseTensor &saved_variance,
                                    const DenseTensor &reserve_space,
                                    float momentum,
                                    float epsilon,
                                    const std::string &act_type,
                                    DenseTensor *x_grad,
                                    DenseTensor *z_grad,
                                    DenseTensor *scale_grad,
                                    DenseTensor *bias_grad) {
#if CUDNN_VERSION < 7401
  PADDLE_THROW(phi::errors::Unimplemented(
      "The fused_bn_add_activation operator is not supported on GPU "
      "when CUDNN version < 7.4.1"));
#endif
  bool is_gpu_place = dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU;
  PADDLE_ENFORCE_EQ(is_gpu_place,
                    true,
                    phi::errors::PreconditionNotMet("It must use CUDAPlace."));

  const auto *d_y = &y;
  const auto &in_dims = x.dims();

  int N, C, H, W, D;
  const DataLayout data_layout = DataLayout::kNHWC;
  phi::funcs::ExtractNCWHD(in_dims, data_layout, &N, &C, &H, &W, &D);

  dev_ctx.template Alloc<T>(x_grad);
  dev_ctx.template Alloc<T>(z_grad);
  PADDLE_ENFORCE_EQ(
      scale_grad && bias_grad,
      true,
      phi::errors::PreconditionNotMet(
          "Both the scale grad and the bias grad must not be null."));
  dev_ctx.template Alloc<BatchNormParamType<T>>(scale_grad);
  dev_ctx.template Alloc<BatchNormParamType<T>>(bias_grad);
  PADDLE_ENFORCE_EQ(
      scale.dims().size(),
      1UL,
      phi::errors::PreconditionNotMet("The scale only has one dimension."));
  PADDLE_ENFORCE_EQ(
      scale.dims()[0],
      C,
      phi::errors::PreconditionNotMet(
          "The size of scale is equal to the channel of Input(X)."));

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
  if (epsilon <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
    LOG(ERROR) << "Provided epsilon is smaller than "
               << "CUDNN_BN_MIN_EPSILON. Setting it to "
               << "CUDNN_BN_MIN_EPSILON instead.";
  }
  epsilon = std::max(static_cast<double>(epsilon), CUDNN_BN_MIN_EPSILON);

  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
      data_desc_,
      CudnnDataType<T>::type,
      in_dims.size() > 3 ? in_dims.size() : 4,
      dims.data(),
      strides.data()));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnDeriveBNTensorDescriptor(
      bn_param_desc_, data_desc_, mode_));

  const auto *saved_mean_data =
      saved_mean.template data<BatchNormParamType<T>>();
  const auto *saved_var_data =
      saved_variance.template data<BatchNormParamType<T>>();

  size_t workspace_size = 0;
  void *workspace_ptr = nullptr;
  phi::DenseTensor workspace_tensor;
  auto reserve_space_size = reserve_space.memory_size();
  cudnnBatchNormOps_t bnOps_ = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
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
          /*dzDesc=*/data_desc_,
          /*dxDesc=*/data_desc_,
          /*bnScaleBiasMeanVarDesc=*/bn_param_desc_,
          /*activationDesc=*/activation_desc_,
          /*sizeInBytes=*/&workspace_size));

  workspace_tensor.Resize({static_cast<int64_t>(workspace_size)});
  workspace_ptr = dev_ctx.template Alloc<T>(&workspace_tensor);

  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBatchNormalizationBackwardEx(
      /*handle=*/dev_ctx.cudnn_handle(),
      /*mode=*/mode_,
      /*bnOps=*/bnOps_,
      /*alphaDataDiff=*/CudnnDataType<T>::kOne(),
      /*betaDataDiff=*/CudnnDataType<T>::kZero(),
      /*alphaParamDiff=*/CudnnDataType<T>::kOne(),
      /*betaParamDiff=*/CudnnDataType<T>::kZero(),
      /*xDesc=*/data_desc_,
      /*xData=*/x.template data<T>(),
      /*yDesc=*/data_desc_,
      /*yData=*/y.template data<T>(),
      /*dyDesc=*/data_desc_,
      /*dyData=*/d_y->template data<T>(),
      /*dzDesc=*/data_desc_,
      /*dzData=*/z_grad->template data<T>(),
      /*dxDesc=*/data_desc_,
      /*dxData=*/x_grad->template data<T>(),
      /*dBnScaleBiasDesc=*/bn_param_desc_,
      /*bnScaleData=*/scale.template data<BatchNormParamType<T>>(),
      /*bnBiasData=*/bias.template data<BatchNormParamType<T>>(),
      /*dBnScaleData=*/scale_grad->template data<BatchNormParamType<T>>(),
      /*dBnBiasData=*/bias_grad->template data<BatchNormParamType<T>>(),
      /*epsilon=*/epsilon,
      /*savedMean=*/saved_mean_data,
      /*savedInvVariance=*/saved_var_data,
      /*activationDesmc=*/activation_desc_,
      /*workspace=*/workspace_ptr,
      /*workSpaceSizeInBytes=*/workspace_size,
      /*reserveSpace=*/const_cast<T *>(reserve_space.template data<T>()),
      /*reserveSpaceSizeInBytes=*/reserve_space_size));

  // clean when exit.
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(data_desc_));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(bn_param_desc_));
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_bn_add_activation_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedBatchNormAddActGradKernel,
                   phi::dtype::float16) {
  kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
}
