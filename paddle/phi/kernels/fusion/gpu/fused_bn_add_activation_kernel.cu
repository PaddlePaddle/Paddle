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
#include "paddle/phi/kernels/fused_bn_add_activation_kernel.h"

COMMON_DECLARE_bool(cudnn_batchnorm_spatial_persistent);

namespace phi {
namespace fusion {

template <typename T>
using CudnnDataType = phi::backends::gpu::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;

template <typename T, typename Context>
void FusedBatchNormAddActKernel(const Context &dev_ctx,
                                const DenseTensor &x,
                                const DenseTensor &z,
                                const DenseTensor &scale,
                                const DenseTensor &bias,
                                const DenseTensor &mean,
                                const DenseTensor &variance,
                                float momentum,
                                float epsilon,
                                const std::string &act_type,
                                DenseTensor *y,
                                DenseTensor *mean_out,
                                DenseTensor *variance_out,
                                DenseTensor *saved_mean,
                                DenseTensor *saved_variance,
                                DenseTensor *reserve_space) {
#if defined(PADDLE_WITH_CUDA) and CUDNN_VERSION >= 7401
  bool is_gpu_place = dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU;
  PADDLE_ENFORCE_EQ(
      is_gpu_place,
      true,
      common::errors::PreconditionNotMet("It must use CUDAPlace."));

  double epsilon1 = static_cast<double>(epsilon);
  if (epsilon1 <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
    LOG(ERROR) << "Provided epsilon is smaller than "
               << "CUDNN_BN_MIN_EPSILON. Setting it to "
               << "CUDNN_BN_MIN_EPSILON instead.";
  }
  epsilon1 = std::max(static_cast<double>(epsilon1), CUDNN_BN_MIN_EPSILON);

  // Get the size for each dimension.
  // NHWC [batch_size, in_height, in_width, in_channels]
  const auto &in_dims = x.dims();

  dev_ctx.template Alloc<BatchNormParamType<T>>(
      mean_out, mean_out->numel() * sizeof(BatchNormParamType<T>));
  dev_ctx.template Alloc<BatchNormParamType<T>>(
      variance_out, variance_out->numel() * sizeof(BatchNormParamType<T>));

  dev_ctx.template Alloc<BatchNormParamType<T>>(
      saved_mean, saved_mean->numel() * sizeof(BatchNormParamType<T>));
  dev_ctx.template Alloc<BatchNormParamType<T>>(
      saved_variance, saved_variance->numel() * sizeof(BatchNormParamType<T>));

  dev_ctx.template Alloc<T>(y, y->numel() * sizeof(T));

  int N, C, H, W, D;
  const DataLayout data_layout = DataLayout::kNHWC;
  phi::funcs::ExtractNCWHD(in_dims, data_layout, &N, &C, &H, &W, &D);

  // ------------------- cudnn descriptors ---------------------
  auto handle = dev_ctx.cudnn_handle();
  cudnnTensorDescriptor_t data_desc_;
  cudnnTensorDescriptor_t bn_param_desc_;
  cudnnBatchNormMode_t mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&data_desc_));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&bn_param_desc_));

  std::vector<int> dims = {N, C, H, W, D};
  std::vector<int> strides = {H * W * D * C, 1, W * D * C, D * C, C};

  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
      data_desc_,
      CudnnDataType<T>::type,
      in_dims.size() > 3 ? in_dims.size() : 4,
      dims.data(),
      strides.data()));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnDeriveBNTensorDescriptor(
      bn_param_desc_, data_desc_, mode_));

  double this_factor = 1. - momentum;
  cudnnBatchNormOps_t bnOps_ = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
  phi::backends::gpu::ScopedActivationDescriptor scope_act_desc;
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
  PADDLE_ENFORCE_NOT_NULL(
      reserve_space,
      common::errors::NotFound(
          "The argument ReserveSpace of batch_norm op is not found."));

  // --------------- cudnn batchnorm workspace ---------------
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
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
      phi::dynload::cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
          /*handle=*/handle,
          /*mode=*/mode_,
          /*bnOps=*/bnOps_,
          /*activationDesc=*/activation_desc_,
          /*xDesc=*/data_desc_,
          /*sizeInBytes=*/&reserve_space_size));

  reserve_space->Resize(
      {static_cast<int64_t>((reserve_space_size + phi::SizeOf(x.dtype()) - 1) /
                            phi::SizeOf(x.dtype()))});
  reserve_space_ptr = dev_ctx.template Alloc<T>(
      reserve_space, reserve_space->numel() * sizeof(T));
  workspace_tensor.Resize({static_cast<int64_t>(
      (workspace_size + phi::SizeOf(x.dtype()) - 1) / phi::SizeOf(x.dtype()))});
  workspace_ptr = dev_ctx.template Alloc<T>(
      &workspace_tensor, workspace_tensor.numel() * sizeof(T));

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnBatchNormalizationForwardTrainingEx(
          handle,
          mode_,
          bnOps_,
          CudnnDataType<T>::kOne(),
          CudnnDataType<T>::kZero(),
          data_desc_,
          x.template data<T>(),
          data_desc_,
          z.template data<T>(),
          data_desc_,
          y->template data<T>(),
          bn_param_desc_,
          scale.template data<BatchNormParamType<T>>(),
          bias.template data<BatchNormParamType<T>>(),
          this_factor,
          dev_ctx.template Alloc<BatchNormParamType<T>>(
              mean_out, mean_out->numel() * sizeof(BatchNormParamType<T>)),
          dev_ctx.template Alloc<BatchNormParamType<T>>(
              variance_out,
              variance_out->numel() * sizeof(BatchNormParamType<T>)),
          epsilon1,
          dev_ctx.template Alloc<BatchNormParamType<T>>(
              saved_mean, saved_mean->numel() * sizeof(BatchNormParamType<T>)),
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
      phi::dynload::cudnnDestroyTensorDescriptor(data_desc_));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(bn_param_desc_));
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "The fused_bn_add_activation operator is not supported on GPU "
      "when CUDNN version < 7.4.1"));
#endif
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_bn_add_activation,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedBatchNormAddActKernel,
                   phi::dtype::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
}
