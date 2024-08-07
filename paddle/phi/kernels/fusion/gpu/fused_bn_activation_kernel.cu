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

COMMON_DECLARE_bool(cudnn_batchnorm_spatial_persistent);

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedBatchNormActKernel(const Context &dev_ctx,
                             const DenseTensor &x,
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

  if (epsilon1 <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
    LOG(ERROR) << "Provided epsilon is smaller than "
               << "CUDNN_BN_MIN_EPSILON. Setting it to "
               << "CUDNN_BN_MIN_EPSILON instead.";
  }
  epsilon1 = std::max(epsilon1, CUDNN_BN_MIN_EPSILON);

  // Get the size for each dimension.
  // NHWC [batch_size, in_height, in_width, in_channels]
  const auto &x_dims = x.dims();
  PADDLE_ENFORCE_EQ(x_dims.size() >= 2 && x_dims.size() <= 5,
                    true,
                    common::errors::PreconditionNotMet(
                        "The Input dim size should be between 2 and 5"));

  // Run training mode.
  // obtain running mean and running inv var, and see if we need to
  // initialize them.
  dev_ctx.template Alloc<BatchNormParamType>(mean_out);
  dev_ctx.template Alloc<BatchNormParamType>(variance_out);

  dev_ctx.template Alloc<BatchNormParamType>(saved_mean);
  dev_ctx.template Alloc<BatchNormParamType>(saved_variance);

  dev_ctx.template Alloc<T>(y);

  int N, C, H, W, D;
  const DataLayout data_layout = phi::DataLayout::kNHWC;
  phi::funcs::ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);

  if ((N * H * W * D) == 1) {
    // Only 1 element in normalization dimension,
    // skip the batch norm calculation, let y = act(x).
    auto x_v = phi::EigenVector<T>::Flatten(x);
    auto y_v = phi::EigenVector<T>::Flatten(*y);
    auto &dev = *dev_ctx.eigen_device();
    if (act_type == "relu") {
      phi::funcs::ReluCUDAFunctor<T>()(dev, x_v, y_v);
    } else {
      PADDLE_THROW(
          common::errors::Unimplemented("Unsupported activation type"));
    }
    return;
  }

  // ------------------- cudnn descriptors ---------------------
  auto handle = dev_ctx.cudnn_handle();
  cudnnTensorDescriptor_t data_desc_;
  cudnnTensorDescriptor_t bn_param_desc_;
  cudnnBatchNormMode_t mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&data_desc_));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&bn_param_desc_));

  VLOG(3) << "Setting descriptors.";
  std::vector<int> dims = {N, C, H, W, D};
  std::vector<int> strides = {H * W * D * C, 1, W * D * C, D * C, C};

  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
      data_desc_,
      CudnnDataType::type,
      x_dims.size() > 3 ? x_dims.size() : 4,
      dims.data(),
      strides.data()));

  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnDeriveBNTensorDescriptor(
      bn_param_desc_, data_desc_, mode_));

  double this_factor = 1. - momentum;
  cudnnBatchNormOps_t bnOps_ = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
  phi::backends::gpu::ScopedActivationDescriptor scope_act_desc;
  cudnnActivationDescriptor_t activation_desc_ =
      scope_act_desc.descriptor<T>(act_type);
  size_t workspace_size = 0;
  size_t reserve_space_size = 0;
  void *reserve_space_ptr = nullptr;
  void *workspace_ptr = nullptr;
  phi::DenseTensor workspace_tensor;

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
          /*zDesc=*/nullptr,
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
  reserve_space_ptr = dev_ctx.template Alloc<T>(reserve_space);
  workspace_tensor.Resize({static_cast<int64_t>(
      (workspace_size + phi::SizeOf(x.dtype()) - 1) / phi::SizeOf(x.dtype()))});
  workspace_ptr = dev_ctx.template Alloc<T>(&workspace_tensor);

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnBatchNormalizationForwardTrainingEx(
          handle,
          mode_,
          bnOps_,
          CudnnDataType::kOne(),
          CudnnDataType::kZero(),
          data_desc_,
          x.template data<T>(),
          nullptr,
          nullptr,
          data_desc_,
          y->template data<T>(),
          bn_param_desc_,
          scale.template data<BatchNormParamType>(),
          bias.template data<BatchNormParamType>(),
          this_factor,
          dev_ctx.template Alloc<BatchNormParamType>(mean_out),
          dev_ctx.template Alloc<BatchNormParamType>(variance_out),
          epsilon1,
          dev_ctx.template Alloc<BatchNormParamType>(saved_mean),
          dev_ctx.template Alloc<BatchNormParamType>(saved_variance),
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
      "The fused_batch_norm_act operator is not supported on GPU "
      "when CUDNN version < 7.4.1"));
#endif
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_batch_norm_act,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedBatchNormActKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
  }
}
