// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/instance_norm_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/norm_utils.h"
#include "paddle/phi/kernels/gpu/instance_norm_utils.h"

namespace phi {

template <typename T, typename Context>
void InstanceNormKernel(const Context &dev_ctx,
                        const DenseTensor &x,
                        const paddle::optional<DenseTensor> &scale,
                        const paddle::optional<DenseTensor> &bias,
                        float epsilon_f,
                        DenseTensor *y,
                        DenseTensor *saved_mean,
                        DenseTensor *saved_variance) {
  double epsilon = static_cast<double>(epsilon_f);
  auto &x_dims = x.dims();
  PADDLE_ENFORCE_GE(x_dims.size(),
                    2,
                    phi::errors::InvalidArgument(
                        "The `shape` in InstanceNormOp is invalid: "
                        "the size of X's dimensions must greater than "
                        "or equal to 2. But received: "
                        "the size of X's dimensions is [%d]",
                        x_dims.size()));
  PADDLE_ENFORCE_LE(x_dims.size(),
                    5,
                    phi::errors::InvalidArgument(
                        "The `shape` in InstanceNormOp is invalid: "
                        "the size of X's dimensions must smaller than"
                        "or equal to 5. But received: "
                        "the size of X's dimensions is [%d]",
                        x_dims.size()));
  int N, C, H, W, D;
  funcs::ExtractNCWHD(x_dims, DataLayout::kNCHW, &N, &C, &H, &W, &D);
  int NxC = N * C;
  DenseTensor x_tmp;
  x_tmp.ShareDataWith(x).Resize({1, NxC, H, W, D});
  dev_ctx.template Alloc<T>(y);

#ifdef PADDLE_WITH_HIP
  miopenTensorDescriptor_t data_desc_;
  miopenTensorDescriptor_t in_param_desc_;

  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::miopenCreateTensorDescriptor(&data_desc_));
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::miopenCreateTensorDescriptor(&in_param_desc_));
#else
  cudnnTensorDescriptor_t data_desc_;
  cudnnTensorDescriptor_t in_param_desc_;

  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::cudnnCreateTensorDescriptor(&data_desc_));
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::cudnnCreateTensorDescriptor(&in_param_desc_));
#endif
  if (epsilon <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
    LOG(ERROR) << "Provided epsilon is smaller than "
               << "CUDNN_BN_MIN_EPSILON. Setting it to "
               << "CUDNN_BN_MIN_EPSILON instead.";
  }
  epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);

  VLOG(3) << "Setting descriptors.";
  std::vector<int> dims;
  std::vector<int> strides;
  dims = {1, NxC, H, W, D};
  strides = {NxC * H * W * D, H * W * D, W * D, D, 1};

#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::miopenSetTensorDescriptor(
          data_desc_,
          CudnnDataType<T>::type,
          x_dims.size() > 3 ? x_dims.size() : 4,
          const_cast<int *>(dims.data()),
          const_cast<int *>(strides.data())));
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::miopenDeriveBNTensorDescriptor(
          in_param_desc_, data_desc_, miopenBNSpatial));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::cudnnSetTensorNdDescriptor(
          data_desc_,
          CudnnDataType<T>::type,
          x_dims.size() > 3 ? x_dims.size() : 4,
          dims.data(),
          strides.data()));
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::cudnnDeriveBNTensorDescriptor(
          in_param_desc_, data_desc_, CUDNN_BATCHNORM_SPATIAL));
#endif

  const auto scale_ptr = scale.get_ptr();
  const auto bias_ptr = bias.get_ptr();

  DenseTensor scale_tmp;
  scale_tmp.Resize({NxC});
  dev_ctx.template Alloc<T>(&scale_tmp);
  DenseTensor bias_tmp;
  bias_tmp.Resize({NxC});
  dev_ctx.template Alloc<T>(&bias_tmp);

  const int n = x.numel();
  const int block = 512;
  int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
  const int max_blocks = std::max(max_threads / block, 1);
  const int grid = std::min((NxC + block - 1) / block, max_blocks);

  phi::funcs::SetConstant<GPUContext, T> set_constant;
  if (scale_ptr) {
    repeat_param<T><<<grid, block, 0, dev_ctx.stream()>>>(
        scale_ptr->data<T>(), scale_tmp.data<T>(), N, C);
  } else {
    set_constant(dev_ctx, &scale_tmp, static_cast<T>(1));
  }
  if (bias_ptr) {
    repeat_param<T><<<grid, block, 0, dev_ctx.stream()>>>(
        bias_ptr->data<T>(), bias_tmp.data<T>(), N, C);
  } else {
    set_constant(dev_ctx, &bias_tmp, static_cast<T>(0));
  }

  auto handle = dev_ctx.cudnn_handle();

  phi::funcs::SetConstant<GPUContext, BatchNormParamType<T>> functor;
  dev_ctx.template Alloc<BatchNormParamType<T>>(saved_mean);
  dev_ctx.template Alloc<BatchNormParamType<T>>(saved_variance);
  functor(dev_ctx, saved_mean, static_cast<BatchNormParamType<T>>(0));
  functor(dev_ctx, saved_variance, static_cast<BatchNormParamType<T>>(0));

#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::miopenBatchNormalizationForwardTraining(
          handle,
          miopenBNSpatial,
          const_cast<void *>(
              static_cast<const void *>(CudnnDataType<T>::kOne())),
          const_cast<void *>(
              static_cast<const void *>(CudnnDataType<T>::kZero())),
          data_desc_,
          static_cast<const void *>(x_tmp.template data<T>()),
          data_desc_,
          static_cast<void *>(y->template data<T>()),
          in_param_desc_,
          const_cast<void *>(static_cast<const void *>(
              scale_tmp.template data<BatchNormParamType<T>>())),
          const_cast<void *>(static_cast<const void *>(
              bias_tmp.template data<BatchNormParamType<T>>())),
          0,
          nullptr,
          nullptr,
          epsilon,
          static_cast<void *>(
              saved_mean->template data<BatchNormParamType<T>>()),
          static_cast<void *>(
              saved_variance->template data<BatchNormParamType<T>>())));

  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::miopenDestroyTensorDescriptor(data_desc_));
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::miopenDestroyTensorDescriptor(in_param_desc_));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::cudnnBatchNormalizationForwardTraining(
          handle,
          CUDNN_BATCHNORM_SPATIAL,
          CudnnDataType<T>::kOne(),
          CudnnDataType<T>::kZero(),
          data_desc_,
          x_tmp.template data<T>(),
          data_desc_,
          y->template data<T>(),
          in_param_desc_,
          scale_tmp.template data<BatchNormParamType<T>>(),
          bias_tmp.template data<BatchNormParamType<T>>(),
          0,
          nullptr,
          nullptr,
          epsilon,
          saved_mean->template data<BatchNormParamType<T>>(),
          saved_variance->template data<BatchNormParamType<T>>()));

  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::cudnnDestroyTensorDescriptor(data_desc_));
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::cudnnDestroyTensorDescriptor(in_param_desc_));
#endif
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
PD_REGISTER_KERNEL(
    instance_norm, GPU, ALL_LAYOUT, phi::InstanceNormKernel, float) {}
#else
PD_REGISTER_KERNEL(
    instance_norm, GPU, ALL_LAYOUT, phi::InstanceNormKernel, float, double) {}
#endif
