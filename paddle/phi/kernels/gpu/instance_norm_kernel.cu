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

#include "glog/logging.h"

#include "paddle/common/layout.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
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
  using AccT = typename phi::dtype::MPTypeTrait<T>::Type;
  double epsilon = static_cast<double>(epsilon_f);
  auto &x_dims = x.dims();
  PADDLE_ENFORCE_GE(x_dims.size(),
                    2,
                    common::errors::InvalidArgument(
                        "The `shape` in InstanceNormOp is invalid: "
                        "the size of X's dimensions must greater than "
                        "or equal to 2. But received: "
                        "the size of X's dimensions is [%d]",
                        x_dims.size()));
  PADDLE_ENFORCE_LE(x_dims.size(),
                    5,
                    common::errors::InvalidArgument(
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
      phi::dynload::miopenCreateTensorDescriptor(&data_desc_));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::miopenCreateTensorDescriptor(&in_param_desc_));
#else
  cudnnTensorDescriptor_t data_desc_;
  cudnnTensorDescriptor_t in_param_desc_;

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&data_desc_));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&in_param_desc_));
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
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetTensorDescriptor(
      data_desc_,
      CudnnDataType<T>::type,
      x_dims.size() > 3 ? x_dims.size() : 4,
      const_cast<int *>(dims.data()),
      const_cast<int *>(strides.data())));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenDeriveBNTensorDescriptor(
      in_param_desc_, data_desc_, miopenBNSpatial));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
      data_desc_,
      CudnnDataType<T>::type,
      x_dims.size() > 3 ? x_dims.size() : 4,
      dims.data(),
      strides.data()));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnDeriveBNTensorDescriptor(
      in_param_desc_, data_desc_, CUDNN_BATCHNORM_SPATIAL));
#endif

  const auto scale_ptr = scale.get_ptr();
  const auto bias_ptr = bias.get_ptr();

  DenseTensor scale_tmp;
  scale_tmp.Resize({NxC});
  dev_ctx.template Alloc<AccT>(&scale_tmp);
  DenseTensor bias_tmp;
  bias_tmp.Resize({NxC});
  dev_ctx.template Alloc<AccT>(&bias_tmp);

  const int n = x.numel();
  const int block = 512;
  int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
  const int max_blocks = std::max(max_threads / block, 1);
  const int grid = std::min((NxC + block - 1) / block, max_blocks);

  phi::funcs::SetConstant<GPUContext, AccT> set_constant;
  if (scale_ptr) {
    repeat_param<AccT><<<grid, block, 0, dev_ctx.stream()>>>(
        scale_ptr->data<AccT>(), scale_tmp.data<AccT>(), N, C);
  } else {
    set_constant(dev_ctx, &scale_tmp, static_cast<AccT>(1));
  }
  if (bias_ptr) {
    repeat_param<AccT><<<grid, block, 0, dev_ctx.stream()>>>(
        bias_ptr->data<AccT>(), bias_tmp.data<AccT>(), N, C);
  } else {
    set_constant(dev_ctx, &bias_tmp, static_cast<AccT>(0));
  }

  auto handle = dev_ctx.cudnn_handle();

  DenseTensor saved_mean_tmp, saved_variance_tmp;
  phi::funcs::SetConstant<GPUContext, BatchNormParamType<T>> functor;

  if (saved_mean) {
    dev_ctx.template Alloc<BatchNormParamType<T>>(saved_mean);
    functor(dev_ctx, saved_mean, static_cast<BatchNormParamType<T>>(0));
  } else {
    saved_mean_tmp = phi::Full<BatchNormParamType<T>>(
        dev_ctx, {NxC}, static_cast<BatchNormParamType<T>>(0));
  }
  if (saved_variance) {
    dev_ctx.template Alloc<BatchNormParamType<T>>(saved_variance);
    functor(dev_ctx, saved_variance, static_cast<BatchNormParamType<T>>(0));
  } else {
    saved_variance_tmp = phi::Full<BatchNormParamType<T>>(
        dev_ctx, {NxC}, static_cast<BatchNormParamType<T>>(0));
  }
  auto *saved_mean_data = saved_mean
                              ? saved_mean->data<BatchNormParamType<T>>()
                              : saved_mean_tmp.data<BatchNormParamType<T>>();
  auto *saved_variance_data =
      saved_variance ? saved_variance->data<BatchNormParamType<T>>()
                     : saved_variance_tmp.data<BatchNormParamType<T>>();

#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::miopenBatchNormalizationForwardTraining(
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
          static_cast<void *>(saved_mean_data),
          static_cast<void *>(saved_variance_data)));

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::miopenDestroyTensorDescriptor(data_desc_));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::miopenDestroyTensorDescriptor(in_param_desc_));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnBatchNormalizationForwardTraining(
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
          saved_mean_data,
          saved_variance_data));

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(data_desc_));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(in_param_desc_));
#endif
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
PD_REGISTER_KERNEL(instance_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::InstanceNormKernel,
                   float,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}
#elif CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(instance_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::InstanceNormKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16 ||
      kernel_key.dtype() == phi::DataType::BFLOAT16) {
    kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}
#else
PD_REGISTER_KERNEL(instance_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::InstanceNormKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16 ||
      kernel_key.dtype() == phi::DataType::BFLOAT16) {
    kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}
#endif
