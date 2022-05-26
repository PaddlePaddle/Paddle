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

#include "paddle/phi/kernels/instance_norm_grad_kernel.h"

#include "paddle/fluid/operators/norm_utils.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/instance_norm_utils.h"

namespace phi {

template <typename T, int BlockDim>
static __global__ void GradComputeDX(const T *dy,
                                     const BatchNormParamType<T> *scale,
                                     const BatchNormParamType<T> *mean,
                                     const T *x,
                                     const BatchNormParamType<T> *variance,
                                     const int C,
                                     const int sample_size,
                                     T *dx) {
  int beg_idx = blockIdx.x * sample_size + threadIdx.x;
  int end_idx = (blockIdx.x + 1) * sample_size;
  int ncid = blockIdx.x;
  int c = ncid % C;

  BatchNormParamType<T> mean_val = mean[ncid];
  BatchNormParamType<T> inv_var_val = variance[ncid];

  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage dy_storage;
  __shared__ typename BlockReduce::TempStorage dy_x_sub_mean_storage;
  __shared__ BatchNormParamType<T> dy_sum_val;
  __shared__ BatchNormParamType<T> dy_x_sub_mean_sum_val;

  BatchNormParamType<T> dy_sum = static_cast<BatchNormParamType<T>>(0);
  BatchNormParamType<T> dy_x_sub_mean_sum =
      static_cast<BatchNormParamType<T>>(0);

  for (int i = beg_idx; i < end_idx; i += BlockDim) {
    BatchNormParamType<T> dy_i = static_cast<BatchNormParamType<T>>(dy[i]);
    dy_sum += dy_i;
    dy_x_sub_mean_sum +=
        dy_i * (static_cast<BatchNormParamType<T>>(x[i]) - mean_val);
  }
  dy_sum = BlockReduce(dy_storage).Reduce(dy_sum, cub::Sum());
  dy_x_sub_mean_sum =
      BlockReduce(dy_x_sub_mean_storage).Reduce(dy_x_sub_mean_sum, cub::Sum());

  if (threadIdx.x == 0) {
    dy_sum_val = dy_sum;
    dy_x_sub_mean_sum_val = dy_x_sub_mean_sum;
  }
  __syncthreads();

  for (int i = beg_idx; i < end_idx; i += BlockDim) {
    dx[i] =
        (static_cast<BatchNormParamType<T>>(dy[i]) -
         dy_sum_val / static_cast<BatchNormParamType<T>>(sample_size) -
         (static_cast<BatchNormParamType<T>>(x[i]) - mean_val) *
             dy_x_sub_mean_sum_val * inv_var_val * inv_var_val / sample_size) *
        scale[c] * inv_var_val;
  }
}

template <typename T, typename Context>
void InstanceNormGradKernel(const Context &dev_ctx,
                            const DenseTensor &x,
                            const DenseTensor &d_y,
                            const paddle::optional<DenseTensor> &scale,
                            const DenseTensor &saved_mean,
                            const DenseTensor &saved_variance,
                            float epsilon_f,
                            DenseTensor *d_x,
                            DenseTensor *d_scale,
                            DenseTensor *d_bias) {
  double epsilon = static_cast<double>(epsilon_f);
  const auto *scale_ptr = scale.get_ptr();

  const auto &x_dims = x.dims();

  int N, C, H, W, D;
  paddle::operators::ExtractNCWHD(
      x_dims, DataLayout::kNCHW, &N, &C, &H, &W, &D);
  int NxC = N * C;

  DenseTensor x_tmp, d_y_tmp;
  x_tmp.ShareDataWith(x).Resize({1, NxC, H, W, D});
  d_y_tmp.ShareDataWith(d_y).Resize({1, NxC, H, W, D});

  dev_ctx.template Alloc<T>(d_x);
  if (d_scale && d_bias) {
    dev_ctx.template Alloc<T>(d_scale);
    dev_ctx.template Alloc<T>(d_bias);
  }
  if (scale_ptr) {
    PADDLE_ENFORCE_EQ(
        scale_ptr->dims().size(),
        1UL,
        phi::errors::InvalidArgument(
            "The `shape` in InstanceNormOp is invalid: "
            "the size of scale's dimensions must be equal to 1. But "
            "received: the size of scale's dimensions"
            "is [%d]",
            scale_ptr->dims().size()));
    PADDLE_ENFORCE_EQ(scale_ptr->dims()[0],
                      C,
                      phi::errors::InvalidArgument(
                          "The `shape` in InstanceNormOp is invalid: "
                          "the first dimension of scale must be equal to "
                          "Channels([%d]). But received: "
                          "the first dimension of scale is [%d],"
                          "the dimensions of scale is [%s], ",
                          C,
                          scale_ptr->dims()[0],
                          scale_ptr->dims()));
  }

  phi::funcs::SetConstant<GPUContext, T> set_constant;

  const int n = x.numel();
  const int block = 512;
  int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
  const int max_blocks = std::max(max_threads / block, 1);
  const int grid = std::min(NxC, max_blocks);
  const int grid1 = (C + block - 1) / block;

  DenseTensor scale_tmp;
  scale_tmp.Resize({NxC});
  dev_ctx.template Alloc<T>(&scale_tmp);

  DenseTensor d_scale_tmp;
  d_scale_tmp.Resize({NxC});
  dev_ctx.template Alloc<T>(&d_scale_tmp);

  DenseTensor d_bias_tmp;
  d_bias_tmp.Resize({NxC});
  dev_ctx.template Alloc<T>(&d_bias_tmp);

  if (scale_ptr) {
    repeat_param<T><<<grid, block, 0, dev_ctx.stream()>>>(
        scale_ptr->data<T>(), scale_tmp.data<T>(), N, C);
  } else {
    set_constant(dev_ctx, &scale_tmp, static_cast<T>(1));
  }

  std::vector<int> dims;
  std::vector<int> strides;
  dims = {1, NxC, H, W, D};
  strides = {NxC * H * W * D, H * W * D, W * D, D, 1};

  if ((H * W * D) == 1) {
    phi::Copy(dev_ctx, d_y, dev_ctx.GetPlace(), false, d_x);
    phi::funcs::SetConstant<GPUContext, BatchNormParamType<T>> functor;
    functor(dev_ctx, d_scale, static_cast<BatchNormParamType<T>>(0));
    functor(dev_ctx, d_bias, static_cast<BatchNormParamType<T>>(0));
    return;
  }

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

  const auto *saved_mean_data =
      saved_mean.template data<BatchNormParamType<T>>();
  const auto *saved_var_data =
      saved_variance.template data<BatchNormParamType<T>>();
  if (d_scale && d_bias) {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::miopenBatchNormalizationBackward(
            dev_ctx.cudnn_handle(),
            miopenBNSpatial,
            CudnnDataType<T>::kOne(),
            CudnnDataType<T>::kZero(),
            CudnnDataType<T>::kOne(),
            CudnnDataType<T>::kZero(),
            data_desc_,
            x_tmp.template data<T>(),
            data_desc_,
            d_y_tmp.template data<T>(),
            data_desc_,
            d_x->template data<T>(),
            in_param_desc_,
            scale_tmp.template data<BatchNormParamType<T>>(),
            d_scale_tmp.template data<BatchNormParamType<T>>(),
            d_bias_tmp.template data<BatchNormParamType<T>>(),
            epsilon,
            saved_mean_data,
            saved_var_data));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::cudnnBatchNormalizationBackward(
            dev_ctx.cudnn_handle(),
            CUDNN_BATCHNORM_SPATIAL,
            CudnnDataType<T>::kOne(),
            CudnnDataType<T>::kZero(),
            CudnnDataType<T>::kOne(),
            CudnnDataType<T>::kZero(),
            data_desc_,
            x_tmp.template data<T>(),
            data_desc_,
            d_y_tmp.template data<T>(),
            data_desc_,
            d_x->template data<T>(),
            in_param_desc_,
            scale_tmp.template data<BatchNormParamType<T>>(),
            d_scale_tmp.template data<BatchNormParamType<T>>(),
            d_bias_tmp.template data<BatchNormParamType<T>>(),
            epsilon,
            saved_mean_data,
            saved_var_data));
#endif
  } else {
    if (d_x) {
      GradComputeDX<T, block><<<NxC, block, 0, dev_ctx.stream()>>>(
          d_y.data<T>(),
          scale_tmp.data<BatchNormParamType<T>>(),
          saved_mean_data,
          x.data<T>(),
          saved_var_data,
          C,
          H * W * D,
          d_x->data<T>());
    }
  }

  if (d_scale && d_bias) {
    add_param<T, block, false><<<grid1, block, 0, dev_ctx.stream()>>>(
        d_scale_tmp.data<T>(), d_scale->data<T>(), N, C);
    add_param<T, block, false><<<grid1, block, 0, dev_ctx.stream()>>>(
        d_bias_tmp.data<T>(), d_bias->data<T>(), N, C);
  }

#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::miopenDestroyTensorDescriptor(data_desc_));
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::miopenDestroyTensorDescriptor(in_param_desc_));
#else
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
    instance_norm_grad, GPU, ALL_LAYOUT, phi::InstanceNormGradKernel, float) {}
#else
PD_REGISTER_KERNEL(instance_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::InstanceNormGradKernel,
                   float,
                   double) {}
#endif
