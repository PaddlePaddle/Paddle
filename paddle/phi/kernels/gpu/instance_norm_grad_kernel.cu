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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/norm_utils.h"
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

static __device__ __forceinline__ float real_sqrt(float x) {
  return 1. / sqrtf(x);
}
static __device__ __forceinline__ double real_sqrt(double x) {
  return 1. / sqrt(x);
}

template <typename T, int BlockDim>
__global__ void DoubleGradComputeDX(const T *x,
                                    const T *mean,
                                    const T *variance,
                                    const T *ddx,
                                    const T *dy,
                                    const T *scale,
                                    const T *ddscale,
                                    int C,
                                    int sample_size,
                                    const double epsilon,
                                    T *dx) {
  int beg_idx = blockIdx.x * sample_size + threadIdx.x;
  int end_idx = (blockIdx.x + 1) * sample_size;
  int ncid = blockIdx.x;
  int c = ncid % C;

  T mean_val = mean[ncid];
  T var_val = variance[ncid];

  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage dy_storage;
  __shared__ typename BlockReduce::TempStorage ddx_storage;
  __shared__ typename BlockReduce::TempStorage dy_mul_ddx_storage;
  __shared__ typename BlockReduce::TempStorage dy_mul_x_sub_mean_storage;
  __shared__ typename BlockReduce::TempStorage ddx_mul_x_sub_mean_storage;
  __shared__ T dy_sum_val;
  __shared__ T ddx_sum_val;
  __shared__ T dy_mul_ddx_sum_val;
  __shared__ T dy_mul_x_sub_mean_sum_val;
  __shared__ T ddx_mul_x_sub_mean_sum_val;

  T dy_sum = 0;
  T ddx_sum = 0;
  T dy_mul_ddx_sum = 0;
  T dy_mul_x_sub_mean_sum = 0;
  T ddx_mul_x_sub_mean_sum = 0;
  for (int i = beg_idx; i < end_idx; i += BlockDim) {
    T ddx_i = ddx[i];
    T dy_i = dy[i];
    T tmp = x[i] - mean_val;

    dy_sum += dy_i;
    ddx_sum += ddx_i;
    dy_mul_ddx_sum += (ddx_i * dy_i);

    dy_mul_x_sub_mean_sum += (dy_i * tmp);
    ddx_mul_x_sub_mean_sum += (ddx_i * tmp);
  }

  dy_sum = BlockReduce(dy_storage).Reduce(dy_sum, cub::Sum());
  ddx_sum = BlockReduce(ddx_storage).Reduce(ddx_sum, cub::Sum());
  dy_mul_ddx_sum =
      BlockReduce(dy_mul_ddx_storage).Reduce(dy_mul_ddx_sum, cub::Sum());
  dy_mul_x_sub_mean_sum = BlockReduce(dy_mul_x_sub_mean_storage)
                              .Reduce(dy_mul_x_sub_mean_sum, cub::Sum());
  ddx_mul_x_sub_mean_sum = BlockReduce(ddx_mul_x_sub_mean_storage)
                               .Reduce(ddx_mul_x_sub_mean_sum, cub::Sum());

  if (threadIdx.x == 0) {
    dy_sum_val = dy_sum;
    ddx_sum_val = ddx_sum;
    dy_mul_ddx_sum_val = dy_mul_ddx_sum;
    dy_mul_x_sub_mean_sum_val = dy_mul_x_sub_mean_sum;
    ddx_mul_x_sub_mean_sum_val = ddx_mul_x_sub_mean_sum;
  }
  __syncthreads();

  if (ddx != nullptr) {
    for (int i = beg_idx; i < end_idx; i += BlockDim) {
      dx[i] +=
          ((x[i] - mean_val) * var_val * var_val * var_val / sample_size *
               (ddx_sum_val * dy_sum_val / sample_size - dy_mul_ddx_sum_val +
                3. * dy_mul_x_sub_mean_sum_val * var_val *
                    ddx_mul_x_sub_mean_sum_val * var_val / sample_size) +
           ddx_mul_x_sub_mean_sum_val * var_val / sample_size * var_val *
               var_val * (dy_sum_val / sample_size - dy[i]) +
           dy_mul_x_sub_mean_sum_val * var_val / sample_size * var_val *
               var_val * (ddx_sum_val / sample_size - ddx[i])) *
          scale[c];
    }
  }
  __syncthreads();
  if (ddscale != nullptr) {
    for (int i = beg_idx; i < end_idx; i += BlockDim) {
      dx[i] += (dy[i] * var_val - dy_sum_val / sample_size * var_val -
                (x[i] - mean_val) * var_val * dy_mul_x_sub_mean_sum_val *
                    var_val / sample_size) *
               ddscale[c];
    }
  }
}

template <typename T, int BlockDim>
__global__ void DoubleGradComputeDDY(const T *x,
                                     const T *mean,
                                     const T *variance,
                                     const T *ddscale,
                                     const T *ddbias,
                                     const T *ddx,
                                     const T *scale,
                                     int C,
                                     int sample_size,
                                     const double epsilon,
                                     T *ddy) {
  int beg_idx = blockIdx.x * sample_size + threadIdx.x;
  int end_idx = (blockIdx.x + 1) * sample_size;
  int ncid = blockIdx.x;
  int c = ncid % C;
  T mean_val = mean[ncid];
  T var_val = variance[ncid];
  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ddx_storage;
  __shared__ typename BlockReduce::TempStorage ddx_mul_x_sub_mean_storage;
  __shared__ T ddx_sum_val;
  __shared__ T ddx_mul_x_sub_mean_sum_val;

  T ddx_sum = 0;
  T ddx_mul_x_sub_mean_sum = 0;
  for (int i = beg_idx; i < end_idx; i += BlockDim) {
    T ddx_i = ddx[i];
    ddx_sum += ddx_i;
    ddx_mul_x_sub_mean_sum += (ddx_i * (x[i] - mean_val));
  }
  ddx_sum = BlockReduce(ddx_storage).Reduce(ddx_sum, cub::Sum());
  ddx_mul_x_sub_mean_sum = BlockReduce(ddx_mul_x_sub_mean_storage)
                               .Reduce(ddx_mul_x_sub_mean_sum, cub::Sum());
  if (threadIdx.x == 0) {
    ddx_sum_val = ddx_sum;
    ddx_mul_x_sub_mean_sum_val = ddx_mul_x_sub_mean_sum;
  }
  __syncthreads();
  if (ddx != nullptr) {
    for (int i = beg_idx; i < end_idx; i += BlockDim) {
      ddy[i] += scale[c] * var_val *
                (ddx[i] - ddx_sum_val / sample_size -
                 (x[i] - mean_val) * var_val * ddx_mul_x_sub_mean_sum_val *
                     var_val / sample_size);
    }
  }
  __syncthreads();
  if (ddscale != nullptr) {
    for (int i = beg_idx; i < end_idx; i += BlockDim) {
      ddy[i] += (x[i] - mean_val) * var_val * ddscale[c];
    }
  }
  __syncthreads();
  if (ddbias != nullptr) {
    for (int i = beg_idx; i < end_idx; i += BlockDim) {
      ddy[i] += ddbias[c];
    }
  }
}

template <typename T, int BlockDim>
__global__ void DoubleGradComputeDScale(const T *x,
                                        const T *mean,
                                        const T *variance,
                                        const T *ddx,
                                        const T *dy,
                                        int C,
                                        int sample_size,
                                        const double epsilon,
                                        T *dscale) {
  int beg_idx = blockIdx.x * sample_size + threadIdx.x;
  int end_idx = (blockIdx.x + 1) * sample_size;
  int ncid = blockIdx.x;
  int c = ncid % C;
  T mean_val = mean[ncid];
  T var_val = variance[ncid];
  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage dy_storage;
  __shared__ typename BlockReduce::TempStorage dy_mul_x_sub_mean_storage;
  __shared__ typename BlockReduce::TempStorage dscale_tmp_storage;
  __shared__ T dy_sum_val;
  __shared__ T dy_mul_x_sub_mean_sum_val;

  T dy_sum = 0;
  T dy_mul_x_sub_mean_sum = 0;
  for (int i = beg_idx; i < end_idx; i += BlockDim) {
    T dy_i = dy[i];
    dy_sum += dy_i;
    dy_mul_x_sub_mean_sum += (dy_i * (x[i] - mean_val));
  }
  dy_sum = BlockReduce(dy_storage).Reduce(dy_sum, cub::Sum());
  dy_mul_x_sub_mean_sum = BlockReduce(dy_mul_x_sub_mean_storage)
                              .Reduce(dy_mul_x_sub_mean_sum, cub::Sum());

  if (threadIdx.x == 0) {
    dy_sum_val = dy_sum;
    dy_mul_x_sub_mean_sum_val = dy_mul_x_sub_mean_sum;
  }
  __syncthreads();
  if (ddx != nullptr) {
    T dscale_tmp = 0;
    for (int i = beg_idx; i < end_idx; i += BlockDim) {
      dscale_tmp += ddx[i] * var_val *
                    (dy[i] - dy_sum_val / sample_size -
                     dy_mul_x_sub_mean_sum_val * (x[i] - mean_val) * var_val *
                         var_val / sample_size);
    }
    dscale_tmp = BlockReduce(dscale_tmp_storage).Reduce(dscale_tmp, cub::Sum());
    if (threadIdx.x == 0) {
      dscale[ncid] += dscale_tmp;
    }
    __syncthreads();
  }
}

template <typename T, typename Context>
void InstanceNormGradKernel(const Context &dev_ctx,
                            const DenseTensor &x,
                            const paddle::optional<DenseTensor> &scale,
                            const DenseTensor &saved_mean,
                            const DenseTensor &saved_variance,
                            const DenseTensor &d_y,
                            float epsilon_f,
                            DenseTensor *d_x,
                            DenseTensor *d_scale,
                            DenseTensor *d_bias) {
  double epsilon = static_cast<double>(epsilon_f);
  const auto *scale_ptr = scale.get_ptr();

  const auto &x_dims = x.dims();

  int N, C, H, W, D;
  funcs::ExtractNCWHD(x_dims, DataLayout::kNCHW, &N, &C, &H, &W, &D);
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

template <typename T, typename Context>
void InstanceNormDoubleGradKernel(const Context &dev_ctx,
                                  const DenseTensor &x,
                                  const paddle::optional<DenseTensor> &scale,
                                  const DenseTensor &saved_mean,
                                  const DenseTensor &saved_variance,
                                  const DenseTensor &dy,
                                  const paddle::optional<DenseTensor> &ddx,
                                  const paddle::optional<DenseTensor> &ddscale,
                                  const paddle::optional<DenseTensor> &ddbias,
                                  float epsilon_f,
                                  DenseTensor *dx,
                                  DenseTensor *dscale,
                                  DenseTensor *ddy) {
  const auto *Scale = scale.get_ptr();
  const auto *ddX = ddx.get_ptr();
  const auto *ddScale = ddscale.get_ptr();
  const auto *ddBias = ddbias.get_ptr();
  const double epsilon = static_cast<double>(epsilon_f);
  const T *x_data = x.data<T>();
  const T *dy_data = dy.data<T>();
  const T *ddx_data = (ddX == nullptr ? nullptr : ddX->data<T>());
  const T *ddscale_data = (ddScale == nullptr ? nullptr : ddScale->data<T>());
  const T *ddbias_data = (ddScale == nullptr ? nullptr : ddBias->data<T>());
  const T *mean_data = saved_mean.data<T>();
  const T *variance_data = saved_variance.data<T>();
  phi::funcs::SetConstant<GPUContext, T> set_zero;
  auto &x_dims = x.dims();
  int N, C, H, W, D;
  funcs::ExtractNCWHD(x_dims, DataLayout::kNCHW, &N, &C, &H, &W, &D);
  int NxC = N * C;
  const int n = x.numel();
  int sample_size = n / N / C;

  DenseTensor scale_tmp;
  if (!Scale) {
    scale_tmp.Resize({C});
    dev_ctx.template Alloc<T>(&scale_tmp);
    set_zero(dev_ctx, &scale_tmp, static_cast<T>(1));
  }
  const T *scale_data = Scale ? Scale->data<T>() : scale_tmp.data<T>();
  const int block = 512;
  int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
  const int max_blocks = std::max(max_threads / block, 1);
  const int grid = NxC;
  const int grid1 = (C + block - 1) / block;

  if (dx) {
    T *dx_data = dev_ctx.template Alloc<T>(dx);
    set_zero(dev_ctx, dx, static_cast<T>(0));
    DoubleGradComputeDX<T, block>
        <<<grid, block, 0, dev_ctx.stream()>>>(x_data,
                                               mean_data,
                                               variance_data,
                                               ddx_data,
                                               dy_data,
                                               scale_data,
                                               ddscale_data,
                                               C,
                                               sample_size,
                                               epsilon,
                                               dx_data);
  }
  if (dscale) {
    DenseTensor dscale_tmp;
    dscale_tmp.Resize({NxC});
    dev_ctx.template Alloc<T>(&dscale_tmp);
    set_zero(dev_ctx, &dscale_tmp, static_cast<T>(0));
    T *dscale_tmp_data = dscale_tmp.data<T>();

    T *dscale_data = dev_ctx.template Alloc<T>(dscale);
    set_zero(dev_ctx, dscale, static_cast<T>(0));
    DoubleGradComputeDScale<T, block>
        <<<grid, block, 0, dev_ctx.stream()>>>(x_data,
                                               mean_data,
                                               variance_data,
                                               ddx_data,
                                               dy_data,
                                               C,
                                               sample_size,
                                               epsilon,
                                               dscale_tmp_data);
    add_param<T, block, false><<<grid1, block, 0, dev_ctx.stream()>>>(
        dscale_tmp.data<T>(), dscale->data<T>(), N, C);
  }
  if (ddy) {
    T *ddy_data = dev_ctx.template Alloc<T>(ddy);
    set_zero(dev_ctx, ddy, static_cast<T>(0));
    DoubleGradComputeDDY<T, block>
        <<<grid, block, 0, dev_ctx.stream()>>>(x_data,
                                               mean_data,
                                               variance_data,
                                               ddscale_data,
                                               ddbias_data,
                                               ddx_data,
                                               scale_data,
                                               C,
                                               sample_size,
                                               epsilon,
                                               ddy_data);
  }
}
}  // namespace phi

#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
PD_REGISTER_KERNEL(
    instance_norm_grad, GPU, ALL_LAYOUT, phi::InstanceNormGradKernel, float) {}
PD_REGISTER_KERNEL(instance_norm_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::InstanceNormDoubleGradKernel,
                   float) {}
#else
PD_REGISTER_KERNEL(instance_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::InstanceNormGradKernel,
                   float,
                   double) {}
PD_REGISTER_KERNEL(instance_norm_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::InstanceNormDoubleGradKernel,
                   float,
                   double) {}
#endif
