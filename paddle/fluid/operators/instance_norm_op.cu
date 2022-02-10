/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <cfloat>
#include <string>
#include <vector>
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/operators/instance_norm_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;
template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;

template <typename T>
static __global__ void repeat_param(const T *input, T *output,
                                    const int repeat_num, const int C) {
  CUDA_KERNEL_LOOP(i, repeat_num * C) {
    int index = i % C;
    output[i] = input[index];
  }
}

template <typename T, int BlockDim, bool AVG>
static __global__ void add_param(const T *input, T *output,
                                 const int repeat_num, const int C) {
  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ou_storage;
  for (int i = blockIdx.x; i < C; i += gridDim.x) {
    T ou = static_cast<T>(0);
    for (int j = threadIdx.x; j < repeat_num; j += blockDim.x) {
      const int index = j * C + i;
      ou += static_cast<T>(input[index]);
    }
    ou = BlockReduce(ou_storage).Reduce(ou, cub::Sum());
    if (threadIdx.x == 0) {
      output[i] = ou;
    }
    __syncthreads();

    if (AVG) {
      output[i] /= repeat_num;
    }
  }
}

template <typename T>
class InstanceNormKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("It must be CUDAPlace."));
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));

    auto *x = ctx.Input<Tensor>("X");
    auto &x_dims = x->dims();
    PADDLE_ENFORCE_GE(x_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The `shape` in InstanceNormOp is invalid: "
                          "the size of X's dimensions must greater than "
                          "or equal to 2. But received: "
                          "the size of X's dimensions is [%d]",
                          x_dims.size()));
    PADDLE_ENFORCE_LE(x_dims.size(), 5,
                      platform::errors::InvalidArgument(
                          "The `shape` in InstanceNormOp is invalid: "
                          "the size of X's dimensions must smaller than"
                          "or equal to 5. But received: "
                          "the size of X's dimensions is [%d]",
                          x_dims.size()));
    int N, C, H, W, D;
    ExtractNCWHD(x_dims, DataLayout::kNCHW, &N, &C, &H, &W, &D);
    int NxC = N * C;
    Tensor x_tmp;
    x_tmp.ShareDataWith(*x).Resize({1, NxC, H, W, D});

    auto *y = ctx.Output<Tensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());

#ifdef PADDLE_WITH_HIP
    miopenTensorDescriptor_t data_desc_;
    miopenTensorDescriptor_t in_param_desc_;

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::miopenCreateTensorDescriptor(&data_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::miopenCreateTensorDescriptor(&in_param_desc_));
#else
    cudnnTensorDescriptor_t data_desc_;
    cudnnTensorDescriptor_t in_param_desc_;

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&data_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&in_param_desc_));
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

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::miopenSetTensorDescriptor(
        data_desc_, CudnnDataType<T>::type,
        x_dims.size() > 3 ? x_dims.size() : 4, const_cast<int *>(dims.data()),
        const_cast<int *>(strides.data())));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::miopenDeriveBNTensorDescriptor(
            in_param_desc_, data_desc_, miopenBNSpatial));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        data_desc_, CudnnDataType<T>::type,
        x_dims.size() > 3 ? x_dims.size() : 4, dims.data(), strides.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnDeriveBNTensorDescriptor(
        in_param_desc_, data_desc_, CUDNN_BATCHNORM_SPATIAL));
#endif

    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");

    Tensor scale_tmp =
        ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({NxC}, dev_ctx);
    scale_tmp.mutable_data<T>(ctx.GetPlace());
    Tensor bias_tmp =
        ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({NxC}, dev_ctx);
    bias_tmp.mutable_data<T>(ctx.GetPlace());

    const int n = x->numel();
    const int block = 512;
    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    const int max_blocks = std::max(max_threads / block, 1);
    const int grid = std::min((NxC + block - 1) / block, max_blocks);

    math::SetConstant<platform::CUDADeviceContext, T> set_constant;
    if (scale) {
      repeat_param<T><<<grid, block, 0, dev_ctx.stream()>>>(
          scale->data<T>(), scale_tmp.data<T>(), N, C);
    } else {
      set_constant(dev_ctx, &scale_tmp, static_cast<T>(1));
    }
    if (bias) {
      repeat_param<T><<<grid, block, 0, dev_ctx.stream()>>>(
          bias->data<T>(), bias_tmp.data<T>(), N, C);
    } else {
      set_constant(dev_ctx, &bias_tmp, static_cast<T>(0));
    }

    auto handle = dev_ctx.cudnn_handle();

    math::SetConstant<platform::CUDADeviceContext, BatchNormParamType<T>>
        functor;

    auto *saved_mean = ctx.Output<Tensor>("SavedMean");
    auto *saved_variance = ctx.Output<Tensor>("SavedVariance");
    saved_mean->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    saved_variance->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    functor(dev_ctx, saved_mean, static_cast<BatchNormParamType<T>>(0));
    functor(dev_ctx, saved_variance, static_cast<BatchNormParamType<T>>(0));

#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::miopenBatchNormalizationForwardTraining(
            handle, miopenBNSpatial,
            const_cast<void *>(
                static_cast<const void *>(CudnnDataType<T>::kOne())),
            const_cast<void *>(
                static_cast<const void *>(CudnnDataType<T>::kZero())),
            data_desc_, static_cast<const void *>(x_tmp.template data<T>()),
            data_desc_,
            static_cast<void *>(y->template mutable_data<T>(ctx.GetPlace())),
            in_param_desc_,
            const_cast<void *>(static_cast<const void *>(
                scale_tmp.template data<BatchNormParamType<T>>())),
            const_cast<void *>(static_cast<const void *>(
                bias_tmp.template data<BatchNormParamType<T>>())),
            0, nullptr, nullptr, epsilon,
            static_cast<void *>(
                saved_mean->template mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace())),
            static_cast<void *>(
                saved_variance->template mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace()))));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::miopenDestroyTensorDescriptor(data_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::miopenDestroyTensorDescriptor(in_param_desc_));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnBatchNormalizationForwardTraining(
            handle, CUDNN_BATCHNORM_SPATIAL, CudnnDataType<T>::kOne(),
            CudnnDataType<T>::kZero(), data_desc_, x_tmp.template data<T>(),
            data_desc_, y->template mutable_data<T>(ctx.GetPlace()),
            in_param_desc_, scale_tmp.template data<BatchNormParamType<T>>(),
            bias_tmp.template data<BatchNormParamType<T>>(), 0, nullptr,
            nullptr, epsilon,
            saved_mean->template mutable_data<BatchNormParamType<T>>(
                ctx.GetPlace()),
            saved_variance->template mutable_data<BatchNormParamType<T>>(
                ctx.GetPlace())));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(data_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(in_param_desc_));
#endif
  }
};

template <typename T, int BlockDim>
static __global__ void GradComputeDX(const T *dy,
                                     const BatchNormParamType<T> *scale,
                                     const BatchNormParamType<T> *mean,
                                     const T *x,
                                     const BatchNormParamType<T> *variance,
                                     const int C, const int sample_size,
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

template <typename T>
class InstanceNormGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("It must use CUDAPlace."));
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *x = ctx.Input<Tensor>("X");
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));

    const auto &x_dims = x->dims();

    int N, C, H, W, D;
    ExtractNCWHD(x_dims, DataLayout::kNCHW, &N, &C, &H, &W, &D);
    int NxC = N * C;

    Tensor x_tmp, d_y_tmp;
    x_tmp.ShareDataWith(*x).Resize({1, NxC, H, W, D});
    d_y_tmp.ShareDataWith(*d_y).Resize({1, NxC, H, W, D});

    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    d_x->mutable_data<T>(ctx.GetPlace());
    if (d_scale && d_bias) {
      d_scale->mutable_data<T>(ctx.GetPlace());
      d_bias->mutable_data<T>(ctx.GetPlace());
    }
    if (scale) {
      PADDLE_ENFORCE_EQ(
          scale->dims().size(), 1UL,
          platform::errors::InvalidArgument(
              "The `shape` in InstanceNormOp is invalid: "
              "the size of scale's dimensions must be equal to 1. But "
              "received: the size of scale's dimensions"
              "is [%d]",
              scale->dims().size()));
      PADDLE_ENFORCE_EQ(scale->dims()[0], C,
                        platform::errors::InvalidArgument(
                            "The `shape` in InstanceNormOp is invalid: "
                            "the first dimension of scale must be equal to "
                            "Channels([%d]). But received: "
                            "the first dimension of scale is [%d],"
                            "the dimensions of scale is [%s], ",
                            C, scale->dims()[0], scale->dims()));
    }

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    math::SetConstant<platform::CUDADeviceContext, T> set_constant;

    const int n = x->numel();
    const int block = 512;
    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    const int max_blocks = std::max(max_threads / block, 1);
    const int grid = std::min(NxC, max_blocks);
    const int grid1 = (C + block - 1) / block;

    Tensor scale_tmp =
        ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({NxC}, dev_ctx);
    scale_tmp.mutable_data<T>(ctx.GetPlace());
    Tensor d_scale_tmp =
        ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({NxC}, dev_ctx);
    Tensor d_bias_tmp =
        ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({NxC}, dev_ctx);
    if (scale) {
      repeat_param<T><<<grid, block, 0, dev_ctx.stream()>>>(
          scale->data<T>(), scale_tmp.data<T>(), N, C);
    } else {
      set_constant(dev_ctx, &scale_tmp, static_cast<T>(1));
    }

    std::vector<int> dims;
    std::vector<int> strides;
    dims = {1, NxC, H, W, D};
    strides = {NxC * H * W * D, H * W * D, W * D, D, 1};

    if ((H * W * D) == 1) {
      framework::TensorCopy(*d_y, ctx.GetPlace(), d_x);
      math::SetConstant<platform::CUDADeviceContext, BatchNormParamType<T>>
          functor;
      functor(dev_ctx, d_scale, static_cast<BatchNormParamType<T>>(0));
      functor(dev_ctx, d_bias, static_cast<BatchNormParamType<T>>(0));
      return;
    }

#ifdef PADDLE_WITH_HIP
    miopenTensorDescriptor_t data_desc_;
    miopenTensorDescriptor_t in_param_desc_;

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::miopenCreateTensorDescriptor(&data_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::miopenCreateTensorDescriptor(&in_param_desc_));
#else
    cudnnTensorDescriptor_t data_desc_;
    cudnnTensorDescriptor_t in_param_desc_;

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&data_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&in_param_desc_));
#endif

    if (epsilon <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
      LOG(ERROR) << "Provided epsilon is smaller than "
                 << "CUDNN_BN_MIN_EPSILON. Setting it to "
                 << "CUDNN_BN_MIN_EPSILON instead.";
    }
    epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);

#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::miopenSetTensorDescriptor(
        data_desc_, CudnnDataType<T>::type,
        x_dims.size() > 3 ? x_dims.size() : 4, const_cast<int *>(dims.data()),
        const_cast<int *>(strides.data())));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::miopenDeriveBNTensorDescriptor(
            in_param_desc_, data_desc_, miopenBNSpatial));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        data_desc_, CudnnDataType<T>::type,
        x_dims.size() > 3 ? x_dims.size() : 4, dims.data(), strides.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnDeriveBNTensorDescriptor(
        in_param_desc_, data_desc_, CUDNN_BATCHNORM_SPATIAL));
#endif

    const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
    const auto *saved_var = ctx.Input<Tensor>("SavedVariance");
    const auto *saved_mean_data =
        saved_mean->template data<BatchNormParamType<T>>();
    const auto *saved_var_data =
        saved_var->template data<BatchNormParamType<T>>();
    if (d_scale && d_bias) {
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::miopenBatchNormalizationBackward(
              dev_ctx.cudnn_handle(), miopenBNSpatial, CudnnDataType<T>::kOne(),
              CudnnDataType<T>::kZero(), CudnnDataType<T>::kOne(),
              CudnnDataType<T>::kZero(), data_desc_, x_tmp.template data<T>(),
              data_desc_, d_y_tmp.template data<T>(), data_desc_,
              d_x->template mutable_data<T>(ctx.GetPlace()), in_param_desc_,
              scale_tmp.template data<BatchNormParamType<T>>(),
              d_scale_tmp.template mutable_data<BatchNormParamType<T>>(
                  ctx.GetPlace()),
              d_bias_tmp.template mutable_data<BatchNormParamType<T>>(
                  ctx.GetPlace()),
              epsilon, saved_mean_data, saved_var_data));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnBatchNormalizationBackward(
              dev_ctx.cudnn_handle(), CUDNN_BATCHNORM_SPATIAL,
              CudnnDataType<T>::kOne(), CudnnDataType<T>::kZero(),
              CudnnDataType<T>::kOne(), CudnnDataType<T>::kZero(), data_desc_,
              x_tmp.template data<T>(), data_desc_, d_y_tmp.template data<T>(),
              data_desc_, d_x->template mutable_data<T>(ctx.GetPlace()),
              in_param_desc_, scale_tmp.template data<BatchNormParamType<T>>(),
              d_scale_tmp.template mutable_data<BatchNormParamType<T>>(
                  ctx.GetPlace()),
              d_bias_tmp.template mutable_data<BatchNormParamType<T>>(
                  ctx.GetPlace()),
              epsilon, saved_mean_data, saved_var_data));
#endif
    } else {
      if (d_x) {
        GradComputeDX<T, block><<<NxC, block, 0, dev_ctx.stream()>>>(
            d_y->data<T>(), scale_tmp.data<BatchNormParamType<T>>(),
            saved_mean_data, x->data<T>(), saved_var_data, C, H * W * D,
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
        platform::dynload::miopenDestroyTensorDescriptor(data_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::miopenDestroyTensorDescriptor(in_param_desc_));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(data_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(in_param_desc_));
#endif
  }
};

static __device__ __forceinline__ float real_sqrt(float x) {
  return 1. / sqrtf(x);
}
static __device__ __forceinline__ double real_sqrt(double x) {
  return 1. / sqrt(x);
}

template <typename T, int BlockDim>
__global__ void DoubleGradComputeDX(const T *x, const T *mean,
                                    const T *variance, const T *ddx,
                                    const T *dy, const T *scale,
                                    const T *ddscale, int C, int sample_size,
                                    const double epsilon, T *dx) {
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
__global__ void DoubleGradComputeDDY(const T *x, const T *mean,
                                     const T *variance, const T *ddscale,
                                     const T *ddbias, const T *ddx,
                                     const T *scale, int C, int sample_size,
                                     const double epsilon, T *ddy) {
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
__global__ void DoubleGradComputeDScale(const T *x, const T *mean,
                                        const T *variance, const T *ddx,
                                        const T *dy, int C, int sample_size,
                                        const double epsilon, T *dscale) {
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
      dscale_tmp +=
          ddx[i] * var_val * (dy[i] - dy_sum_val / sample_size -
                              dy_mul_x_sub_mean_sum_val * (x[i] - mean_val) *
                                  var_val * var_val / sample_size);
    }
    dscale_tmp = BlockReduce(dscale_tmp_storage).Reduce(dscale_tmp, cub::Sum());

    if (threadIdx.x == 0) {
      dscale[ncid] += dscale_tmp;
    }
    __syncthreads();
  }
}

template <typename T>
class InstanceNormDoubleGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *X = ctx.Input<Tensor>("X");
    const auto *Scale = ctx.Input<Tensor>("Scale");
    const auto *dY = ctx.Input<Tensor>("DY");
    const auto *Saved_mean = ctx.Input<Tensor>("SavedMean");
    const auto *Saved_variance = ctx.Input<Tensor>("SavedVariance");
    const auto *running_mean = ctx.Input<Tensor>("Mean");
    const auto *running_var = ctx.Input<Tensor>("Variance");
    const auto *ddX = ctx.Input<Tensor>("DDX");
    const auto *ddScale = ctx.Input<Tensor>("DDScale");
    const auto *ddBias = ctx.Input<Tensor>("DDBias");
    const double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));

    auto *dX = ctx.Output<Tensor>("DX");
    auto *dScale = ctx.Output<Tensor>("DScale");
    auto *ddY = ctx.Output<Tensor>("DDY");

    const T *x_data = X->data<T>();
    const T *dy_data = dY->data<T>();
    const T *ddx_data = (ddX == nullptr ? nullptr : ddX->data<T>());

    const T *ddscale_data = (ddScale == nullptr ? nullptr : ddScale->data<T>());
    const T *ddbias_data = (ddScale == nullptr ? nullptr : ddBias->data<T>());

    const T *mean_data = Saved_mean->data<T>();
    const T *variance_data = Saved_variance->data<T>();

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    math::SetConstant<platform::CUDADeviceContext, T> set_zero;

    auto &x_dims = X->dims();
    int N, C, H, W, D;
    ExtractNCWHD(x_dims, DataLayout::kNCHW, &N, &C, &H, &W, &D);
    int NxC = N * C;
    const int n = X->numel();
    int sample_size = n / N / C;

    Tensor scale_tmp;
    if (!Scale) {
      scale_tmp.mutable_data<T>({C}, ctx.GetPlace());
      set_zero(dev_ctx, &scale_tmp, static_cast<T>(1));
    }
    const T *scale_data = Scale ? Scale->data<T>() : scale_tmp.data<T>();

    const int block = 512;
    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    const int max_blocks = std::max(max_threads / block, 1);
    const int grid = NxC;
    const int grid1 = (C + block - 1) / block;

    if (dX) {
      T *dx_data = dX->mutable_data<T>(ctx.GetPlace());
      set_zero(dev_ctx, dX, static_cast<T>(0));
      DoubleGradComputeDX<T, block><<<grid, block, 0, dev_ctx.stream()>>>(
          x_data, mean_data, variance_data, ddx_data, dy_data, scale_data,
          ddscale_data, C, sample_size, epsilon, dx_data);
    }
    if (dScale) {
      Tensor dscale_tmp =
          ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({NxC}, dev_ctx);
      set_zero(dev_ctx, &dscale_tmp, static_cast<T>(0));
      T *dscale_tmp_data = dscale_tmp.mutable_data<T>(ctx.GetPlace());

      T *dscale_data = dScale->mutable_data<T>(ctx.GetPlace());
      set_zero(dev_ctx, dScale, static_cast<T>(0));
      DoubleGradComputeDScale<T, block><<<grid, block, 0, dev_ctx.stream()>>>(
          x_data, mean_data, variance_data, ddx_data, dy_data, C, sample_size,
          epsilon, dscale_tmp_data);
      add_param<T, block, false><<<grid1, block, 0, dev_ctx.stream()>>>(
          dscale_tmp.data<T>(), dScale->data<T>(), N, C);
    }
    if (ddY) {
      T *ddy_data = ddY->mutable_data<T>(ctx.GetPlace());
      set_zero(dev_ctx, ddY, static_cast<T>(0));
      DoubleGradComputeDDY<T, block><<<grid, block, 0, dev_ctx.stream()>>>(
          x_data, mean_data, variance_data, ddscale_data, ddbias_data, ddx_data,
          scale_data, C, sample_size, epsilon, ddy_data);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
REGISTER_OP_CUDA_KERNEL(
    instance_norm, ops::InstanceNormKernel<plat::CUDADeviceContext, float>);
REGISTER_OP_CUDA_KERNEL(
    instance_norm_grad,
    ops::InstanceNormGradKernel<plat::CUDADeviceContext, float>);
REGISTER_OP_CUDA_KERNEL(instance_norm_grad_grad,
                        ops::InstanceNormDoubleGradKernel<
                            paddle::platform::CUDADeviceContext, float>);
#else
REGISTER_OP_CUDA_KERNEL(
    instance_norm, ops::InstanceNormKernel<plat::CUDADeviceContext, float>,
    ops::InstanceNormKernel<plat::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    instance_norm_grad,
    ops::InstanceNormGradKernel<plat::CUDADeviceContext, float>,
    ops::InstanceNormGradKernel<plat::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    instance_norm_grad_grad,
    ops::InstanceNormDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                      float>,
    ops::InstanceNormDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                      double>);
#endif
