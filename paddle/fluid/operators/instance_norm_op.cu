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
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"
#include "paddle/phi/kernels/funcs/math_function.h"

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
    phi::funcs::SetConstant<platform::CUDADeviceContext, T> set_zero;

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
REGISTER_OP_CUDA_KERNEL(instance_norm_grad_grad,
                        ops::InstanceNormDoubleGradKernel<
                            paddle::platform::CUDADeviceContext, float>);
#else
REGISTER_OP_CUDA_KERNEL(
    instance_norm_grad_grad,
    ops::InstanceNormDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                      float>,
    ops::InstanceNormDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                      double>);
#endif
