/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
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
#include "paddle/fluid/operators/math/math_function.h"
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/miopen_helper.h"
#else
#include "paddle/fluid/platform/cudnn_helper.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;

// math: dx = scale * ((x - mean) * inv_var / NxHxW * (np.mean(ddx,
// axis=(n,h,w)) *
//          np.sum(dy, axis=(n,h,w)) -
//          np.sum(dy * ddx, axis=(n,h,w)) + 3 * np.mean(dy * (x -
//          mean),
//          axis=(n,h,w)) * inv_var.pow(2) *
//          np.sum(ddx * (x - mean), axis=(n,h,w))) + inv_var.pow(3) /
//          NxHxW *
//          np.sum(ddx * (x - mean)) *
//          (np.mean(dy, axis=(n,h,w)) - dy) + inv_var.pow(3) / NxHxW *
//          np.sum(dy,
//          axis=(n,h,w)) * (x - mean) *
//          (np.mean(ddx, axis=(n,h,w)) - ddx)) + ddr * (dy * inv_var -
//          inv_var
//          *
//          np.mean(dy, axis=(n,h,w)) -
//          inv_var.pow(3) * (x - mean) * np.mean(dy * (x - mean),
//          axis=(n,h,w)))

template <typename T, int BlockDim, framework::DataLayout layout>
__global__ void DoubleGradComputeDX(const T *x, const T *mean,
                                    const T *variance, const T *ddx,
                                    const T *dy, const T *scale,
                                    const T *ddscale, const int N, const int C,
                                    const int sample_size, const double epsilon,
                                    T *dx) {
  const int outer_size = C;
  const int inner_size = N * sample_size;

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

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T mean_val = mean[i];
    T var_val = variance[i];
    T dy_sum = 0;
    T ddx_sum = 0;
    T dy_mul_ddx_sum = 0;
    T dy_mul_x_sub_mean_sum = 0;
    T ddx_mul_x_sub_mean_sum = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index =
          layout == framework::DataLayout::kNCHW
              ? (j / sample_size * C + i) * sample_size + j % sample_size
              : j * outer_size + i;
      T ddx_i = ddx[index];
      T dy_i = dy[index];
      T tmp = x[index] - mean_val;

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
      for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
        const int index =
            layout == framework::DataLayout::kNCHW
                ? (j / sample_size * C + i) * sample_size + j % sample_size
                : j * outer_size + i;
        dx[index] +=
            ((x[index] - mean_val) * var_val * var_val * var_val / inner_size *
                 (ddx_sum_val * dy_sum_val / inner_size - dy_mul_ddx_sum_val +
                  3. * dy_mul_x_sub_mean_sum_val * var_val *
                      ddx_mul_x_sub_mean_sum_val * var_val / inner_size) +
             ddx_mul_x_sub_mean_sum_val * var_val / inner_size * var_val *
                 var_val * (dy_sum_val / inner_size - dy[index]) +
             dy_mul_x_sub_mean_sum_val * var_val / inner_size * var_val *
                 var_val * (ddx_sum_val / inner_size - ddx[index])) *
            scale[i];
      }
    }
    __syncthreads();
    if (ddscale != nullptr) {
      for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
        const int index =
            layout == framework::DataLayout::kNCHW
                ? (j / sample_size * C + i) * sample_size + j % sample_size
                : j * outer_size + i;
        dx[index] += (dy[index] * var_val - dy_sum_val / inner_size * var_val -
                      (x[index] - mean_val) * var_val * var_val *
                          dy_mul_x_sub_mean_sum_val * var_val / inner_size) *
                     ddscale[i];
      }
    }
  }
}

// math: ddy = (x - mean) * inv_var * ddscale + ddbias +
//           scale * inv_var * (ddx - (x - mean) * inv_var.pow(2) *
//           np.mean(ddx * (x - mean), axis=(n,h,w)))
template <typename T, int BlockDim, framework::DataLayout layout>
__global__ void DoubleGradComputeDDY(const T *x, const T *mean,
                                     const T *variance, const T *ddscale,
                                     const T *ddbias, const T *ddx,
                                     const T *scale, const int N, const int C,
                                     const int sample_size,
                                     const double epsilon, T *ddy) {
  const int outer_size = C;
  const int inner_size = N * sample_size;

  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ddx_storage;
  __shared__ typename BlockReduce::TempStorage ddx_mul_x_sub_mean_storage;
  __shared__ T ddx_sum_val;
  __shared__ T ddx_mul_x_sub_mean_sum_val;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T mean_val = mean[i];
    T var_val = variance[i];
    T ddx_sum = 0;
    T ddx_mul_x_sub_mean_sum = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index =
          layout == framework::DataLayout::kNCHW
              ? (j / sample_size * C + i) * sample_size + j % sample_size
              : j * outer_size + i;
      T ddx_i = ddx[index];
      ddx_sum += ddx_i;
      ddx_mul_x_sub_mean_sum += (ddx_i * (x[index] - mean_val));
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
      for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
        const int index =
            layout == framework::DataLayout::kNCHW
                ? (j / sample_size * C + i) * sample_size + j % sample_size
                : j * outer_size + i;
        ddy[index] += scale[i] * var_val *
                      (ddx[index] - ddx_sum_val / inner_size -
                       (x[index] - mean_val) * var_val *
                           ddx_mul_x_sub_mean_sum_val * var_val / inner_size);
      }
    }
    __syncthreads();
    if (ddscale != nullptr) {
      for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
        const int index =
            layout == framework::DataLayout::kNCHW
                ? (j / sample_size * C + i) * sample_size + j % sample_size
                : j * outer_size + i;
        ddy[index] += (x[index] - mean_val) * var_val * ddscale[i];
      }
    }
    __syncthreads();
    if (ddbias != nullptr) {
      for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
        const int index =
            layout == framework::DataLayout::kNCHW
                ? (j / sample_size * C + i) * sample_size + j % sample_size
                : j * outer_size + i;
        ddy[index] += ddbias[i];
      }
    }
  }
}

// math: dscale = inv_var * (dy - np.mean(dy, axis=(n,h,w) - (x-mean) *
//            inv_var.pow(2) * np.mean(dy * (x-mean), axis=(n,h,w)))) *
//            ddx
template <typename T, int BlockDim, framework::DataLayout layout>
__global__ void DoubleGradComputeDScale(const T *x, const T *mean,
                                        const T *variance, const T *ddx,
                                        const T *dy, const int N, const int C,
                                        const int sample_size,
                                        const double epsilon, T *dscale) {
  const int outer_size = C;
  const int inner_size = N * sample_size;

  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage dy_storage;
  __shared__ typename BlockReduce::TempStorage dy_mul_x_sub_mean_storage;
  __shared__ typename BlockReduce::TempStorage dscale_tmp_storage;
  __shared__ T dy_sum_val;
  __shared__ T dy_mul_x_sub_mean_sum_val;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T dy_sum = 0;
    T dy_mul_x_sub_mean_sum = 0;
    T mean_val = mean[i];
    T var_val = variance[i];
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index =
          layout == framework::DataLayout::kNCHW
              ? (j / sample_size * C + i) * sample_size + j % sample_size
              : j * outer_size + i;
      T dy_i = dy[index];
      dy_sum += dy_i;
      dy_mul_x_sub_mean_sum += (dy_i * (x[index] - mean_val));
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
      for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
        const int index =
            layout == framework::DataLayout::kNCHW
                ? (j / sample_size * C + i) * sample_size + j % sample_size
                : j * outer_size + i;
        dscale_tmp += ddx[index] * var_val *
                      (dy[index] - dy_sum_val / inner_size -
                       dy_mul_x_sub_mean_sum_val * (x[index] - mean_val) *
                           var_val * var_val / inner_size);
      }
      dscale_tmp =
          BlockReduce(dscale_tmp_storage).Reduce(dscale_tmp, cub::Sum());

      if (threadIdx.x == 0) {
        dscale[i] += dscale_tmp;
      }
      __syncthreads();
    }
  }
}

// math: dscale = np.sum(ddx * dy, axis=(n,h,w)) * inv_var
template <typename T, int BlockDim, framework::DataLayout layout>
__global__ void DoubleGradComputeDScaleWithGlobal(
    const T *ddx, const T *variance, const T *dy, const double epsilon,
    const int N, const int C, const int sample_size, T *dscale) {
  int outer_size = C;
  int inner_size = N * sample_size;
  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ddx_mul_dy_storage;
  __shared__ T ddx_mul_dy_sum_val;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T inv_var_i = 1.0 / sqrt(variance[i] + epsilon);
    T ddx_mul_dy_sum = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index =
          layout == framework::DataLayout::kNCHW
              ? (j / sample_size * C + i) * sample_size + j % sample_size
              : j * outer_size + i;
      T ddx_i = ddx[index];
      T dy_i = dy[index];
      ddx_mul_dy_sum += (ddx_i * dy_i);
    }
    ddx_mul_dy_sum =
        BlockReduce(ddx_mul_dy_storage).Reduce(ddx_mul_dy_sum, cub::Sum());
    if (threadIdx.x == 0) {
      ddx_mul_dy_sum_val = ddx_mul_dy_sum;
    }
    __syncthreads();

    if (ddx != nullptr) {
      dscale[i] = inv_var_i * ddx_mul_dy_sum_val;
    }
  }
}

// math: dx = ddscale * dy * inv_var
template <typename T, framework::DataLayout layout>
__global__ void DoubleGradComputeDXWithGlobal(const T *dy, const T *ddscale,
                                              const T *variance,
                                              const double epsilon, const int C,
                                              const int sample_size,
                                              const int num, T *dx) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  if (ddscale != nullptr) {
    for (int i = gid; i < num; i += stride) {
      const int c =
          layout == framework::DataLayout::kNCHW ? i / sample_size % C : i % C;
      T inv_var = 1.0 / sqrt(variance[c] + epsilon);
      dx[i] = dy[i] * ddscale[c] * inv_var;
    }
  }
}

// math: ddy = scale * ddx * inv_var + ddbias +
//             ddscale * (x - mean) * inv_var
template <typename T, framework::DataLayout layout>
__global__ void DoubleGradComputeDDYWithGlobal(
    const T *ddx, const T *scale, const T *mean, const T *variance, const T *x,
    const T *ddbias, const T *ddscale, const double epsilon, const int C,
    const int sample_size, const int num, T *ddy) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  if (ddx != nullptr) {
    for (int i = gid; i < num; i += stride) {
      const int c =
          layout == framework::DataLayout::kNCHW ? i / sample_size % C : i % C;
      T inv_var = 1.0 / sqrt(variance[c] + epsilon);
      ddy[i] += ddx[i] * scale[c] * inv_var;
    }
  }
  __syncthreads();
  if (ddscale != nullptr) {
    for (int i = gid; i < num; i += stride) {
      const int c =
          layout == framework::DataLayout::kNCHW ? i / sample_size % C : i % C;
      T inv_var = 1.0 / sqrt(variance[c] + epsilon);
      ddy[i] += (x[i] - mean[c]) * inv_var * ddscale[c];
    }
  }
  __syncthreads();
  if (ddbias != nullptr) {
    for (int i = gid; i < num; i += stride) {
      const int c =
          layout == framework::DataLayout::kNCHW ? i / sample_size % C : i % C;
      ddy[i] += ddbias[c];
    }
  }
}

template <typename DeviceContext, typename T>
void NormDoubleGradFunctor(const framework::ExecutionContext &ctx,
                           const DataLayout data_layout, const Tensor *X,
                           const Tensor *Scale, const Tensor *dY,
                           const Tensor *Saved_mean,
                           const Tensor *Saved_variance, const double epsilon,
                           const bool use_global_stats, const Tensor *ddX,
                           const Tensor *ddScale, const Tensor *ddBias,
                           Tensor *dX, Tensor *dScale, Tensor *ddY) {
  const T *x_data = X->data<T>();
  const T *dy_data = dY->data<T>();
  const T *ddx_data = (ddX == nullptr ? nullptr : ddX->data<T>());

  const T *ddscale_data = (ddScale == nullptr ? nullptr : ddScale->data<T>());
  const T *ddbias_data = (ddBias == nullptr ? nullptr : ddBias->data<T>());

  auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
  math::SetConstant<platform::CUDADeviceContext, T> set_constant;

  auto &x_dims = X->dims();
  const int C = (data_layout == DataLayout::kNCHW ? x_dims[1]
                                                  : x_dims[x_dims.size() - 1]);
  const int N = x_dims[0];
  const int num = X->numel();
  const int sample_size = num / N / C;
  Tensor scale_tmp;
  if (!Scale) {
    scale_tmp.mutable_data<T>({C}, ctx.GetPlace());
    set_constant(dev_ctx, &scale_tmp, static_cast<T>(1));
  }
  const T *scale_data = Scale ? Scale->data<T>() : scale_tmp.data<T>();

  const int block = 512;
  int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
  const int max_blocks = std::max(max_threads / block, 1);
  int grid = std::min(C, max_blocks);
  int grid1 = (num + block - 1) / block;

  const T *mean_data, *variance_data;
  if (use_global_stats) {
    const auto *running_mean = ctx.Input<Tensor>("Mean");
    const auto *running_var = ctx.Input<Tensor>("Variance");
    const auto *running_mean_data = running_mean->template data<T>();
    const auto *running_var_data = running_var->template data<T>();
    mean_data = running_mean_data;
    variance_data = running_var_data;
  } else {
    const T *smean_data = Saved_mean->data<T>();
    const T *svariance_data = Saved_variance->data<T>();
    mean_data = smean_data;
    variance_data = svariance_data;
  }

  if (dX) {
    T *dx_data = dX->mutable_data<T>(ctx.GetPlace());
    set_constant(dev_ctx, dX, static_cast<T>(0));
    if (use_global_stats) {
      if (data_layout == DataLayout::kNHWC) {
        DoubleGradComputeDXWithGlobal<
            T, DataLayout::kNHWC><<<grid1, block, 0, dev_ctx.stream()>>>(
            dy_data, ddscale_data, variance_data, epsilon, C, sample_size, num,
            dx_data);
      } else {
        DoubleGradComputeDXWithGlobal<
            T, DataLayout::kNCHW><<<grid1, block, 0, dev_ctx.stream()>>>(
            dy_data, ddscale_data, variance_data, epsilon, C, sample_size, num,
            dx_data);
      }
    } else {
      if (data_layout == DataLayout::kNHWC) {
        DoubleGradComputeDX<
            T, block, DataLayout::kNHWC><<<grid, block, 0, dev_ctx.stream()>>>(
            x_data, mean_data, variance_data, ddx_data, dy_data, scale_data,
            ddscale_data, N, C, sample_size, epsilon, dx_data);
      } else {
        DoubleGradComputeDX<
            T, block, DataLayout::kNCHW><<<grid, block, 0, dev_ctx.stream()>>>(
            x_data, mean_data, variance_data, ddx_data, dy_data, scale_data,
            ddscale_data, N, C, sample_size, epsilon, dx_data);
      }
    }
  }
  if (dScale) {
    T *dscale_data = dScale->mutable_data<T>(ctx.GetPlace());
    set_constant(dev_ctx, dScale, static_cast<T>(0));
    if (use_global_stats) {
      if (data_layout == DataLayout::kNHWC) {
        DoubleGradComputeDScaleWithGlobal<
            T, block, DataLayout::kNHWC><<<grid, block, 0, dev_ctx.stream()>>>(
            ddx_data, variance_data, dy_data, epsilon, N, C, sample_size,
            dscale_data);
      } else {
        DoubleGradComputeDScaleWithGlobal<
            T, block, DataLayout::kNCHW><<<grid, block, 0, dev_ctx.stream()>>>(
            ddx_data, variance_data, dy_data, epsilon, N, C, sample_size,
            dscale_data);
      }
    } else {
      if (data_layout == DataLayout::kNHWC) {
        DoubleGradComputeDScale<
            T, block, DataLayout::kNHWC><<<grid, block, 0, dev_ctx.stream()>>>(
            x_data, mean_data, variance_data, ddx_data, dy_data, N, C,
            sample_size, epsilon, dscale_data);
      } else {
        DoubleGradComputeDScale<
            T, block, DataLayout::kNCHW><<<grid, block, 0, dev_ctx.stream()>>>(
            x_data, mean_data, variance_data, ddx_data, dy_data, N, C,
            sample_size, epsilon, dscale_data);
      }
    }
  }
  if (ddY) {
    T *ddy_data = ddY->mutable_data<T>(ctx.GetPlace());
    set_constant(dev_ctx, ddY, static_cast<T>(0));
    if (use_global_stats) {
      if (data_layout == DataLayout::kNHWC) {
        DoubleGradComputeDDYWithGlobal<
            T, DataLayout::kNHWC><<<grid1, block, 0, dev_ctx.stream()>>>(
            ddx_data, scale_data, mean_data, variance_data, x_data, ddbias_data,
            ddscale_data, epsilon, C, sample_size, num, ddy_data);
      } else {
        DoubleGradComputeDDYWithGlobal<
            T, DataLayout::kNCHW><<<grid1, block, 0, dev_ctx.stream()>>>(
            ddx_data, scale_data, mean_data, variance_data, x_data, ddbias_data,
            ddscale_data, epsilon, C, sample_size, num, ddy_data);
      }
    } else {
      if (data_layout == DataLayout::kNHWC) {
        DoubleGradComputeDDY<
            T, block, DataLayout::kNHWC><<<grid, block, 0, dev_ctx.stream()>>>(
            x_data, mean_data, variance_data, ddscale_data, ddbias_data,
            ddx_data, scale_data, N, C, sample_size, epsilon, ddy_data);
      } else {
        DoubleGradComputeDDY<
            T, block, DataLayout::kNCHW><<<grid, block, 0, dev_ctx.stream()>>>(
            x_data, mean_data, variance_data, ddscale_data, ddbias_data,
            ddx_data, scale_data, N, C, sample_size, epsilon, ddy_data);
      }
    }
  }
}

}  // namespace operators
}  // namespace paddle
