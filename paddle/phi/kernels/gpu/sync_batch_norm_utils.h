/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include <cmath>
#include <string>
#include <vector>
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/process_group_nccl.h"
#endif
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/kernels/funcs/norm_utils.h"

namespace phi {

template <typename T>
using CudnnDataType = phi::backends::gpu::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;

template <typename T, int BlockDim, DataLayout layout>
__global__ void KeLocalStats(
    const T *x, int N, int M, int C, BatchNormParamType<T> *mean_var) {
  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int k = blockIdx.x; k < C; k += gridDim.x) {
    BatchNormParamType<T> x_sum = 0.;
    BatchNormParamType<T> x2_sum = 0.;
    for (int i = threadIdx.x; i < N * M; i += BlockDim) {
      int id = layout == DataLayout::kNCHW ? (i / M) * C * M + k * M + i % M
                                           : i * C + k;
      auto x_in = static_cast<BatchNormParamType<T>>(x[id]);
      x_sum += x_in;
      x2_sum += x_in * x_in;
    }
    __syncthreads();
    auto out = BlockReduce(temp_storage).Reduce(x_sum, cub::Sum());
    __syncthreads();
    if (threadIdx.x == 0) {
      mean_var[k] = out / (N * M);
    }
    out = BlockReduce(temp_storage).Reduce(x2_sum, cub::Sum());
    __syncthreads();
    if (threadIdx.x == 0) {
      mean_var[k + C] = out / (N * M);
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    mean_var[2 * C] = static_cast<BatchNormParamType<T>>(1.0);
  }
}

template <typename T>
__global__ void KeSyncAndMovingStats(BatchNormParamType<T> *means,
                                     BatchNormParamType<T> *variances,
                                     BatchNormParamType<T> *num_dev,
                                     const int C,
                                     const BatchNormParamType<T> momentum,
                                     const double epsilon,
                                     BatchNormParamType<T> *sv_mean_data,
                                     BatchNormParamType<T> *sv_inv_var_data,
                                     BatchNormParamType<T> *moving_means,
                                     BatchNormParamType<T> *moving_variances) {
  // sync stats across multi-devices
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = gid; i < C; i += stride) {
    auto mean = means[i] / (*num_dev);
    auto var = variances[i] / (*num_dev);
    var = var - mean * mean;

    // sync stats
    sv_mean_data[i] = mean;
    sv_inv_var_data[i] = 1.0 / sqrt(var + epsilon);
    variances[i] = var;

    // moving stats
    moving_means[i] = moving_means[i] * momentum + mean * (1. - momentum);
    moving_variances[i] =
        moving_variances[i] * momentum + var * (1. - momentum);
  }
}

template <typename T, DataLayout layout>
static __global__ void KeNormAffine(const T *x,
                                    const BatchNormParamType<T> *scale,
                                    const BatchNormParamType<T> *bias,
                                    const BatchNormParamType<T> *mean,
                                    const BatchNormParamType<T> *variance,
                                    const double epsilon,
                                    const int C,
                                    const int M,
                                    const int num,
                                    T *y) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = gid; i < num; i += stride) {
    const int c = layout == DataLayout::kNCHW ? (i / M) % C : i % C;
    auto x_i = static_cast<BatchNormParamType<T>>(x[i]);
    auto y_i =
        (x_i - mean[c]) / sqrt(variance[c] + epsilon) * scale[c] + bias[c];
    y[i] = static_cast<T>(y_i);
  }
}

template <typename T, const int BlockDim, DataLayout layout>
__global__ void KeBackwardLocalStats(const T *dy,
                                     const T *x,
                                     const BatchNormParamType<T> *means,
                                     int N,
                                     int M,
                                     int C,
                                     BatchNormParamType<T> *sum_dy_prod) {
  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int k = blockIdx.x; k < C; k += gridDim.x) {
    BatchNormParamType<T> sum1 = 0.;
    BatchNormParamType<T> sum2 = 0.;
    auto mean = means[k];
    for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
      int id = layout == DataLayout::kNCHW ? (i / M) * C * M + k * M + i % M
                                           : i * C + k;
      auto g = static_cast<BatchNormParamType<T>>(dy[id]);
      sum1 += g;
      auto x_i = static_cast<BatchNormParamType<T>>(x[id]);
      sum2 += g * (x_i - mean);
    }

    __syncthreads();
    auto out = BlockReduce(temp_storage).Reduce(sum1, cub::Sum());
    __syncthreads();
    if (threadIdx.x == 0) {
      sum_dy_prod[k] = out;
    }
    out = BlockReduce(temp_storage).Reduce(sum2, cub::Sum());
    __syncthreads();
    if (threadIdx.x == 0) {
      sum_dy_prod[k + C] = out;
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    sum_dy_prod[2 * C] = 1.0;
  }
}

template <typename T, int BlockDim, DataLayout layout>
static __global__ void KeBNBackwardScaleBias(
    const T *dy,
    const T *x,
    const BatchNormParamType<T> *mean,
    const BatchNormParamType<T> *inv_variance,
    const double epsilon,
    const int N,
    const int C,
    const int HxW,
    BatchNormParamType<T> *dscale,
    BatchNormParamType<T> *dbias) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    BatchNormParamType<T> ds_sum = 0.;
    BatchNormParamType<T> db_sum = 0.;

    auto inv_var_i = inv_variance[i];
    auto mean_i = mean[i];
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int id = layout == DataLayout::kNCHW
                         ? ((j / HxW) * C + i) * HxW + (j % HxW)
                         : j * outer_size + i;
      auto x_i = static_cast<BatchNormParamType<T>>(x[id]);
      auto dy_i = static_cast<BatchNormParamType<T>>(dy[id]);
      ds_sum += dy_i * (x_i - mean_i);
      db_sum += dy_i;
    }
    __syncthreads();
    auto os = BlockReduce(temp_storage).Reduce(ds_sum, cub::Sum());
    __syncthreads();
    auto ob = BlockReduce(temp_storage).Reduce(db_sum, cub::Sum());
    __syncthreads();
    if (threadIdx.x == 0) {
      dscale[i] = os * inv_var_i;
      dbias[i] = ob;
    }
    __syncthreads();
  }
}

template <typename T, DataLayout layout>
static __global__ void KeBNRestoreData(T *x,
                                       const BatchNormParamType<T> *scale,
                                       const BatchNormParamType<T> *bias,
                                       const BatchNormParamType<T> *mean,
                                       const BatchNormParamType<T> *sv_inv,
                                       const double epsilon,
                                       int C,
                                       int M,
                                       int num,
                                       const T *y) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = gid; i < num; i += stride) {
    const int c = layout == DataLayout::kNCHW ? (i / M) % C : i % C;
    auto y_i = static_cast<BatchNormParamType<T>>(y[i]);
    auto x_i = (y_i - bias[c]) / scale[c] / sv_inv[c] + mean[c];
    x[i] = static_cast<T>(x_i);
  }
}

template <typename T, DataLayout layout>
static __global__ void KeBNBackwardData(
    const T *dy,
    const T *x,
    const BatchNormParamType<T> *gamma,
    const BatchNormParamType<T> *mean,
    const BatchNormParamType<T> *inv_variance,
    const BatchNormParamType<T> *g_sum_dy,
    const BatchNormParamType<T> *g_sum_dy_prod,
    const BatchNormParamType<T> *num_dev,
    const double epsilon,
    const int C,
    const int HxW,
    const int num,
    T *dx) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  auto scale = static_cast<BatchNormParamType<T>>(C) / num;
  auto dev_num = num_dev[0];
  for (int i = gid; i < num; i += stride) {
    const int c = layout == DataLayout::kNCHW ? i / HxW % C : i % C;
    auto inv_var = inv_variance[c];
    auto s_d = gamma[c];
    auto gvar =
        -(g_sum_dy_prod[c] / dev_num) * s_d * inv_var * (inv_var * inv_var);
    auto gmean = -(g_sum_dy[c] / dev_num) * s_d * inv_var;

    auto x_i = static_cast<BatchNormParamType<T>>(x[i]);
    auto dy_i = static_cast<BatchNormParamType<T>>(dy[i]);
    auto dx_i =
        dy_i * s_d * inv_var + gmean * scale + gvar * scale * (x_i - mean[c]);
    dx[i] = static_cast<T>(dx_i);
  }
}

template <typename T, typename Context>
void SyncBatchNormGradFunctor(
    const Context &ctx,
    const DenseTensor *input_x,
    const DenseTensor *input_y,
    const DenseTensor &scale,
    const DenseTensor &bias,
    // const paddle::optional<DenseTensor>& mean,
    // const paddle::optional<DenseTensor>& variance,
    const DenseTensor &saved_mean,
    const DenseTensor &saved_variance,
    // const paddle::optional<DenseTensor>& reserve_space,
    const DenseTensor &y_grad,
    // float momentum,
    float epsilon_f,
    const std::string &data_layout_str,
    // bool is_test,
    // bool use_global_stats,
    // bool trainable_statistics,
    // bool fuse_with_relu,
    DenseTensor *x_grad,
    DenseTensor *scale_grad,
    DenseTensor *bias_grad) {
  double epsilon = static_cast<double>(epsilon_f);

  const DataLayout layout = phi::StringToDataLayout(data_layout_str);

  const auto *d_y = &y_grad;

  auto *d_x = x_grad;
  auto *d_scale = scale_grad;
  auto *d_bias = bias_grad;

  const DenseTensor *x;
  bool is_inplace = false;
  if (input_y) {
    is_inplace = true;
    x = input_y;
  } else {
    x = input_x;
  }
  const auto &x_dims = x->dims();

  PADDLE_ENFORCE_GE(x_dims.size(),
                    2,
                    phi::errors::InvalidArgument(
                        "The Input X dim size should be larger than 1."));
  PADDLE_ENFORCE_LE(x_dims.size(),
                    5,
                    phi::errors::InvalidArgument(
                        "The Input X dim size should be less than 6."));

  int N, C, H, W, D;
  funcs::ExtractNCWHD(x_dims, layout, &N, &C, &H, &W, &D);
  PADDLE_ENFORCE_EQ(scale.dims()[0],
                    C,
                    phi::errors::InvalidArgument(
                        "Expected first dim for input parameter(scale) of "
                        "OP(sync_batch_norm) be (%d), but given (%d).",
                        C,
                        scale.dims()[0]));

  ctx.template Alloc<T>(d_x);
  if (d_scale && d_bias) {
    ctx.template Alloc<BatchNormParamType<T>>(d_scale);
    ctx.template Alloc<BatchNormParamType<T>>(d_bias);
  }
  PADDLE_ENFORCE_EQ(scale.dims().size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "Expected rank for input parameter(scale) of "
                        "OP(sync_batch_norm) be (1), but given (%d).",
                        scale.dims().size()));

  std::vector<int> dims;
  std::vector<int> strides;
  if (layout == DataLayout::kNCHW) {
    dims = {N, C, H, W, D};
    strides = {C * H * W * D, H * W * D, W * D, D, 1};
  } else {
    dims = {N, C, H, W, D};
    strides = {H * W * C * D, 1, W * D * C, D * C, C};
  }
  const T *x_d = x->data<T>();
  auto px = *x;
  const T *dy_d = d_y->data<T>();

  auto stream = ctx.stream();

  const auto *saved_mean_ptr =
      saved_mean.template data<BatchNormParamType<T>>();
  const auto *saved_inv_var =
      saved_variance.template data<BatchNormParamType<T>>();
  const int bytes = (C * 2 + 1) * sizeof(BatchNormParamType<T>);
  auto alloc_ptr = paddle::memory::Alloc(
      ctx.GetPlace(),
      bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  auto *stats = reinterpret_cast<BatchNormParamType<T> *>(alloc_ptr->ptr());

  const int block = 512;
  const int threads = 256;
  int x_numel = x->numel();
  int fsize = H * W * D;
  int max_threads = ctx.GetMaxPhysicalThreadCount();
  int grid = std::min(C, (max_threads + threads - 1) / threads);
  int grid2 = (std::min(x_numel, max_threads) + block - 1) / block;

  if (is_inplace) {
    if (layout == DataLayout::kNCHW) {
      KeBNRestoreData<T, DataLayout::kNCHW><<<grid2, block, 0, stream>>>(
          ctx.template Alloc<T>(&px),
          scale.template data<BatchNormParamType<T>>(),
          bias.template data<BatchNormParamType<T>>(),
          saved_mean_ptr,
          saved_inv_var,
          epsilon,
          C,
          H * W * D,
          x_numel,
          x->data<T>());
    } else {
      KeBNRestoreData<T, DataLayout::kNHWC><<<grid2, block, 0, stream>>>(
          ctx.template Alloc<T>(&px),
          scale.template data<BatchNormParamType<T>>(),
          bias.template data<BatchNormParamType<T>>(),
          saved_mean_ptr,
          saved_inv_var,
          epsilon,
          C,
          H * W * D,
          x_numel,
          x->data<T>());
    }
  }

  if (layout == DataLayout::kNCHW) {
    KeBackwardLocalStats<T, threads, DataLayout::kNCHW>
        <<<grid, threads, 0, stream>>>(
            dy_d, x_d, saved_mean_ptr, N, fsize, C, stats);
  } else {
    KeBackwardLocalStats<T, threads, DataLayout::kNHWC>
        <<<grid, threads, 0, stream>>>(
            dy_d, x_d, saved_mean_ptr, N, fsize, C, stats);
  }

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  int global_gid = 0;
  ncclComm_t comm = nullptr;

  if (paddle::distributed::ProcessGroupMapFromGid::getInstance()->has(
          global_gid)) {
    auto *nccl_pg = static_cast<paddle::distributed::ProcessGroupNCCL *>(
        paddle::distributed::ProcessGroupMapFromGid::getInstance()->get(
            global_gid));
    comm = nccl_pg->NCCLComm(x->place());
  } else {
    comm = ctx.nccl_comm();
  }

  if (comm) {
    int dtype = paddle::platform::ToNCCLDataType(scale.dtype());
    // In-place operation
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::ncclAllReduce(stats,
                                    stats,
                                    2 * C + 1,
                                    static_cast<ncclDataType_t>(dtype),
                                    ncclSum,
                                    comm,
                                    stream));
    VLOG(3) << "Sync result using all reduce";
  }
#endif

  if (layout == DataLayout::kNCHW) {
    if (d_scale && d_bias) {
      KeBNBackwardScaleBias<T, threads, DataLayout::kNCHW>
          <<<grid, threads, 0, stream>>>(dy_d,
                                         x_d,
                                         saved_mean_ptr,
                                         saved_inv_var,
                                         epsilon,
                                         N,
                                         C,
                                         fsize,
                                         d_scale->data<BatchNormParamType<T>>(),
                                         d_bias->data<BatchNormParamType<T>>());
    }
    if (d_x) {
      KeBNBackwardData<T, DataLayout::kNCHW><<<grid2, block, 0, stream>>>(
          dy_d,
          x_d,
          scale.template data<BatchNormParamType<T>>(),
          saved_mean_ptr,
          saved_inv_var,
          stats,
          stats + C,
          stats + 2 * C,
          epsilon,
          C,
          fsize,
          x->numel(),
          d_x->data<T>());
    }
  } else {
    if (d_scale && d_bias) {
      KeBNBackwardScaleBias<T, threads, DataLayout::kNHWC>
          <<<grid, threads, 0, stream>>>(dy_d,
                                         x_d,
                                         saved_mean_ptr,
                                         saved_inv_var,
                                         epsilon,
                                         N,
                                         C,
                                         fsize,
                                         d_scale->data<BatchNormParamType<T>>(),
                                         d_bias->data<BatchNormParamType<T>>());
    }
    if (d_x) {
      KeBNBackwardData<T, DataLayout::kNHWC><<<grid2, block, 0, stream>>>(
          dy_d,
          x_d,
          scale.template data<BatchNormParamType<T>>(),
          saved_mean_ptr,
          saved_inv_var,
          stats,
          stats + C,
          stats + 2 * C,
          epsilon,
          C,
          fsize,
          x->numel(),
          d_x->data<T>());
    }
  }
}

}  // namespace phi
