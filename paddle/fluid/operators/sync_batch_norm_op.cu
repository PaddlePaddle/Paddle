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
#include "cub/cub.cuh"
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/operators/batch_norm_op.h"
#include "paddle/fluid/platform/cudnn_helper.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/nccl_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;
template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;

template <typename T, int BlockDim, framework::DataLayout layout>
__global__ void KeLocalStats(const T *x, int N, int M, int C, T *mean_var) {
  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int k = blockIdx.x; k < C; k += gridDim.x) {
    T x_sum = 0;
    T x2_sum = 0;
    for (int i = threadIdx.x; i < N * M; i += BlockDim) {
      int id = layout == framework::DataLayout::kNCHW
                   ? (i / M) * C * M + k * M + i % M
                   : i * C + k;
      T x_in = x[id];
      x_sum += x_in;
      x2_sum += x_in * x_in;
    }
    __syncthreads();
    T out = BlockReduce(temp_storage).Reduce(x_sum, cub::Sum());
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
    mean_var[2 * C] = static_cast<T>(1.0);
  }
}

template <typename T>
__global__ void KeSyncAndMovingStats(T *means, T *variances, T *num_dev,
                                     const int C, const T momentum,
                                     const double epsilon, T *sv_mean_data,
                                     T *sv_inv_var_data, T *moving_means,
                                     T *moving_variances) {
  // sync stats across multi-devices
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = gid; i < C; i += stride) {
    T mean = means[i] / (*num_dev);
    T var = variances[i] / (*num_dev);
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

template <typename T, framework::DataLayout layout>
static __global__ void KeNormAffine(const T *x, const T *scale, const T *bias,
                                    const T *mean, const T *variance,
                                    const double epsilon, const int C,
                                    const int M, const int num, T *y) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = gid; i < num; i += stride) {
    const int c = layout == framework::DataLayout::kNCHW ? (i / M) % C : i % C;
    y[i] = (x[i] - mean[c]) / sqrt(variance[c] + epsilon) * scale[c] + bias[c];
  }
}

template <typename DeviceContext, typename T>
class SyncBatchNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    const float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");
    const std::string layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout layout = framework::StringToDataLayout(layout_str);
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    PADDLE_ENFORCE(
        !use_global_stats,
        "sync_batch_norm doesn't support to set use_global_stats True. ",
        "Please use batch_norm in this case.");

    const auto *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE(x_dims.size() >= 2 && x_dims.size() <= 5,
                   "The Input dim size should be between 2 and 5");
    int N, C, H, W, D;
    ExtractNCWHD(x_dims, layout, &N, &C, &H, &W, &D);
    int x_numel = x->numel();

    const T *x_d = x->data<T>();
    const T *s_d = ctx.Input<Tensor>("Scale")->data<T>();
    const T *b_d = ctx.Input<Tensor>("Bias")->data<T>();

    auto *y = ctx.Output<Tensor>("Y");
    T *y_d = y->mutable_data<T>(ctx.GetPlace());

    const T *mean_data = nullptr;
    const T *var_data = nullptr;

    auto &dev_ctx = ctx.cuda_device_context();
    auto stream = dev_ctx.stream();
    auto *comm = dev_ctx.nccl_comm();
    const int block = 512;
    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();

    paddle::memory::AllocationPtr alloc_ptr{nullptr};

    if (is_test) {
      const auto *est_mean = ctx.Input<Tensor>("Mean");
      const auto *est_var = ctx.Input<Tensor>("Variance");
      mean_data = est_mean->data<T>();
      var_data = est_var->data<T>();
    } else {
      auto &allocator =
          platform::DeviceTemporaryAllocator::Instance().Get(dev_ctx);
      // x, x^2, 1, here 1 is used to calc device num
      // device num also can be got from platform::DeviceContextPool
      const int bytes = (C * 2 + 1) * sizeof(T);
      alloc_ptr = allocator.Allocate(bytes);

      T *stats = reinterpret_cast<T *>(alloc_ptr->ptr());
      const int threads = 256;
      int grid = std::min(C, (max_threads + threads - 1) / threads);
      if (layout == framework::DataLayout::kNCHW) {
        KeLocalStats<
            T, threads,
            framework::DataLayout::kNCHW><<<grid, threads, 0, stream>>>(
            x_d, N, H * W * D, C, stats);
      } else {
        KeLocalStats<
            T, threads,
            framework::DataLayout::kNHWC><<<grid, threads, 0, stream>>>(
            x_d, N, H * W * D, C, stats);
      }

      Tensor c_g_st;
      T *c_g_st_d = c_g_st.mutable_data<T>({2 * C + 1}, platform::CPUPlace());
      auto gplace = boost::get<platform::CUDAPlace>(ctx.GetPlace());
      memory::Copy(platform::CPUPlace(), c_g_st_d, gplace, stats, bytes, 0);

      int dtype = platform::ToNCCLDataType(x->type());
      // In-place operation
      PADDLE_ENFORCE(platform::dynload::ncclAllReduce(
          stats, stats, 2 * C + 1, static_cast<ncclDataType_t>(dtype), ncclSum,
          comm, stream));

      // moving mean/variance
      auto *mean_out = ctx.Output<Tensor>("MeanOut");
      auto *variance_out = ctx.Output<Tensor>("VarianceOut");
      T *est_mean_data = mean_out->mutable_data<T>(ctx.GetPlace());
      T *est_var_data = variance_out->mutable_data<T>(ctx.GetPlace());

      auto *saved_mean = ctx.Output<Tensor>("SavedMean");
      auto *saved_inv_variance = ctx.Output<Tensor>("SavedVariance");
      T *sv_mean_data = saved_mean->mutable_data<T>(ctx.GetPlace());
      T *sv_inv_var_data = saved_inv_variance->mutable_data<T>(ctx.GetPlace());

      // Note, Input('Mean')/Input('Variance') share variable with
      // Output('MeanOut')/Output('VarianceOut')
      KeSyncAndMovingStats<T><<<(C + block - 1) / block, block, 0, stream>>>(
          stats, stats + C, stats + 2 * C, C, momentum, epsilon, sv_mean_data,
          sv_inv_var_data, est_mean_data, est_var_data);

      mean_data = sv_mean_data;
      var_data = stats + C;
    }

    int grid2 = (std::min(x_numel, max_threads) + block - 1) / block;
    if (layout == framework::DataLayout::kNCHW) {
      KeNormAffine<T,
                   framework::DataLayout::kNCHW><<<grid2, block, 0, stream>>>(
          x_d, s_d, b_d, mean_data, var_data, epsilon, C, H * W * D, x_numel,
          y_d);
    } else {
      KeNormAffine<T,
                   framework::DataLayout::kNHWC><<<grid2, block, 0, stream>>>(
          x_d, s_d, b_d, mean_data, var_data, epsilon, C, H * W * D, x_numel,
          y_d);
    }
  }
};

template <typename T, const int BlockDim, framework::DataLayout layout>
__global__ void KeBackwardLocalStats(const T *dy, const T *x, const T *means,
                                     int N, int M, int C, T *sum_dy_prod) {
  typedef cub::BlockReduce<double, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int k = blockIdx.x; k < C; k += gridDim.x) {
    T sum1 = 0;
    T sum2 = 0;
    T mean = means[k];
    for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
      int id = layout == framework::DataLayout::kNCHW
                   ? (i / M) * C * M + k * M + i % M
                   : i * C + k;
      T g = dy[id];
      sum1 += g;
      sum2 += g * (x[id] - mean);
    }

    __syncthreads();
    T out = BlockReduce(temp_storage).Reduce(sum1, cub::Sum());
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
    sum_dy_prod[2 * C] = static_cast<T>(1.0);
  }
}

template <typename T, int BlockDim, framework::DataLayout layout>
static __global__ void KeBNBackwardScaleBias(const T *dy, const T *x,
                                             const T *mean,
                                             const T *inv_variance,
                                             const double epsilon, const int N,
                                             const int C, const int HxW,
                                             T *dscale, T *dbias) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  typedef cub::BlockReduce<double, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T ds_sum = static_cast<T>(0);
    T db_sum = static_cast<T>(0);

    T inv_var_i = inv_variance[i];
    T mean_i = mean[i];
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int id = layout == framework::DataLayout::kNCHW
                         ? ((j / HxW) * C + i) * HxW + (j % HxW)
                         : j * outer_size + i;
      ds_sum += dy[id] * (x[id] - mean_i);
      db_sum += dy[id];
    }
    __syncthreads();
    double os = BlockReduce(temp_storage)
                    .Reduce(static_cast<double>(ds_sum), cub::Sum());
    __syncthreads();
    double ob = BlockReduce(temp_storage)
                    .Reduce(static_cast<double>(db_sum), cub::Sum());
    __syncthreads();
    if (threadIdx.x == 0) {
      dscale[i] = static_cast<T>(os * inv_var_i);
      dbias[i] = static_cast<T>(ob);
    }
    __syncthreads();
  }
}

template <typename T, framework::DataLayout layout>
static __global__ void KeBNBackwardData(const T *dy, const T *x, const T *beta,
                                        const T *mean, const T *inv_variance,
                                        const T *g_sum_dy,
                                        const T *g_sum_dy_prod,
                                        const T *num_dev, const double epsilon,
                                        const int C, const int HxW,
                                        const int num, T *dx) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  T scale = static_cast<T>(C) / num;
  T dev_num = num_dev[0];
  for (int i = gid; i < num; i += stride) {
    const int c = layout == framework::DataLayout::kNCHW ? i / HxW % C : i % C;
    T inv_var = inv_variance[c];
    T s_d = beta[c];
    T gvar = -1.0 * (g_sum_dy_prod[c] / dev_num) * s_d * inv_var *
             (inv_var * inv_var);
    T gmean = -1.0 * (g_sum_dy[c] / dev_num) * s_d * inv_var;

    dx[i] =
        dy[i] * s_d * inv_var + gmean * scale + gvar * scale * (x[i] - mean[c]);
  }
}

// Deriving the Gradient for the Backward Pass of Batch Normalization
// https://kevinzakka.github.io/2016/09/14/batch_normalization/
template <typename DeviceContext, typename T>
class SyncBatchNormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    const std::string layout_str = ctx.Attr<std::string>("data_layout");

    const DataLayout layout = framework::StringToDataLayout(layout_str);
    const auto *x = ctx.Input<Tensor>("X");
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");

    const auto &x_dims = x->dims();

    PADDLE_ENFORCE(x_dims.size() >= 2 && x_dims.size() <= 5,
                   "The Input dim size should be between 2 and 5");
    int N, C, H, W, D;
    ExtractNCWHD(x_dims, layout, &N, &C, &H, &W, &D);

    // init output
    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    d_x->mutable_data<T>(ctx.GetPlace());
    if (d_scale && d_bias) {
      d_scale->mutable_data<T>(ctx.GetPlace());
      d_bias->mutable_data<T>(ctx.GetPlace());
    }
    PADDLE_ENFORCE_EQ(scale->dims().size(), 1UL);
    PADDLE_ENFORCE_EQ(scale->dims()[0], C);

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
    const T *dy_d = d_y->data<T>();

    auto &dev_ctx = ctx.cuda_device_context();
    auto stream = dev_ctx.stream();
    auto *comm = dev_ctx.nccl_comm();

    const T *saved_mean = ctx.Input<Tensor>("SavedMean")->data<T>();
    const T *saved_inv_var = ctx.Input<Tensor>("SavedVariance")->data<T>();
    auto &allocator =
        platform::DeviceTemporaryAllocator::Instance().Get(dev_ctx);
    const int bytes = (C * 2 + 1) * sizeof(T);
    auto alloc_ptr = allocator.Allocate(bytes);
    T *stats = reinterpret_cast<T *>(alloc_ptr->ptr());

    const int threads = 256;
    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    int grid = std::min(C, (max_threads + threads - 1) / threads);
    int x_numel = x->numel();
    int fsize = H * W * D;

    if (layout == framework::DataLayout::kNCHW) {
      KeBackwardLocalStats<
          T, threads,
          framework::DataLayout::kNCHW><<<grid, threads, 0, stream>>>(
          dy_d, x_d, saved_mean, N, fsize, C, stats);
    } else {
      KeBackwardLocalStats<
          T, threads,
          framework::DataLayout::kNHWC><<<grid, threads, 0, stream>>>(
          dy_d, x_d, saved_mean, N, fsize, C, stats);
    }
    int dtype = platform::ToNCCLDataType(x->type());
    // In-place operation
    PADDLE_ENFORCE(platform::dynload::ncclAllReduce(
        stats, stats, 2 * C + 1, static_cast<ncclDataType_t>(dtype), ncclSum,
        comm, stream));

    const int block = 512;
    int grid2 = (std::min(x_numel, max_threads) + block - 1) / block;
    if (layout == framework::DataLayout::kNCHW) {
      if (d_scale && d_bias) {
        KeBNBackwardScaleBias<
            T, threads,
            framework::DataLayout::kNCHW><<<grid, threads, 0, stream>>>(
            dy_d, x_d, saved_mean, saved_inv_var, epsilon, N, C, fsize,
            d_scale->data<T>(), d_bias->data<T>());
      }
      if (d_x) {
        KeBNBackwardData<
            T, framework::DataLayout::kNCHW><<<grid2, block, 0, stream>>>(
            dy_d, x_d, scale->data<T>(), saved_mean, saved_inv_var, stats,
            stats + C, stats + 2 * C, epsilon, C, fsize, x->numel(),
            d_x->data<T>());
      }
    } else {
      if (d_scale && d_bias) {
        KeBNBackwardScaleBias<
            T, threads,
            framework::DataLayout::kNHWC><<<grid, threads, 0, stream>>>(
            dy_d, x_d, saved_mean, saved_inv_var, epsilon, N, C, fsize,
            d_scale->data<T>(), d_bias->data<T>());
      }
      if (d_x) {
        KeBNBackwardData<
            T, framework::DataLayout::kNHWC><<<grid2, block, 0, stream>>>(
            dy_d, x_d, scale->data<T>(), saved_mean, saved_inv_var, stats,
            stats + C, stats + 2 * C, epsilon, C, fsize, x->numel(),
            d_x->data<T>());
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    sync_batch_norm, ops::SyncBatchNormKernel<plat::CUDADeviceContext, float>,
    ops::SyncBatchNormKernel<plat::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    sync_batch_norm_grad,
    ops::SyncBatchNormGradKernel<plat::CUDADeviceContext, float>,
    ops::SyncBatchNormGradKernel<plat::CUDADeviceContext, double>);
