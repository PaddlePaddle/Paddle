/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
  T x_sum = 0;
  T x2_sum = 0;
  if (layout == framework::DataLayout::kNCHW) {
    // for (int i = threadIdx.y; i < N; i += blockDim.y) {
    for (int i = 0; i < N; i += 1) {
      for (int j = threadIdx.x; j < M; j += blockDim.x) {
        int id = i * C * M + blockIdx.x * M + j;
        T x_in = x[id];
        x_sum += x_in;
        x2_sum += x_in * x_in;
      }
    }
  } else {
    // for (int i = threadIdx.y; i < N; i += blockDim.y) {
    for (int i = 0; i < N; i += 1) {
      for (int j = threadIdx.x; j < M; j += blockDim.x) {
        int id = i * C * M + j * C + blockIdx.x;
        T x_in = x[id];
        x_sum += x_in;
        x2_sum += x_in * x_in;
      }
    }
  }
  __syncthreads();
  T out = BlockReduce(temp_storage).Reduce(x_sum, cub::Sum());
  __syncthreads();
  // if (threadIdx.x == 0 && threadIdx.y == 0) {
  if (threadIdx.x == 0) {
    mean_var[blockIdx.x] = out / (N * M);
  }
  out = BlockReduce(temp_storage).Reduce(x2_sum, cub::Sum());
  __syncthreads();
  // if (threadIdx.x == 0 && threadIdx.y == 0) {
  if (threadIdx.x == 0) {
    mean_var[blockIdx.x + C] = out / (N * M);
  }
  // if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
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
    LOG(ERROR) << "SyncBatchNormKernel Compute Start";
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    const float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");
    const std::string layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout layout = framework::StringToDataLayout(layout_str);

    const auto *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE(x_dims.size() >= 2 && x_dims.size() <= 5,
                   "The Input dim size should be between 2 and 5");
    int N, C, H, W, D;
    ExtractNCWHD(x_dims, layout, &N, &C, &H, &W, &D);
    int x_numel = x->numel();
    LOG(ERROR) << "NCHW " << N << " " << C << " " << H << " " << W;

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

    LOG(ERROR) << "========= 00 =========";
    paddle::memory::AllocationPtr alloc_ptr{nullptr};

    if (is_test) {
      const auto *est_mean = ctx.Input<Tensor>("Mean");
      const auto *est_var = ctx.Input<Tensor>("Variance");
      mean_data = est_mean->data<T>();
      var_data = est_var->data<T>();
    } else {
      LOG(ERROR) << "========= 11 =========";
      auto &allocator =
          platform::DeviceTemporaryAllocator::Instance().Get(dev_ctx);
      // x, x^2, 1, here 1 is used to calc device num
      // device num also can be got from platform::DeviceContextPool
      const int bytes = (C * 2 + 1) * sizeof(T);
      alloc_ptr = allocator.Allocate(bytes);

      T *stats = reinterpret_cast<T *>(alloc_ptr->ptr());
      const int block = 256;
      dim3 threads(block, 2);
      dim3 grid(C, 1);
      LOG(ERROR) << "========= 22 =========";

      Tensor ct;
      TensorCopy(*x, platform::CPUPlace(), dev_ctx, &ct);
      Tensor c_st, c_st2;
      T *c_st_d = c_st.mutable_data<T>({C}, platform::CPUPlace());
      T *c_st_d2 = c_st2.mutable_data<T>({C}, platform::CPUPlace());
      T *xx_d = ct.data<T>();
      for (int c = 0; c < C; ++c) {
        T sm = 0;
        T sm2 = 0;
        for (int i = 0; i < N; ++i) {
          for (int j = 0; j < H * W * D; ++j) {
            sm += xx_d[i * C * H * W * D + j];
            sm2 += xx_d[i * C * H * W * D + c * H * W * D + j] *
                   xx_d[i * c * H * W * D + c * H * W * D + j];
          }
        }
        c_st_d[c] = sm / (N * H * W * D);
        c_st_d2[c] = sm2 / (N * H * W * D);
      }
      T ass = 0;
      T ass2 = 0;
      for (int c = 0; c < C; ++c) {
        ass += std::abs(c_st_d[c]);
        ass2 += std::abs(c_st_d2[c]);
      }
      LOG(ERROR) << "mean sum = " << ass << ", m2 sum " << ass2;

      if (layout == framework::DataLayout::kNCHW) {
        KeLocalStats<T, 512,
                     framework::DataLayout::kNCHW><<<C, 512, 0, stream>>>(
            x_d, N, H * W * D, C, stats);
        // framework::DataLayout::kNCHW><<<grid, threads, 0, stream>>>(
      } else {
        KeLocalStats<
            T, 512, framework::DataLayout::kNHWC><<<grid, threads, 0, stream>>>(
            x_d, N, H * W * D, C, stats);
      }

      Tensor c_g_st;
      T *c_g_st_d = c_g_st.mutable_data<T>({2 * C + 1}, platform::CPUPlace());
      auto gplace = boost::get<platform::CUDAPlace>(ctx.GetPlace());
      memory::Copy(platform::CPUPlace(), c_g_st_d, gplace, stats, bytes, 0);

      ass = 0;
      ass2 = 0;
      for (int c = 0; c < C; ++c) {
        ass += std::abs(c_g_st_d[c]);
        ass2 += std::abs(c_g_st_d[C + c]);
      }
      LOG(ERROR) << "local mean sum = " << ass << ", local m2 sum " << ass2
                 << ", NUM " << c_g_st_d[2 * C];

      LOG(ERROR) << "========= 33 =========";
      int dtype = platform::ToNCCLDataType(x->type());
      // In-place operation
      PADDLE_ENFORCE(platform::dynload::ncclAllReduce(
          stats, stats, 2 * C + 1, static_cast<ncclDataType_t>(dtype), ncclSum,
          comm, stream));

      LOG(ERROR) << "========= 44 =========";
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
      KeSyncAndMovingStats<T><<<(C + 512 - 1) / 512, 512, 0, stream>>>(
          stats, stats + C, stats + 2 * C, C, momentum, epsilon, sv_mean_data,
          sv_inv_var_data, est_mean_data, est_var_data);

      mean_data = sv_mean_data;
      var_data = stats + C;
    }

    const int block = 512;
    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    LOG(ERROR) << "max_threads = " << max_threads;
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

    Tensor ctt;
    TensorCopy(*y, platform::CPUPlace(), dev_ctx, &ctt);
    T *ctt_d = ctt.data<T>();
    T smm = 0;
    for (int i = 0; i < ctt.numel(); ++i) {
      smm += std::abs(ctt_d[i]);
    }
    LOG(ERROR) << "BN Y shape " << y->dims() << ", osum " << smm;
  }
};

template <typename T, const int BlockDim, framework::DataLayout layout>
__global__ void KeBackwardLocalStats(const T *dy, const T *x, const T *means,
                                     int N, int M, int C, T *sum_dy_prod) {
  typedef cub::BlockReduce<double, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T sum1 = 0;
  T sum2 = 0;
  T mean = means[blockIdx.x];
  for (int i = threadIdx.y; i < N; i += blockDim.y) {
    for (int j = threadIdx.x; j < M; j += blockDim.x) {
      int id = layout == framework::DataLayout::kNCHW
                   ? (i * C + blockIdx.x) * M + j
                   : i * C * M + j * C + blockIdx.x;
      T g = dy[id];
      sum1 += g;
      sum2 += g * (x[id] - mean);
    }
  }
  __syncthreads();
  T out = BlockReduce(temp_storage).Reduce(sum1, cub::Sum());
  __syncthreads();
  if (threadIdx.x == 0) {
    sum_dy_prod[blockIdx.x] = out;
  }
  out = BlockReduce(temp_storage).Reduce(sum2, cub::Sum());
  __syncthreads();
  if (threadIdx.x == 0) {
    sum_dy_prod[blockIdx.x + C] = out;
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
  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ds_storage;
  __shared__ typename BlockReduce::TempStorage db_storage;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T ds_sum = static_cast<T>(0);
    T db_sum = static_cast<T>(0);

    T inv_var_i = inv_variance[i];
    T mean_i = mean[i];
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == framework::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      ds_sum += static_cast<T>(dy[index]) * (static_cast<T>(x[index]) - mean_i);
      db_sum += static_cast<T>(dy[index]);
    }
    ds_sum = BlockReduce(ds_storage).Reduce(ds_sum, cub::Sum());
    db_sum = BlockReduce(db_storage).Reduce(db_sum, cub::Sum());
    if (threadIdx.x == 0) {
      dscale[i] = ds_sum * inv_var_i;
      dbias[i] = db_sum;
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
    // T var = variance[c];
    T inv_var = inv_variance[c];
    T s_d = beta[c];
    T gvar =
        -1.0 * (g_sum_dy[c] / dev_num) * s_d * inv_var * (inv_var * inv_var);
    T gmean = -1.0 * (g_sum_dy_prod[c] / dev_num) * s_d * inv_var;

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
    LOG(ERROR) << "SyncBatchNormGradKernel Compute";
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
    LOG(ERROR) << "=======bn grad x_d=======";

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
    LOG(ERROR) << "======== 0 =======";

    const int block = 512;
    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    dim3 threads(block / 2, 2);
    dim3 grid(C, 1);
    int x_numel = x->numel();
    int grid2 = (std::min(x_numel, max_threads) + block - 1) / block;

    LOG(ERROR) << "======== 1 =======";
    if (layout == framework::DataLayout::kNCHW) {
      KeBackwardLocalStats<
          T, 256, framework::DataLayout::kNCHW><<<grid, threads, 0, stream>>>(
          dy_d, x_d, saved_mean, N, H * W * D, C, stats);
    } else {
      KeBackwardLocalStats<
          T, 256, framework::DataLayout::kNHWC><<<grid, threads, 0, stream>>>(
          dy_d, x_d, saved_mean, N, H * W * D, C, stats);
    }
    LOG(ERROR) << "======== 2 =======";
    int dtype = platform::ToNCCLDataType(x->type());
    // In-place operation
    PADDLE_ENFORCE(platform::dynload::ncclAllReduce(
        stats, stats, 2 * C + 1, static_cast<ncclDataType_t>(dtype), ncclSum,
        comm, stream));

    if (layout == framework::DataLayout::kNCHW) {
      LOG(ERROR) << "======== 3 =======";
      if (d_scale && d_bias) {
        KeBNBackwardScaleBias<T, block, framework::DataLayout::kNCHW><<<
            std::min(C, max_threads), block, 0, stream>>>(
            dy_d, x_d, saved_mean, saved_inv_var, epsilon, N, C, H * W * D,
            d_scale->data<T>(), d_bias->data<T>());
      }
      LOG(ERROR) << "======== 4 =======";
      if (d_x) {
        KeBNBackwardData<
            T, framework::DataLayout::kNCHW><<<grid2, block, 0, stream>>>(
            dy_d, x_d, scale->data<T>(), saved_mean, saved_inv_var, stats,
            stats + C, stats + 2 * C, epsilon, C, H * W, x->numel(),
            d_x->data<T>());
      }
    } else {
      if (d_scale && d_bias) {
        KeBNBackwardScaleBias<T, block, framework::DataLayout::kNHWC><<<
            std::min(C, max_threads), block, 0, stream>>>(
            dy_d, x_d, saved_mean, saved_inv_var, epsilon, N, C, H * W * D,
            d_scale->data<T>(), d_bias->data<T>());
      }
      if (d_x) {
        KeBNBackwardData<
            T, framework::DataLayout::kNHWC><<<grid2, block, 0, stream>>>(
            dy_d, x_d, scale->data<T>(), saved_mean, saved_inv_var, stats,
            stats + C, stats + 2 * C, epsilon, C, H * W, x->numel(),
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
