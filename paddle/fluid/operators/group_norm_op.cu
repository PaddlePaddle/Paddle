/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/fluid/operators/group_norm_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

using DataLayout = framework::DataLayout;
enum GroupNormKernelFlags { kHasScale = 1, kHasBias = 2 };
#define ALIGN_BYTES 16

#define CHECK_CASE(i, flags, kernel_name, ...)                              \
  if (i == flags) {                                                         \
    kernel_name<T, i><<<grid, threads, 0, dev_ctx.stream()>>>(__VA_ARGS__); \
  }

// 0 for no scale, no bias
// 1 for has scale, no bias
// 2 for no scale, has bias
// 3 for has scale, has bias
#define UNROLL_ALL_CASES(flags, kernel_name, ...) \
  CHECK_CASE(0, flags, kernel_name, __VA_ARGS__)  \
  CHECK_CASE(1, flags, kernel_name, __VA_ARGS__)  \
  CHECK_CASE(2, flags, kernel_name, __VA_ARGS__)  \
  CHECK_CASE(3, flags, kernel_name, __VA_ARGS__)

template <typename T>
__device__ __inline__ void CudaAtomicAddWithWarp(T* sum, T value) {
  typedef cub::WarpReduce<T> WarpReduce;
  typename WarpReduce::TempStorage temp_storage;
  value = WarpReduce(temp_storage).Sum(value);
  if (cub::LaneId() == 0) platform::CudaAtomicAdd(sum, value);
}

template <typename T>
__global__ void GroupNormForwardGetMeanAndVar(const T* x,
                                              int N,
                                              int C,
                                              int W,
                                              int imsize,
                                              int groups,
                                              int group_size,
                                              T* mean,
                                              T* var) {
  int gid = blockIdx.y;
  int cid = blockIdx.x;
  int bid = blockIdx.z;
  int H = imsize / W;
  int number = min(group_size, static_cast<int>(C - gid * group_size));
  int ccid = gid * group_size + cid;
  if (ccid >= C) return;
  T x_mean = 0, x_var = 0;
  for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
    T val;
    int hid = imid / W;
    int wid = imid % W;
    val = x[(bid * H + hid) * W * C + wid * C + ccid];

    x_mean += val;
    x_var += val * val;
  }
  x_mean /= number * imsize;
  x_var /= number * imsize;
  CudaAtomicAddWithWarp(&mean[bid * groups + gid], x_mean);
  CudaAtomicAddWithWarp(&var[bid * groups + gid], x_var);
}

template <typename T, typename AccT, int VecSize, int Num>
__device__ __forceinline__ void ThreadReduce(phi::Array<const T*, Num> arrs,
                                             int size,
                                             const int offset,
                                             AccT* out_mean,
                                             AccT* out_var) {
  const T* x = arrs[0];
  const T* y;
  if (Num == 2) {
    y = arrs[1];
  }
  using VecT = kps::details::VectorType<T, VecSize>;
  int tid = threadIdx.x;
  if (offset > 0) {
    x -= offset;
    if (Num == 2) {
      y -= offset;
    }
    size += offset;
    if (tid >= offset) {
      if (Num == 1) {
        *out_mean += x[tid];
        *out_var += x[tid] * x[tid];
      } else if (Num == 2) {
        *out_mean += y[tid];
        *out_var += y[tid] * x[tid];
      }
    }
    size -= blockDim.x;
    x += blockDim.x;
    if (Num == 2) {
      y += blockDim.x;
    }
  }
  int remain = size % (VecSize * blockDim.x);

  T ins_x[VecSize];
  T ins_y[VecSize];
  VecT* ins_vec_x = reinterpret_cast<VecT*>(&ins_x);
  VecT* ins_vec_y = reinterpret_cast<VecT*>(&ins_y);

  // vector part
  for (; VecSize * tid < (size - remain); tid += blockDim.x) {
    *ins_vec_x = reinterpret_cast<const VecT*>(x)[tid];
    if (Num == 2) {
      *ins_vec_y = reinterpret_cast<const VecT*>(y)[tid];
    }

#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      if (Num == 1) {
        *out_mean += ins_x[i];
        *out_var += ins_x[i] * ins_x[i];
      } else if (Num == 2) {
        *out_mean += ins_y[i];
        *out_var += ins_y[i] * ins_x[i];
      }
    }
  }

  // scalar part
  tid = size - remain + threadIdx.x;
  for (; tid < size; tid += blockDim.x) {
    if (Num == 1) {
      *out_mean += x[tid];
      *out_var += x[tid] * x[tid];
    } else if (Num == 2) {
      *out_mean += y[tid];
      *out_var += y[tid] * x[tid];
    }
  }
}

template <typename T>
__device__ __forceinline__ void ReduceMeanAndVar(
    T* mean, T* var, T x_mean, T x_var, int size) {
  const int nc = blockIdx.x;
  x_mean = kps::details::BlockXReduce<T, kps::AddFunctor<T>>(
      x_mean, kps::AddFunctor<T>());
  x_var = kps::details::BlockXReduce<T, kps::AddFunctor<T>>(
      x_var, kps::AddFunctor<T>());
  __syncthreads();
  if (threadIdx.x == 0) {
    mean[nc] = static_cast<T>(x_mean / size);
    var[nc] = static_cast<T>(x_var / size);
  }
}

template <typename T>
__global__ void ScalarGetMeanAndVarNCHW(const T* x, T* mean, T* var, int size) {
  int i = blockIdx.x;
  T x_mean = 0, x_var = 0;
  for (int j = threadIdx.x; j < size; j += blockDim.x) {
    T val;
    val = x[i * size + j];
    x_mean += val;
    x_var += val * val;
  }
  ReduceMeanAndVar<T>(mean, var, x_mean, x_var, size);
}

template <typename T, typename AccT, int VecSize>
__global__ void VectorizedGetMeanAndVarNCHW(const T* x,
                                            T* mean,
                                            T* var,
                                            int size) {
  int i = blockIdx.x;
  AccT x_mean = static_cast<AccT>(0);
  AccT x_var = static_cast<AccT>(0);
  x += i * size;
  const int input_offset = ((uint64_t)x) % ALIGN_BYTES / sizeof(T);
  phi::Array<const T*, 1> ins;
  ins[0] = x;
  ThreadReduce<T, AccT, VecSize, 1>(ins, size, input_offset, &x_mean, &x_var);
  ReduceMeanAndVar<AccT>(mean, var, x_mean, x_var, size);
}

template <typename T, int flags>
__global__ void GroupNormForward(const T* x,
                                 const T* mean,
                                 const T* var,
                                 const T* scale,
                                 const T* bias,
                                 int N,
                                 int C,
                                 int W,
                                 int imsize,
                                 int groups,
                                 int group_size,
                                 T epsilon,
                                 T* y,
                                 T* real_var,
                                 const DataLayout data_layout) {
  int gid = blockIdx.y;
  int cid = blockIdx.x;
  int bid = blockIdx.z;
  int H = imsize / W;
  int ccid = gid * group_size + cid;
  if (ccid >= C) return;
  auto ng = bid * groups + gid;
  T x_mean = mean[ng];
  T x_var = var[ng];
  x_var = x_var - x_mean * x_mean;
  T var_inv = rsqrt(x_var + epsilon);
  if (cid == 0 && threadIdx.x == 0) {
    real_var[ng] = x_var;
  }
  for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
    T val;
    int hid, wid;
    int index = (bid * C + ccid) * imsize + imid;
    if (data_layout == DataLayout::kNCHW) {
      val = x[index];
    } else {
      hid = imid / W;
      wid = imid % W;
      val = x[(bid * H + hid) * W * C + wid * C + ccid];
    }
    val = (val - x_mean) * var_inv;
    if (flags & kHasScale) {
      val *= scale[ccid];
    }
    if (flags & kHasBias) {
      val += bias[ccid];
    }
    if (data_layout == DataLayout::kNCHW) {
      y[index] = val;
    } else {
      y[(bid * H + hid) * W * C + wid * C + ccid] = val;
    }
  }
}

template <typename T>
class GroupNormKernel<phi::GPUContext, T> : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    const float epsilon = ctx.Attr<float>("epsilon");
    auto* scale = ctx.Input<Tensor>("Scale");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* x = ctx.Input<Tensor>("X");

    auto* y = ctx.Output<Tensor>("Y");
    auto* mean = ctx.Output<Tensor>("Mean");
    auto* var = ctx.Output<Tensor>("Variance");
    const auto groups = ctx.Attr<int>("groups");

    const auto x_dims = x->dims();
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);
    const int group_size = C / groups;

    const int W =
        (data_layout == DataLayout::kNCHW ? x_dims[x_dims.size() - 1]
                                          : x_dims[x_dims.size() - 2]);

    y->mutable_data<T>(ctx.GetPlace());
    mean->mutable_data<T>(ctx.GetPlace());
    var->mutable_data<T>(ctx.GetPlace());
    phi::funcs::SetConstant<phi::GPUContext, T> set_zero;
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
    Tensor temp_var;
    temp_var.mutable_data<T>(var->dims(), ctx.GetPlace());
    auto* x_data = x->data<T>();
    auto* y_data = y->data<T>();
    auto* mean_data = mean->data<T>();
    auto* var_data = var->data<T>();
    auto* temp_var_data = temp_var.data<T>();

    const T* scale_data = nullptr;
    if (scale) scale_data = scale->data<T>();
    const T* bias_data = nullptr;
    if (bias) bias_data = bias->data<T>();

    int imsize = 1;
    if (data_layout == DataLayout::kNCHW) {
      for (int i = 2; i < x_dims.size(); ++i) {
        imsize *= x_dims[i];
      }
    } else {
      for (int i = 1; i < x_dims.size() - 1; ++i) {
        imsize *= x_dims[i];
      }
    }

#ifdef __HIPCC__
    int block_size = std::max(std::min(256, imsize), 64);
#else
    int block_size = std::min(1024, imsize);
#endif

    dim3 grid(group_size, groups, x_dims[0]);
    dim3 threads(block_size, 1, 1);
    if (data_layout == DataLayout::kNCHW) {
      using AccT = typename details::MPTypeTrait<T>::Type;
      constexpr int vec_size = sizeof(float4) / sizeof(T);
      int size = group_size * imsize;
      const int max_num_threads = 1024;
      int max_block_size = std::min(size / vec_size, max_num_threads);
      int block_size_nchw = 1;
      while (block_size_nchw < max_block_size) {
        block_size_nchw *= 2;
      }
      block_size_nchw = std::max(block_size_nchw, kps::details::kWarpSize);
      dim3 grids(x_dims[0] * groups);
      dim3 blocks(block_size_nchw);
      if (size < vec_size * block_size_nchw) {
        ScalarGetMeanAndVarNCHW<T><<<grids, blocks, 0, dev_ctx.stream()>>>(
            x_data, mean_data, temp_var_data, size);
      } else {
        VectorizedGetMeanAndVarNCHW<T, AccT, vec_size>
            <<<grids, blocks, 0, dev_ctx.stream()>>>(
                x_data, mean_data, temp_var_data, size);
      }
    } else {
      set_zero(dev_ctx, mean, static_cast<T>(0));
      set_zero(dev_ctx, &temp_var, static_cast<T>(0));
      GroupNormForwardGetMeanAndVar<T>
          <<<grid, threads, 0, dev_ctx.stream()>>>(x_data,
                                                   x_dims[0],
                                                   C,
                                                   W,
                                                   imsize,
                                                   groups,
                                                   group_size,
                                                   mean_data,
                                                   temp_var_data);
    }
    int flags =
        (scale_data != nullptr) * kHasScale + (bias_data != nullptr) * kHasBias;
    UNROLL_ALL_CASES(flags,
                     GroupNormForward,
                     x_data,
                     mean_data,
                     temp_var_data,
                     scale_data,
                     bias_data,
                     x_dims[0],
                     C,
                     W,
                     imsize,
                     groups,
                     group_size,
                     epsilon,
                     y_data,
                     var_data,
                     data_layout);
  }
};

template <typename T, int flags>
__global__ void GroupNormBackwardGetMeanAndVar(const T* x,
                                               const T* scale,
                                               const T* bias,
                                               const T* d_y,
                                               int N,
                                               int C,
                                               int W,
                                               int imsize,
                                               int groups,
                                               int group_size,
                                               T epsilon,
                                               T* d_mean,
                                               T* d_var,
                                               T* d_scale,
                                               T* d_bias) {
  int gid = blockIdx.y;
  int cid = blockIdx.x;
  int bid = blockIdx.z;
  int H = imsize / W;
  int number = min(group_size, static_cast<int>(C - gid * group_size));
  int ccid = gid * group_size + cid;
  if (ccid >= C) return;
  T x_scale = (flags & kHasScale) ? scale[ccid] : 1;
  T x_bias = (flags & kHasBias) ? bias[ccid] : 0;
  T x_scale_inv = 0;
  if (x_scale != 0) x_scale_inv = 1.0 / x_scale;
  T d_mean_data = 0, d_var_data = 0, d_scale_data = 0, d_bias_data = 0;

  for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
    T val, dval;

    int hid = imid / W;
    int wid = imid % W;
    val = x[(bid * H + hid) * W * C + wid * C + ccid] - x_bias;
    dval = d_y[(bid * H + hid) * W * C + wid * C + ccid];

    d_var_data += val * dval;
    d_mean_data += dval * x_scale;

    val = val * x_scale_inv;
    d_bias_data += dval;
    d_scale_data += val * dval;
  }
  CudaAtomicAddWithWarp(&(d_mean[bid * groups + gid]), d_mean_data);
  CudaAtomicAddWithWarp(&(d_var[bid * groups + gid]), d_var_data);

  if (flags & kHasScale) {
#if CUDA_VERSION >= 11070
    platform::CudaAtomicAdd(&(d_scale[ccid]), d_scale_data);
#else
    CudaAtomicAddWithWarp(&(d_scale[ccid]), d_scale_data);
#endif
  }
  if (flags & kHasBias) {
#if CUDA_VERSION >= 11070
    platform::CudaAtomicAdd(&(d_bias[ccid]), d_bias_data);
#else
    CudaAtomicAddWithWarp(&(d_bias[ccid]), d_bias_data);
#endif
  }
}

template <typename T, int flags>
__global__ void GroupNormBackward(const T* x,
                                  const T* d_y,
                                  const T* scale,
                                  const T* bias,
                                  const T* var,
                                  const T* d_mean,
                                  const T* d_var,
                                  int N,
                                  int C,
                                  int W,
                                  int imsize,
                                  int groups,
                                  int group_size,
                                  T epsilon,
                                  T* d_x) {
  int gid = blockIdx.y;
  int cid = blockIdx.x;
  int bid = blockIdx.z;
  int H = imsize / W;
  int number = min(group_size, static_cast<int>(C - gid * group_size));
  int ccid = gid * group_size + cid;
  if (ccid >= C) return;
  T x_var = var[bid * groups + gid];
  T d_x_mean = d_mean[bid * groups + gid];
  T d_x_var = d_var[bid * groups + gid];

  T x_var_inv = 1.0 / sqrt(x_var + epsilon);
  T number_inv = 1.0 / (number * imsize);

  T x_scale = (flags & kHasScale) ? scale[ccid] : 1;
  T x_bias = (flags & kHasBias) ? bias[ccid] : 0;
  T x_scale_inv = 0;
  if (x_scale != 0) x_scale_inv = 1.0 / x_scale;

  for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
    int hid = imid / W;
    int wid = imid % W;
    T tmp = x[(bid * H + hid) * W * C + wid * C + ccid];
    T v_y = (tmp - x_bias) * x_scale_inv;
    T dly = d_y[(bid * H + hid) * W * C + wid * C + ccid];
    d_x[(bid * H + hid) * W * C + wid * C + ccid] =
        x_var_inv *
        (dly * x_scale - number_inv * d_x_var * v_y - number_inv * d_x_mean);
  }
}

template <typename T>
__global__ void ScalarGetDsDbCUDAKernel(
    int imsize, const T* x, const T* dy, T* ds, T* db) {
  const int nc = blockIdx.x;
  T ds_sum = 0;
  T db_sum = 0;
  for (int i = threadIdx.x; i < imsize; i += blockDim.x) {
    const int index = nc * imsize + i;
    ds_sum += dy[index] * x[index];
    db_sum += dy[index];
  }
  ReduceMeanAndVar<T>(db, ds, db_sum, ds_sum, 1);
}

template <typename T>
__global__ void GetScaleBiasGradientCUDAKernel(int N,
                                               int C,
                                               int group,
                                               T epsilon,
                                               const T* mean,
                                               const T* var,
                                               const T* ds,
                                               const T* db,
                                               T* d_scale,
                                               T* d_bias) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < C) {
    const int G = group;
    const int D = C / G;
    T sum1 = 0;
    T sum2 = 0;
    for (int n = 0; n < N; ++n) {
      const int nc = n * C + c;
      const int ng = n * G + c / D;
      sum1 += (d_scale == nullptr)
                  ? T(0)
                  : ((ds[nc] - db[nc] * static_cast<T>(mean[ng])) *
                     static_cast<T>(rsqrt(var[ng] + epsilon)));
      sum2 += (d_bias == nullptr) ? T(0) : db[nc];
    }
    if (d_scale != nullptr) {
      d_scale[c] = sum1;
    }
    if (d_bias != nullptr) {
      d_bias[c] = sum2;
    }
  }
}

template <typename T, int BlockDim>
__global__ void GetBackwardParamsCUDAKernel(int imsize,
                                            int groups,
                                            int group_size,
                                            T epsilon,
                                            const T* mean,
                                            const T* var,
                                            const T* scale,
                                            const T* ds,
                                            const T* db,
                                            T* p1,
                                            T* p2,
                                            T* p3) {
  const int n = blockIdx.x;
  const int g = blockIdx.y;
  const int ng = n * groups + g;
  T sum1 = 0;
  T sum2 = 0;
  T var_inv = rsqrt(var[ng] + epsilon);
  for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
    const int64_t index = ng * group_size + i;
    const int64_t c = g * group_size + i;
    const T scale_v = scale == nullptr ? T(1) : static_cast<T>(scale[c]);
    sum1 += ds[index] * scale_v;
    sum2 += db[index] * scale_v;
    const T scale_c = scale == nullptr ? T(0) : static_cast<T>(scale[c]);
    p1[index] = scale_c * var_inv;
  }

  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ds_storage;
  __shared__ typename BlockReduce::TempStorage db_storage;
  sum1 = BlockReduce(ds_storage).Reduce(sum1, cub::Sum());
  sum2 = BlockReduce(db_storage).Reduce(sum2, cub::Sum());

  if (threadIdx.x == 0) {
    const T s = T(1) / static_cast<T>(group_size * imsize);
    const T x = (sum2 * static_cast<T>(mean[ng]) - sum1) *
                static_cast<T>(var_inv) * static_cast<T>(var_inv) *
                static_cast<T>(var_inv) * s;
    p2[ng] = x;
    p3[ng] = -x * static_cast<T>(mean[ng]) - sum2 * static_cast<T>(var_inv) * s;
  }
}

template <typename T>
__global__ void GetXGradientCUDAKernel(int imsize,
                                       int C,
                                       int group_size,
                                       int groups,
                                       T* p1,
                                       T* p2,
                                       T* p3,
                                       const T* x,
                                       const T* dy,
                                       T* dx) {
  int cid = blockIdx.x;
  int gid = blockIdx.y;
  int bid = blockIdx.z;
  int ccid = bid * C + gid * group_size + cid;
  int ng = bid * groups + gid;
  int nc = gid * group_size + cid;
  for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
    int index = (bid * C + nc) * imsize + imid;
    dx[index] = p1[ccid] * dy[index] + p2[ng] * x[index] + p3[ng];
  }
}

template <typename T>
class GroupNormGradKernel<phi::GPUContext, T> : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    const float epsilon = ctx.Attr<float>("epsilon");
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* mean = ctx.Input<Tensor>("Mean");
    auto* var = ctx.Input<Tensor>("Variance");
    auto* scale = ctx.Input<Tensor>("Scale");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto groups = ctx.Attr<int>("groups");

    // init output
    auto* d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto* d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    const auto& x_dims = x->dims();
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);
    const int group_size = C / groups;
    const int W =
        (data_layout == DataLayout::kNCHW ? x_dims[x_dims.size() - 1]
                                          : x_dims[x_dims.size() - 2]);

    d_x->mutable_data<T>(ctx.GetPlace());
    phi::funcs::SetConstant<phi::GPUContext, T> set_zero;
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();

    Tensor ds, db;
    ds.mutable_data<T>({x_dims[0], C}, ctx.GetPlace());
    db.mutable_data<T>({x_dims[0], C}, ctx.GetPlace());
    T* ds_data = ds.data<T>();
    T* db_data = db.data<T>();

    auto* y_data = y->data<T>();
    auto* x_data = x->data<T>();
    T* d_x_data = nullptr;
    if (d_x) d_x_data = d_x->data<T>();
    auto* dy_data = d_y->data<T>();
    auto* var_data = var->data<T>();
    auto* mean_data = mean->data<T>();
    T* d_scale_data = nullptr;
    if (d_scale) {
      d_scale->mutable_data<T>(ctx.GetPlace());
      d_scale_data = d_scale->data<T>();
    }
    T* d_bias_data = nullptr;
    if (d_bias) {
      d_bias->mutable_data<T>(ctx.GetPlace());
      d_bias_data = d_bias->data<T>();
    }

    const T* scale_data = nullptr;
    if (scale) scale_data = scale->data<T>();
    const T* bias_data = nullptr;
    if (bias) bias_data = bias->data<T>();

    int imsize = 1;
    if (data_layout == DataLayout::kNCHW) {
      for (int i = 2; i < x_dims.size(); ++i) {
        imsize *= x_dims[i];
      }
    } else {
      for (int i = 1; i < x_dims.size() - 1; ++i) {
        imsize *= x_dims[i];
      }
    }

#ifdef __HIPCC__
    int block_size = std::max(std::min(256, imsize), 64);
    const int block_dims = 256;
#else
    int block_size = std::min(1024, imsize);
    const int block_dims = 1024;
#endif
    dim3 grid(group_size, groups, x_dims[0]);
    dim3 threads(block_size, 1, 1);
    int flags =
        (scale_data != nullptr) * kHasScale + (bias_data != nullptr) * kHasBias;
    if (data_layout == DataLayout::kNCHW) {
      const int max_num_threads = 1024;
      int max_block_size = std::min(imsize, max_num_threads);
      int block_size_nchw = 1;
      while (block_size_nchw < max_block_size) {
        block_size_nchw *= 2;
      }
      block_size_nchw = std::max(block_size_nchw, kps::details::kWarpSize);
      dim3 blocks(block_size_nchw);
      ScalarGetDsDbCUDAKernel<T>
          <<<x_dims[0] * C, blocks, 0, dev_ctx.stream()>>>(
              imsize, x_data, dy_data, ds_data, db_data);

      if (d_scale || d_bias) {
        const int block = 256;
        GetScaleBiasGradientCUDAKernel<T>
            <<<(C + block - 1) / block, block, 0, dev_ctx.stream()>>>(
                x_dims[0],
                C,
                groups,
                epsilon,
                mean_data,
                var_data,
                ds_data,
                db_data,
                d_scale_data,
                d_bias_data);
      }

      if (d_x_data != nullptr) {
        // p1 * dy + p2 * x + p3,
        // p1, p2, p3 represent the reverse calculation of temporary variables
        // p1 = scale * var_inv
        // p2 = (db * scale * mean - ds * scale) * pow(var_inv, 3) * (1/n)
        // p3 = -p2 * mean[ng] - db * scale * var_inv * (1/n);
        Tensor p1, p2, p3;
        p1.mutable_data<T>({x_dims[0] * C}, ctx.GetPlace());
        p2.mutable_data<T>({x_dims[0], groups}, ctx.GetPlace());
        p3.mutable_data<T>({x_dims[0], groups}, ctx.GetPlace());
        T* p1_data = p1.data<T>();
        T* p2_data = p2.data<T>();
        T* p3_data = p3.data<T>();

        GetBackwardParamsCUDAKernel<T, block_dims>
            <<<dim3(x_dims[0], groups), block_dims, 0, dev_ctx.stream()>>>(
                imsize,
                groups,
                group_size,
                epsilon,
                mean_data,
                var_data,
                scale_data,
                ds_data,
                db_data,
                p1_data,
                p2_data,
                p3_data);
        GetXGradientCUDAKernel<T>
            <<<grid, threads, 0, dev_ctx.stream()>>>(imsize,
                                                     C,
                                                     group_size,
                                                     groups,
                                                     p1_data,
                                                     p2_data,
                                                     p3_data,
                                                     x_data,
                                                     dy_data,
                                                     d_x_data);
      }
    } else {
      if (d_scale) {
        set_zero(dev_ctx, d_scale, static_cast<T>(0));
      }
      if (d_bias) {
        set_zero(dev_ctx, d_bias, static_cast<T>(0));
      }

      Tensor temp_var;
      temp_var.mutable_data<T>(var->dims(), ctx.GetPlace());
      set_zero(dev_ctx, &temp_var, static_cast<T>(0));
      T* temp_var_data = temp_var.data<T>();

      Tensor temp_mean;
      temp_mean.mutable_data<T>(var->dims(), ctx.GetPlace());
      set_zero(dev_ctx, &temp_mean, static_cast<T>(0));
      T* temp_mean_data = temp_mean.data<T>();

      int flags = (scale_data != nullptr) * kHasScale +
                  (bias_data != nullptr) * kHasBias;
      UNROLL_ALL_CASES(flags,
                       GroupNormBackwardGetMeanAndVar,
                       y_data,
                       scale_data,
                       bias_data,
                       dy_data,
                       x_dims[0],
                       C,
                       W,
                       imsize,
                       groups,
                       group_size,
                       epsilon,
                       temp_mean_data,
                       temp_var_data,
                       d_scale_data,
                       d_bias_data);
      if (d_x_data != nullptr) {
        UNROLL_ALL_CASES(flags,
                         GroupNormBackward,
                         y_data,
                         dy_data,
                         scale_data,
                         bias_data,
                         var_data,
                         temp_mean_data,
                         temp_var_data,
                         x_dims[0],
                         C,
                         W,
                         imsize,
                         groups,
                         group_size,
                         epsilon,
                         d_x_data);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(group_norm,
                        ops::GroupNormKernel<phi::GPUContext, float>,
                        ops::GroupNormKernel<phi::GPUContext, double>);
REGISTER_OP_CUDA_KERNEL(group_norm_grad,
                        ops::GroupNormGradKernel<phi::GPUContext, float>,
                        ops::GroupNormGradKernel<phi::GPUContext, double>);
