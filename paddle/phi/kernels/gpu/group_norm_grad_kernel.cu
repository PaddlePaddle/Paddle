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

#include "paddle/phi/kernels/group_norm_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/group_norm_utils.h"

namespace phi {

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
    phi::CudaAtomicAdd(&(d_scale[ccid]), d_scale_data);
#else
    CudaAtomicAddWithWarp(&(d_scale[ccid]), d_scale_data);
#endif
  }
  if (flags & kHasBias) {
#if CUDA_VERSION >= 11070
    phi::CudaAtomicAdd(&(d_bias[ccid]), d_bias_data);
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

template <typename T, typename Context>
void GroupNormGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const paddle::optional<DenseTensor>& scale,
                         const paddle::optional<DenseTensor>& bias,
                         const DenseTensor& y,
                         const DenseTensor& mean,
                         const DenseTensor& var,
                         const DenseTensor& d_y,
                         float epsilon,
                         int groups,
                         const std::string& data_layout_str,
                         DenseTensor* d_x,
                         DenseTensor* d_scale,
                         DenseTensor* d_bias) {
  const DataLayout data_layout = phi::StringToDataLayout(data_layout_str);
  const auto scale_ptr = scale.get_ptr();
  const auto bias_ptr = bias.get_ptr();

  const auto& x_dims = x.dims();
  const int C = (data_layout == DataLayout::kNCHW ? x_dims[1]
                                                  : x_dims[x_dims.size() - 1]);
  const int group_size = C / groups;
  const int W = (data_layout == DataLayout::kNCHW ? x_dims[x_dims.size() - 1]
                                                  : x_dims[x_dims.size() - 2]);

  dev_ctx.template Alloc<T>(d_x);
  phi::funcs::SetConstant<GPUContext, T> set_zero;

  DenseTensor ds, db;
  ds.Resize({x_dims[0], C});
  T* ds_data = dev_ctx.template Alloc<T>(&ds);
  db.Resize({x_dims[0], C});
  T* db_data = dev_ctx.template Alloc<T>(&db);

  auto* y_data = y.data<T>();
  auto* x_data = x.data<T>();
  T* d_x_data = nullptr;
  if (d_x) d_x_data = d_x->data<T>();
  auto* dy_data = d_y.data<T>();
  auto* var_data = var.data<T>();
  auto* mean_data = mean.data<T>();
  T* d_scale_data = nullptr;
  if (d_scale) {
    dev_ctx.template Alloc<T>(d_scale);
    d_scale_data = d_scale->data<T>();
  }
  T* d_bias_data = nullptr;
  if (d_bias) {
    dev_ctx.template Alloc<T>(d_bias);
    d_bias_data = d_bias->data<T>();
  }

  const T* scale_data = nullptr;
  if (scale_ptr) scale_data = scale_ptr->data<T>();
  const T* bias_data = nullptr;
  if (bias_ptr) bias_data = bias_ptr->data<T>();

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
    ScalarGetDsDbCUDAKernel<T><<<x_dims[0] * C, blocks, 0, dev_ctx.stream()>>>(
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
      DenseTensor p1, p2, p3;
      p1.Resize({x_dims[0] * C});
      T* p1_data = dev_ctx.template Alloc<T>(&p1);
      p2.Resize({x_dims[0], groups});
      T* p2_data = dev_ctx.template Alloc<T>(&p2);
      p3.Resize({x_dims[0], groups});
      T* p3_data = dev_ctx.template Alloc<T>(&p3);

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

    DenseTensor temp_var;
    temp_var.Resize(var.dims());
    dev_ctx.template Alloc<T>(&temp_var);
    set_zero(dev_ctx, &temp_var, static_cast<T>(0));
    T* temp_var_data = temp_var.data<T>();

    DenseTensor temp_mean;
    temp_mean.Resize(var.dims());
    dev_ctx.template Alloc<T>(&temp_mean);
    set_zero(dev_ctx, &temp_mean, static_cast<T>(0));
    T* temp_mean_data = temp_mean.data<T>();

    int flags =
        (scale_data != nullptr) * kHasScale + (bias_data != nullptr) * kHasBias;
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

}  // namespace phi

PD_REGISTER_KERNEL(
    group_norm_grad, GPU, ALL_LAYOUT, phi::GroupNormGradKernel, float, double) {
}
