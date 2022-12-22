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

template <typename T, typename AccT, int flags>
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
                                               float epsilon,
                                               AccT* d_mean,
                                               AccT* d_var,
                                               T* d_scale,
                                               T* d_bias) {
  int gid = blockIdx.y;
  int cid = blockIdx.x;
  int bid = blockIdx.z;
  int H = imsize / W;
  int number = min(group_size, static_cast<int>(C - gid * group_size));
  int ccid = gid * group_size + cid;
  if (ccid >= C) return;
  T x_scale = (flags & kHasScale) ? scale[ccid] : static_cast<T>(1);
  T x_bias = (flags & kHasBias) ? bias[ccid] : static_cast<T>(0);
  T x_scale_inv = static_cast<T>(0);
  if (x_scale != static_cast<T>(0)) x_scale_inv = static_cast<T>(1.0) / x_scale;
  AccT d_mean_data = static_cast<AccT>(0);
  AccT d_var_data = static_cast<AccT>(0);
  T d_scale_data = static_cast<T>(0);
  T d_bias_data = static_cast<T>(0);

  for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
    AccT val, dval;

    int hid = imid / W;
    int wid = imid % W;
    val = static_cast<AccT>(x[(bid * H + hid) * W * C + wid * C + ccid]) -
          static_cast<AccT>(x_bias);
    dval = static_cast<AccT>(d_y[(bid * H + hid) * W * C + wid * C + ccid]);

    d_var_data += val * dval;
    d_mean_data += dval * static_cast<AccT>(x_scale);

    val = val * static_cast<AccT>(x_scale_inv);
    d_bias_data += static_cast<T>(dval);
    d_scale_data += static_cast<T>(val * dval);
  }
  CudaAtomicAddWithWarp(&(d_mean[bid * groups + gid]),
                        static_cast<AccT>(d_mean_data));
  CudaAtomicAddWithWarp(&(d_var[bid * groups + gid]),
                        static_cast<AccT>(d_var_data));

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

template <typename T, typename AccT, int flags>
__global__ void GroupNormBackward(const T* x,
                                  const T* d_y,
                                  const T* scale,
                                  const T* bias,
                                  const AccT* var,
                                  const AccT* d_mean,
                                  const AccT* d_var,
                                  int N,
                                  int C,
                                  int W,
                                  int imsize,
                                  int groups,
                                  int group_size,
                                  float epsilon,
                                  T* d_x) {
  // using AccT = typename kps::details::MPTypeTrait<T>::Type;

  int gid = blockIdx.y;
  int cid = blockIdx.x;
  int bid = blockIdx.z;
  int H = imsize / W;
  int number = min(group_size, static_cast<int>(C - gid * group_size));
  int ccid = gid * group_size + cid;
  if (ccid >= C) return;
  AccT x_var = var[bid * groups + gid];
  AccT d_x_mean = static_cast<AccT>(d_mean[bid * groups + gid]);
  AccT d_x_var = static_cast<AccT>(d_var[bid * groups + gid]);

  AccT x_var_inv = static_cast<AccT>(1.0) / sqrt((x_var) + epsilon);
  AccT number_inv =
      static_cast<AccT>(1.0) / static_cast<AccT>((number * imsize));

  AccT x_scale = (flags & kHasScale) ? static_cast<AccT>(scale[ccid])
                                     : static_cast<AccT>(1);
  AccT x_bias =
      (flags & kHasBias) ? static_cast<AccT>(bias[ccid]) : static_cast<AccT>(0);
  AccT x_scale_inv = static_cast<T>(0);
  if (x_scale != static_cast<AccT>(0))
    x_scale_inv = static_cast<AccT>(1.0) / x_scale;

  for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
    int hid = imid / W;
    int wid = imid % W;
    AccT tmp = static_cast<AccT>(x[(bid * H + hid) * W * C + wid * C + ccid]);
    AccT v_y = (tmp - x_bias) * x_scale_inv;
    AccT dly = static_cast<AccT>(d_y[(bid * H + hid) * W * C + wid * C + ccid]);
    d_x[(bid * H + hid) * W * C + wid * C + ccid] =
        static_cast<T>(x_var_inv * ((dly) * (x_scale)-number_inv * d_x_var *
                                    (v_y)-number_inv * d_x_mean));
  }
}

template <typename T, typename AccT>
__global__ void ScalarGetDsDbCUDAKernel(
    int imsize, const T* x, const T* dy, AccT* ds, AccT* db) {
  const int nc = blockIdx.x;
  AccT ds_sum = 0;
  AccT db_sum = 0;
  for (int i = threadIdx.x; i < imsize; i += blockDim.x) {
    const int index = nc * imsize + i;
    ds_sum += static_cast<AccT>(dy[index]) * static_cast<AccT>(x[index]);
    db_sum += static_cast<AccT>(dy[index]);
  }
  ReduceMeanAndVar<AccT>(db, ds, db_sum, ds_sum, 1);
}

template <typename T, typename AccT>
__global__ void GetScaleBiasGradientCUDAKernel(int N,
                                               int C,
                                               int group,
                                               float epsilon,
                                               const AccT* mean,
                                               const AccT* var,
                                               const AccT* ds,
                                               const AccT* db,
                                               T* d_scale,
                                               T* d_bias) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < C) {
    const int G = group;
    const int D = C / G;
    AccT sum1 = static_cast<AccT>(0);
    AccT sum2 = static_cast<AccT>(0);
    for (int n = 0; n < N; ++n) {
      const int nc = n * C + c;
      const int ng = n * G + c / D;
      sum1 +=
          (d_scale == nullptr)
              ? AccT(0)
              : ((ds[nc] - db[nc] * (mean[ng])) * (rsqrt((var[ng]) + epsilon)));
      sum2 += (d_bias == nullptr) ? AccT(0) : db[nc];
    }
    if (d_scale != nullptr) {
      d_scale[c] = static_cast<T>(sum1);
    }
    if (d_bias != nullptr) {
      d_bias[c] = static_cast<T>(sum2);
    }
  }
}

template <typename T, typename AccT, int BlockDim>
__global__ void GetBackwardParamsCUDAKernel(int imsize,
                                            int groups,
                                            int group_size,
                                            float epsilon,
                                            const AccT* mean,
                                            const AccT* var,
                                            const T* scale,
                                            const AccT* ds,
                                            const AccT* db,
                                            AccT* p1,
                                            AccT* p2,
                                            AccT* p3) {
  const int n = blockIdx.x;
  const int g = blockIdx.y;
  const int ng = n * groups + g;
  AccT sum1 = 0;
  AccT sum2 = 0;
  AccT var_inv = rsqrt(static_cast<AccT>(var[ng]) + epsilon);
  for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
    const int64_t index = ng * group_size + i;
    const int64_t c = g * group_size + i;
    const AccT scale_v =
        scale == nullptr ? static_cast<AccT>(1) : static_cast<AccT>(scale[c]);
    sum1 += static_cast<AccT>(ds[index]) * scale_v;
    sum2 += static_cast<AccT>(db[index]) * scale_v;
    const AccT scale_c =
        scale == nullptr ? static_cast<AccT>(0) : static_cast<T>(scale[c]);
    p1[index] = static_cast<AccT>(scale_c) * var_inv;
  }

  typedef cub::BlockReduce<AccT, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ds_storage;
  __shared__ typename BlockReduce::TempStorage db_storage;
  sum1 = BlockReduce(ds_storage).Reduce(sum1, cub::Sum());
  sum2 = BlockReduce(db_storage).Reduce(sum2, cub::Sum());

  if (threadIdx.x == 0) {
    const AccT s =
        static_cast<AccT>(1) / static_cast<AccT>(group_size * imsize);
    const AccT x = (sum2 * static_cast<AccT>(mean[ng]) - sum1) * (var_inv) *
                   (var_inv) * (var_inv)*s;
    p2[ng] = x;
    p3[ng] = -x * (mean[ng]) - (sum2 * var_inv) * s;
  }
}

template <typename T, typename AccT>
__global__ void GetXGradientCUDAKernel(int imsize,
                                       int C,
                                       int group_size,
                                       int groups,
                                       AccT* p1,
                                       AccT* p2,
                                       AccT* p3,
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
    dx[index] = static_cast<T>(p1[ccid] * static_cast<AccT>(dy[index]) +
                               p2[ng] * static_cast<AccT>(x[index]) + p3[ng]);
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
  using AccT = typename kps::details::MPTypeTrait<T>::Type;
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
  phi::funcs::SetConstant<GPUContext, AccT> set_zero_AccT;
  DenseTensor ds, db;
  ds.Resize({x_dims[0], C});
  AccT* ds_data = dev_ctx.template Alloc<AccT>(&ds);
  db.Resize({x_dims[0], C});
  AccT* db_data = dev_ctx.template Alloc<AccT>(&db);

  auto* y_data = y.data<T>();
  auto* x_data = x.data<T>();
  T* d_x_data = nullptr;
  if (d_x) d_x_data = d_x->data<T>();
  auto* dy_data = d_y.data<T>();
  auto* var_data = var.data<AccT>();
  auto* mean_data = mean.data<AccT>();
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
    ScalarGetDsDbCUDAKernel<T, AccT>
        <<<x_dims[0] * C, blocks, 0, dev_ctx.stream()>>>(
            imsize, x_data, dy_data, ds_data, db_data);

    if (d_scale || d_bias) {
      const int block = 256;
      GetScaleBiasGradientCUDAKernel<T, AccT>
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
      AccT* p1_data = dev_ctx.template Alloc<AccT>(&p1);
      p2.Resize({x_dims[0], groups});
      AccT* p2_data = dev_ctx.template Alloc<AccT>(&p2);
      p3.Resize({x_dims[0], groups});
      AccT* p3_data = dev_ctx.template Alloc<AccT>(&p3);

      GetBackwardParamsCUDAKernel<T, AccT, block_dims>
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
    set_zero_AccT(dev_ctx, &temp_var, static_cast<AccT>(0));
    auto* temp_var_data = temp_var.data<AccT>();

    DenseTensor temp_mean;
    temp_mean.Resize(var.dims());
    dev_ctx.template Alloc<AccT>(&temp_mean);
    set_zero_AccT(dev_ctx, &temp_mean, static_cast<AccT>(0));
    auto* temp_mean_data = temp_mean.data<AccT>();

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

PD_REGISTER_KERNEL(group_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::GroupNormGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
