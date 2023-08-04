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
#include <stdio.h>

#include "paddle/phi/kernels/group_norm_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/group_norm_utils.h"

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/device_context.h"

namespace phi {

template <typename T, typename AccT>
__global__ void GroupNormForwardGetMeanAndVar(const T* x,
                                              int N,
                                              int C,
                                              int W,
                                              int imsize,
                                              int groups,
                                              int group_size,
                                              AccT* mean,
                                              AccT* var) {
  int hid = blockIdx.x;
  int bid = blockIdx.y;
  int gwsize = groups * W;
  AccT x_mean = static_cast<AccT>(0);
  AccT x_var = static_cast<AccT>(0);

#ifdef __HIPCC__
  __shared__ AccT smem_m[256];
  __shared__ AccT smem_v[256];
#else
  __shared__ AccT smem_m[1024];
  __shared__ AccT smem_v[1024];
#endif

  for (int gwid = threadIdx.x; gwid < gwsize; gwid += blockDim.x) {
    int index_gw = bid * C * imsize + hid * C * W + gwid * group_size;
    x_mean = static_cast<AccT>(0);
    x_var = static_cast<AccT>(0);
    AccT val;
#pragma unroll
    for (int gsid = 0; gsid < group_size; ++gsid) {
      int index_gs = index_gw + gsid;
      val = static_cast<AccT>(x[index_gs]);

      x_mean += val;
      x_var += val * val;
    }
    x_mean /= group_size * imsize;
    x_var /= group_size * imsize;

    int sid = gwid % blockDim.x;
    smem_m[sid] = x_mean;
    smem_v[sid] = x_var;
    __syncthreads();

    if (sid < groups) {
      for (sid = sid + groups; sid < blockDim.x; sid += groups) {
        x_mean += smem_m[sid];
        x_var += smem_v[sid];
      }

      phi::CudaAtomicAdd(&mean[bid * groups + gwid % groups], x_mean);
      phi::CudaAtomicAdd(&var[bid * groups + gwid % groups], x_var);
    }
  }
}

template <typename T, typename AccT, int flags>
__global__ void GroupNormForward(const T* x,
                                 const AccT* mean,
                                 const AccT* var,
                                 const T* scale,
                                 const T* bias,
                                 int N,
                                 int C,
                                 int W,
                                 int imsize,
                                 int groups,
                                 int group_size,
                                 AccT epsilon,
                                 T* y,
                                 AccT* real_var,
                                 const DataLayout data_layout) {
  int gid = blockIdx.y;
  int cid = blockIdx.x;
  int bid = blockIdx.z;
  int H = imsize / W;
  int ccid = gid * group_size + cid;
  if (ccid >= C) return;
  auto ng = bid * groups + gid;
  AccT x_mean = mean[ng];
  AccT x_var = var[ng];
  x_var = x_var - x_mean * x_mean;

  AccT var_inv = rsqrt(x_var + epsilon);
  if (cid == 0 && threadIdx.x == 0) {
    real_var[ng] = x_var;
  }
  for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
    AccT val;
    int hid, wid;
    int index = (bid * C + ccid) * imsize + imid;
    if (data_layout == DataLayout::kNCHW) {
      val = static_cast<AccT>(x[index]);
    } else {
      hid = imid / W;
      wid = imid % W;
      val = static_cast<AccT>(x[(bid * H + hid) * W * C + wid * C + ccid]);
    }
    val = (val - x_mean) * var_inv;
    if (flags & kHasScale) {
      val *= static_cast<AccT>(scale[ccid]);
    }
    if (flags & kHasBias) {
      val += static_cast<AccT>(bias[ccid]);
    }
    if (data_layout == DataLayout::kNCHW) {
      y[index] = static_cast<T>(val);
    } else {
      y[(bid * H + hid) * W * C + wid * C + ccid] = static_cast<T>(val);
    }
  }
}

template <typename T, typename AccT, int flags, int VecSize = 2>
__global__ void GroupNormForwardNHWC(const T* x,
                                     const AccT* mean,
                                     const AccT* var,
                                     const T* scale,
                                     const T* bias,
                                     int N,
                                     int C,
                                     int W,
                                     int imsize,
                                     int groups,
                                     int group_size,
                                     AccT epsilon,
                                     T* y,
                                     AccT* real_var,
                                     const DataLayout data_layout) {
  int index = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
  int stride = gridDim.x * blockDim.x * VecSize;
  int hwc_size = imsize * C;
  int size = N * hwc_size;
  int deal_size = blockDim.x * VecSize;
  AccT x_value[VecSize];
  int ng_value[VecSize];
  int gs_value[VecSize];
  int h_value[VecSize];
  int rem_value[VecSize];

  for (; index < size; index += stride) {
#pragma unroll
    for (int nx = 0; nx < VecSize; ++nx) {
      x_value[nx] = static_cast<AccT>(x[index + nx]);

      int index_nx = index + nx;
      int index_n = index_nx / hwc_size;
      int index_r = index_nx - index_n * hwc_size;
      rem_value[nx] = index_r % C;
      int index_g = rem_value[nx] / group_size;
      ng_value[nx] = index_n * groups + index_g;
      gs_value[nx] = rem_value[nx] - index_g * group_size;
      h_value[nx] = index_r / (W * C);
    }
#pragma unroll
    for (int nx = 0; nx < VecSize; ++nx) {
      AccT x_mean = mean[ng_value[nx]];
      AccT x_var = var[ng_value[nx]];
      x_var = x_var - x_mean * x_mean;

      AccT var_inv = rsqrt(x_var + epsilon);

      if (gs_value[nx] == 0 && h_value[nx] == 0) {
        real_var[ng_value[nx]] = x_var;
      }

      x_value[nx] = (x_value[nx] - x_mean) * var_inv;
      if (flags & kHasScale) {
        x_value[nx] *= static_cast<AccT>(scale[rem_value[nx]]);
      }
      if (flags & kHasBias) {
        x_value[nx] += static_cast<AccT>(bias[rem_value[nx]]);
      }

      y[index + nx] = static_cast<T>(x_value[nx]);
    }
  }
}

template <typename T, typename Context>
void GroupNormKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& scale,
                     const paddle::optional<DenseTensor>& bias,
                     float epsilon,
                     int groups,
                     const std::string& data_layout_str,
                     DenseTensor* y,
                     DenseTensor* mean,
                     DenseTensor* var) {
  using AccT = typename phi::dtype::MPTypeTrait<T>::Type;
  const DataLayout data_layout = phi::StringToDataLayout(data_layout_str);
  const auto scale_ptr = scale.get_ptr();
  const auto bias_ptr = bias.get_ptr();

  const auto x_dims = x.dims();
  const int C = (data_layout == DataLayout::kNCHW ? x_dims[1]
                                                  : x_dims[x_dims.size() - 1]);
  const int group_size = C / groups;

  const int W = (data_layout == DataLayout::kNCHW ? x_dims[x_dims.size() - 1]
                                                  : x_dims[x_dims.size() - 2]);

  dev_ctx.template Alloc<T>(y);
  dev_ctx.template Alloc<AccT>(mean);
  dev_ctx.template Alloc<AccT>(var);
  // temp_var is used to calculate the mean^2
  DenseTensor temp_var;
  temp_var.Resize(var->dims());
  dev_ctx.template Alloc<AccT>(&temp_var);
  phi::funcs::SetConstant<GPUContext, T> set_zero;
  phi::funcs::SetConstant<GPUContext, AccT> set_zero_AccT;
  auto* x_data = x.data<T>();
  auto* y_data = y->data<T>();
  auto* mean_data = mean->data<AccT>();
  auto* var_data = var->data<AccT>();
  auto* temp_var_data = temp_var.data<AccT>();

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
#else
  int block_size = std::min(1024, imsize);
#endif

  int flags =
      (scale_data != nullptr) * kHasScale + (bias_data != nullptr) * kHasBias;

  dim3 grid(group_size, groups, x_dims[0]);
  dim3 threads(block_size, 1, 1);
  if (data_layout == DataLayout::kNCHW) {
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
      ScalarGetMeanAndVarNCHW<T, AccT><<<grids, blocks, 0, dev_ctx.stream()>>>(
          x_data, mean_data, temp_var_data, size);
    } else {
      VectorizedGetMeanAndVarNCHW<T, AccT, vec_size>
          <<<grids, blocks, 0, dev_ctx.stream()>>>(
              x_data, mean_data, temp_var_data, size);
    }

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
                     static_cast<AccT>(epsilon),
                     y_data,
                     var_data,
                     data_layout);
  } else {
    set_zero_AccT(dev_ctx, mean, static_cast<AccT>(0));
    set_zero_AccT(dev_ctx, &temp_var, static_cast<AccT>(0));

#ifdef __HIPCC__
    int block_size_nhwc = std::max(std::min(256, (groups * W)), 64);
#else
    int block_size_nhwc = std::min(1024, (groups * W));
#endif
    dim3 grid_get(x_dims[1], x_dims[0], 1);
    dim3 threads_get(block_size_nhwc, 1, 1);

    GroupNormForwardGetMeanAndVar<T, AccT>
        <<<grid_get, threads_get, 0, dev_ctx.stream()>>>(x_data,
                                                         x_dims[0],
                                                         C,
                                                         W,
                                                         imsize,
                                                         groups,
                                                         group_size,
                                                         mean_data,
                                                         temp_var_data);

    int numel = x.numel();
    constexpr const int vec_size = 2;
    auto config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel, vec_size);
    int grid_nhwc = config.block_per_grid.x;
    int block_nhwc = config.thread_per_block.x;

    UNROLL_ALL_CASES_NHWC(flags,
                          GroupNormForwardNHWC,
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
                          static_cast<AccT>(epsilon),
                          y_data,
                          var_data,
                          data_layout);
  }
}

template <typename T, typename AccT>
void GroupNormDirectCUDAFunctor<T, AccT>::operator()(
    gpuStream_t stream,
    const T* input,
    std::vector<int> input_shape,
    const T* bias,
    const T* scale,
    AccT* temp_variance,
    int groups,
    float eps,
    T* output,
    AccT* mean,
    AccT* variance,
    const DataLayout data_layout) {
  const auto input_ddim = phi::make_ddim(input_shape);
  const int C =
      (data_layout == DataLayout::kNCHW ? input_ddim[1]
                                        : input_ddim[input_ddim.size() - 1]);
  const int group_size = C / groups;
  const int W =
      (data_layout == DataLayout::kNCHW ? input_ddim[input_ddim.size() - 1]
                                        : input_ddim[input_ddim.size() - 2]);

  int image_size = 1;
  if (data_layout == DataLayout::kNCHW) {
    for (int i = 2; i < input_ddim.size(); ++i) {
      image_size *= input_ddim[i];
    }
  } else {
    for (int i = 1; i < input_ddim.size() - 1; ++i) {
      image_size *= input_ddim[i];
    }
  }
#ifdef __HIPCC__
  int block_size = std::max(std::min(256, image_size), 64);
#else
  int block_size = std::min(1024, image_size);
#endif
  dim3 grid(group_size, groups, input_ddim[0]);
  dim3 threads(block_size, 1, 1);
  if (data_layout == DataLayout::kNCHW) {
    constexpr int vec_size = sizeof(float4) / sizeof(T);
    int size = group_size * image_size;  // group element size
    const int max_num_threads = 1024;
    int max_block_size = std::min(size / vec_size, max_num_threads);
    int block_size_nchw = 1;
    while (block_size_nchw < max_block_size) {
      block_size_nchw *= 2;
    }

    block_size_nchw = std::max(block_size_nchw, phi::kps::details::kWarpSize);
    dim3 grids(input_ddim[0] * groups);
    dim3 blocks(block_size_nchw);

    if (size < vec_size * block_size_nchw) {
      phi::ScalarGetMeanAndVarNCHW<T, AccT>
          <<<grids, blocks, 0, stream>>>(input, mean, temp_variance, size);
    } else {
      phi::VectorizedGetMeanAndVarNCHW<T, AccT, vec_size>
          <<<grids, blocks, 0, stream>>>(input, mean, temp_variance, size);
    }
  } else {
#ifdef PADDLE_WITH_HIP
    hipMemset(mean, 0, sizeof(AccT) * input_ddim[0] * groups);
    hipMemset(temp_variance, 0, sizeof(AccT) * input_ddim[0] * groups);
#else
    cudaMemset(mean, 0, sizeof(AccT) * input_ddim[0] * groups);
    cudaMemset(temp_variance, 0, sizeof(AccT) * input_ddim[0] * groups);
#endif

#ifdef __HIPCC__
    int block_size_nhwc = std::max(std::min(256, (groups * W)), 64);
#else
    int block_size_nhwc = std::min(1024, (groups * W));
#endif
    dim3 grid_get(input_ddim[1], input_ddim[0], 1);
    dim3 threads_get(block_size_nhwc, 1, 1);
    phi::GroupNormForwardGetMeanAndVar<T, AccT>
        <<<grid, threads, 0, stream>>>(input,
                                       input_ddim[0],
                                       C,
                                       W,
                                       image_size,
                                       groups,
                                       group_size,
                                       mean,
                                       temp_variance);
  }
  GroupNormForward<T, AccT, 3>
      <<<grid, threads, 0, stream>>>(input,
                                     mean,
                                     temp_variance,
                                     scale,
                                     bias,
                                     input_ddim[0],
                                     C,
                                     W,
                                     image_size,
                                     groups,
                                     group_size,
                                     static_cast<AccT>(eps),
                                     output,
                                     variance,
                                     data_layout);
}
template class GroupNormDirectCUDAFunctor<float, float>;
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
template class GroupNormDirectCUDAFunctor<half, float>;
#endif

}  // namespace phi

PD_REGISTER_KERNEL(group_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::GroupNormKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::BFLOAT16 ||
      kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}
