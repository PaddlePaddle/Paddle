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

#include "paddle/phi/kernels/nanmedian_kernel.h"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/nanmedian_utils.h"
#include "paddle/phi/kernels/top_k_kernel.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"
#endif

constexpr int64_t ELEMWISE_MAX_BLOCK_DIM = 1024;

namespace phi {
template <typename T>
__global__ void KernelNanCounts(const T* input,
                                const int64_t numel,
                                const int64_t pre_dim,
                                const int64_t stride,
                                int64_t* nan_counts) {
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int64_t total1 = 0;
  int64_t total2 = 0;

  for (int64_t j = bx; j < pre_dim; j += gridDim.x) {
    int64_t num = 0;
    int64_t i = tx;
    while (i < stride) {
      int64_t offset = i + j * stride;

      T x = input[offset];
      if (isnan(static_cast<float>(x))) num += 1;

      i += blockDim.x;
    }

    int len = stride > blockDim.x ? blockDim.x : stride;
    num = phi::backends::gpu::reduceSum(num, tx, len);
    if (tx == 0) {
      nan_counts[j] = num;
    }
  }
}

template <typename T>
__global__ void CalcMedianMeanKernel(const T* sort_out_ptr,
                                     const int64_t* sort_indices_ptr,
                                     int64_t* median_val,
                                     T* output,
                                     T div_factor,
                                     const bool is_odd,
                                     const int64_t pre_dim,
                                     const int64_t stride) {
  int64_t begin = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (int64_t index = begin; index < pre_dim; index += step) {
    int64_t pos = static_cast<int64_t>((index + 1) * stride) - 1;
    if (is_odd) {
      median_val[index * 2] = sort_indices_ptr[pos];
      median_val[index * 2 + 1] = sort_indices_ptr[pos];
      output[index] = sort_out_ptr[pos];
    } else {
      T median_val_left = pos > 0 ? sort_out_ptr[pos - 1] : sort_out_ptr[pos];
      T median_val_right = sort_out_ptr[pos];
      median_val[index * 2] =
          pos > 0 ? sort_indices_ptr[pos - 1] : sort_indices_ptr[pos];
      median_val[index * 2 + 1] = sort_indices_ptr[pos];
      output[index] = (median_val_left + median_val_right) / div_factor;
    }
  }
}

template <typename T>
__global__ void CalcMedianMinKernel(const T* sort_out_ptr,
                                    const int64_t* sort_indices_ptr,
                                    int64_t* median_val,
                                    T* output,
                                    T div_factor,
                                    const bool is_odd,
                                    const int64_t pre_dim,
                                    const int64_t stride) {
  int64_t begin = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (int64_t index = begin; index < pre_dim; index += step) {
    int64_t pos = static_cast<int64_t>((index + 1) * stride) - 1;
    if (is_odd) {
      median_val[index] = sort_indices_ptr[pos];
      output[index] = sort_out_ptr[pos];
    } else {
      T median_val_left = pos > 0 ? sort_out_ptr[pos - 1] : sort_out_ptr[pos];
      median_val[index] =
          pos > 0 ? sort_indices_ptr[pos - 1] : sort_indices_ptr[pos];
      output[index] = median_val_left;
    }
  }
}

template <typename T>
__global__ void CalcNanmedianMeanKernel(const T* sort_out_ptr,
                                        const int64_t* sort_indices_ptr,
                                        int64_t* nan_counts,
                                        int64_t* median_val,
                                        T* output,
                                        const bool is_odd,
                                        const int64_t pre_dim,
                                        const int64_t max_valid_num,
                                        const int64_t stride,
                                        const T div_factor,
                                        const T nan_val) {
  int64_t begin = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (int64_t index = begin; index < pre_dim; index += step) {
    int64_t pos = static_cast<int64_t>(index * max_valid_num);
    int64_t nan_cnt = nan_counts[index];
    if (nan_cnt == stride) {
      median_val[index * 2] = -1;
      median_val[index * 2 + 1] = -1;
      output[index] = nan_val;
    } else {
      int64_t nan_k =
          nan_cnt > 0 ? static_cast<int64_t>(stride - nan_cnt) : max_valid_num;
      int64_t row_pos = static_cast<int64_t>(nan_k >> 1);
      pos += row_pos;

      if (nan_k & 1) {
        median_val[index * 2] = sort_indices_ptr[pos];
        median_val[index * 2 + 1] = sort_indices_ptr[pos];
        output[index] = sort_out_ptr[pos];
      } else {
        T median_val_left = pos > 0 ? sort_out_ptr[pos - 1] : sort_out_ptr[pos];
        T median_val_right = sort_out_ptr[pos];
        median_val[index * 2] =
            pos > 0 ? sort_indices_ptr[pos - 1] : sort_indices_ptr[pos];
        median_val[index * 2 + 1] = sort_indices_ptr[pos];
        output[index] = (median_val_left + median_val_right) / div_factor;
      }
    }
  }
}

template <typename T>
__global__ void CalcNanmedianMinKernel(const T* sort_out_ptr,
                                       const int64_t* sort_indices_ptr,
                                       int64_t* nan_counts,
                                       int64_t* median_val,
                                       T* output,
                                       const bool is_odd,
                                       const int64_t pre_dim,
                                       const int64_t max_valid_num,
                                       const int64_t stride,
                                       const T div_factor,
                                       const T nan_val) {
  int64_t begin = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (int64_t index = begin; index < pre_dim; index += step) {
    int64_t pos = static_cast<int64_t>(index * max_valid_num);
    int64_t nan_cnt = nan_counts[index];
    if (nan_cnt == stride) {
      median_val[index] = -1;
      output[index] = nan_val;
    } else {
      int64_t nan_k =
          nan_cnt > 0 ? static_cast<int64_t>(stride - nan_cnt) : max_valid_num;
      int64_t row_pos = static_cast<int64_t>(nan_k >> 1);
      pos += row_pos;

      if (nan_k & 1) {
        median_val[index] = sort_indices_ptr[pos];
        output[index] = sort_out_ptr[pos];
      } else {
        T median_val_left = pos > 0 ? sort_out_ptr[pos - 1] : sort_out_ptr[pos];
        median_val[index] =
            pos > 0 ? sort_indices_ptr[pos - 1] : sort_indices_ptr[pos];
        output[index] = median_val_left;
      }
    }
  }
}

template <typename T, typename Context>
void ProcessMedianKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const std::string& mode,
                         DenseTensor* out,
                         DenseTensor* median_index) {
#ifdef PADDLE_WITH_CUDA
  const auto& exec_policy = thrust::cuda::par.on(dev_ctx.stream());
#else
  const auto& exec_policy = thrust::hip::par.on(dev_ctx.stream());
#endif
  auto stream = dev_ctx.stream();
  const T* x_data = x.data<T>();
  T* out_data = dev_ctx.template Alloc<T>(out);
  int64_t* m_data = dev_ctx.template Alloc<int64_t>(median_index);

  int64_t numel = x.numel();
  auto x_dim = x.dims();
  int x_rank = x_dim.size();
  int64_t stride = x_dim[x_rank - 1];

  PADDLE_ENFORCE_NE(stride,
                    0,
                    common::errors::InvalidArgument(
                        "The input Tensor x's shape[-1] should not "
                        "be 0, but shape is %s now.",
                        x_dim));

  int64_t pre_dim = numel / stride;

  DenseTensor nan_counts;
  int64_t* nan_counts_ptr;
  int64_t max_valid_num = 0;

  bool ignore_nan = true;
  if (ignore_nan) {
    nan_counts.Resize(common::make_ddim({pre_dim}));
    dev_ctx.template Alloc<int64_t>(&nan_counts);
    nan_counts_ptr = nan_counts.data<int64_t>();
    int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, stride);
    int64_t grid_size = pre_dim;
    int64_t max_grid_dim = dev_ctx.GetCUDAMaxGridDimSize()[0];
    grid_size = std::min(grid_size, max_grid_dim);
    KernelNanCounts<T><<<grid_size, block_size, 0, stream>>>(
        x_data, numel, pre_dim, stride, nan_counts_ptr);
    auto nan_stat_mem_cpu =
        phi::memory_utils::Alloc(phi::CPUPlace(), sizeof(int64_t) * 2);
    int64_t* nan_stat_cpu_ptr =
        reinterpret_cast<int64_t*>(nan_stat_mem_cpu->ptr());
    int64_t sum =
        thrust::reduce(exec_policy, nan_counts_ptr, nan_counts_ptr + pre_dim);
    nan_stat_cpu_ptr[0] = sum;
    auto maxx_ptr = thrust::max_element(
        exec_policy, nan_counts_ptr, nan_counts_ptr + pre_dim);
    memory_utils::Copy(phi::CPUPlace(),
                       nan_stat_cpu_ptr + 1,
                       dev_ctx.GetPlace(),
                       maxx_ptr,
                       sizeof(int64_t),
                       stream);
    // all elements are nan values
    T nan_val = std::numeric_limits<T>::quiet_NaN();
    if (nan_stat_cpu_ptr[0] == numel) {
      phi::funcs::SetConstant<Context, T> set_nan;
      set_nan(dev_ctx, out, nan_val);

      phi::funcs::SetConstant<Context, int64_t> set_negatvie;
      set_negatvie(dev_ctx, median_index, static_cast<int64_t>(-1));
      return;
    }

    ignore_nan = nan_stat_cpu_ptr[0] > 0;
    max_valid_num = nan_stat_cpu_ptr[1];
  }

  int64_t sort_k = ignore_nan ? max_valid_num : ((stride >> 1) + 1);
  bool is_ori_odd = stride & 1;

  DenseTensor sort_out, sort_indices;
  auto sort_dim = x.dims();
  int64_t rank = sort_dim.size();
  sort_dim[rank - 1] = sort_k;
  sort_out.Resize(sort_dim);
  sort_indices.Resize(sort_dim);

  dev_ctx.template Alloc<T>(&sort_out);
  T* sort_out_ptr = sort_out.data<T>();
  dev_ctx.template Alloc<int64_t>(&sort_indices);
  int64_t* sort_indices_ptr = sort_indices.data<int64_t>();

  TopkKernel<T, Context>(
      dev_ctx, x, Scalar(sort_k), -1, false, true, &sort_out, &sort_indices);

  T div_factor = static_cast<T>(2.0);
  T nan_val = std::numeric_limits<T>::quiet_NaN();
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, pre_dim);
  if (ignore_nan) {
    if (mode == "avg") {
      CalcNanmedianMeanKernel<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              sort_out_ptr,
              sort_indices_ptr,
              nan_counts_ptr,
              m_data,
              out_data,
              is_ori_odd,
              pre_dim,
              max_valid_num,
              stride,
              div_factor,
              nan_val);
    } else {  // mode == "min"
      CalcNanmedianMinKernel<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              sort_out_ptr,
              sort_indices_ptr,
              nan_counts_ptr,
              m_data,
              out_data,
              is_ori_odd,
              pre_dim,
              max_valid_num,
              stride,
              div_factor,
              nan_val);
    }
  } else {
    if (mode == "avg") {
      CalcMedianMeanKernel<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              sort_out_ptr,
              sort_indices_ptr,
              m_data,
              out_data,
              div_factor,
              is_ori_odd,
              pre_dim,
              sort_k);
    } else {  // mode == "min"
      CalcMedianMinKernel<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              sort_out_ptr,
              sort_indices_ptr,
              m_data,
              out_data,
              div_factor,
              is_ori_odd,
              pre_dim,
              sort_k);
    }
  }
}

template <typename T, typename Context>
void NanmedianKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const IntArray& axes,
                     bool keepdim,
                     const std::string& mode,
                     DenseTensor* out,
                     DenseTensor* median_index) {
  if (x.numel() == 0) {
    phi::Full<T, Context>(
        dev_ctx, phi::IntArray(common::vectorize(out->dims())), NAN, out);
    phi::Full<int64_t, Context>(
        dev_ctx,
        phi::IntArray(common::vectorize(median_index->dims())),
        0,
        median_index);
    return;
  }
  DenseTensor tmp_x;
  auto rank = x.dims().size();
  if ((axes.size() == 0) || rank <= 1) {
    tmp_x = x;
    tmp_x.Resize({x.numel()});
  } else {
    funcs::PreprocessMedianKernel<T, Context>(dev_ctx, x, axes, &tmp_x);
  }

  ProcessMedianKernel<T, Context>(dev_ctx, tmp_x, mode, out, median_index);
}
}  // namespace phi

PD_REGISTER_KERNEL(nanmedian,
                   GPU,
                   ALL_LAYOUT,
                   phi::NanmedianKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}
