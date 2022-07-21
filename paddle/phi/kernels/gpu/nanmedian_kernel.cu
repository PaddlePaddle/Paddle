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

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/impl/nanmedian_kernel_impl.h"
#include "paddle/phi/kernels/top_k_kernel.h"

namespace phi {

using paddle::platform::PADDLE_CUDA_NUM_THREADS;

inline int GET_BLOCKS(const int N) {
  return (N + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS;
}

template <typename T>
__global__ void KernelNanCounts(const T* input,
                                const int numel,
                                const int64_t pre_dim,
                                const int64_t stride,
                                T min_val,
                                int64_t* nan_total,
                                int64_t* nan_counts) {
  extern __shared__ int64_t buf[];
  for (int i = threadIdx.x; i < pre_dim; i += blockDim.x) {
    buf[i] = 0;
    nan_counts[i] = 0;
  }

  if (threadIdx.x == 0) {
    nan_total[0] = 0;
    nan_total[1] = 0;
  }

  __syncthreads();

  CUDA_KERNEL_LOOP(index, numel) {
    const T x = input[index];
    if (isnan(static_cast<float>(x))) {
      auto bin = static_cast<int64_t>(index / stride);
      paddle::platform::CudaAtomicAdd(&buf[bin], 1);
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < pre_dim; i += blockDim.x) {
    paddle::platform::CudaAtomicAdd(&nan_counts[i], buf[i]);
    paddle::platform::CudaAtomicAdd(&nan_total[0], buf[i]);
    paddle::platform::CudaAtomicMax(&nan_total[1], stride - buf[i]);
  }
}

template <typename T>
__global__ void CalcMedianKernel(const T* sort_out_ptr,
                                 const int64_t* sort_indices_ptr,
                                 int64_t* median_val,
                                 T* output,
                                 T div_factor,
                                 const bool is_odd,
                                 const int64_t pre_dim,
                                 const int64_t stride) {
  CUDA_KERNEL_LOOP(index, pre_dim) {
    int64_t pos = static_cast<int64_t>((index + 1) * stride) - 1;
    if (is_odd) {
      median_val[index * 2] = sort_indices_ptr[pos];
      median_val[index * 2 + 1] = sort_indices_ptr[pos];
      output[index] = sort_out_ptr[pos];
    } else {
      median_val[index * 2] =
          pos > 0 ? sort_indices_ptr[pos - 1] : sort_indices_ptr[pos];
      median_val[index * 2 + 1] = sort_indices_ptr[pos];
      T median_val_left = pos > 0 ? sort_out_ptr[pos - 1] : sort_out_ptr[pos];
      T median_val_right = sort_out_ptr[pos];
      output[index] = (median_val_left + median_val_right) / div_factor;
    }
  }
}

template <typename T>
__global__ void CalcNanmedianKernel(const T* sort_out_ptr,
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
  CUDA_KERNEL_LOOP(index, pre_dim) {
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
        median_val[index * 2] =
            pos > 0 ? sort_indices_ptr[pos - 1] : sort_indices_ptr[pos];
        median_val[index * 2 + 1] = sort_indices_ptr[pos];
        T median_val_left = pos > 0 ? sort_out_ptr[pos - 1] : sort_out_ptr[pos];
        T median_val_right = sort_out_ptr[pos];
        output[index] = (median_val_left + median_val_right) / div_factor;
      }
    }
  }
}

template <typename T, typename Context>
void ProcessMedianKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         bool ignore_nan,
                         DenseTensor* out,
                         int64_t* m_ptr) {
  bool should_ignore_nan = ignore_nan;
  auto stream = dev_ctx.stream();

  const T* x_ptr = x.data<T>();
  T* o_ptr = dev_ctx.template Alloc<T>(out);
  int64_t numel = x.numel();
  auto x_dim = x.dims();
  int64_t x_rank = x_dim.size();
  int64_t stride = x_dim[x_rank - 1];
  int64_t pre_dim = numel / stride;
  int64_t i = 0;

  DenseTensor nan_counts, nan_stat;
  int64_t* nan_counts_ptr;
  int64_t max_valid_num = 0;
  if (should_ignore_nan) {
    nan_counts.Resize(phi::make_ddim({pre_dim}));
    dev_ctx.template Alloc<int64_t>(&nan_counts);
    nan_counts_ptr = nan_counts.data<int64_t>();
    nan_stat.Resize(phi::make_ddim({2}));
    int64_t* nan_stat_mem = dev_ctx.template Alloc<int64_t>(&nan_stat);
    int64_t* nan_stat_ptr = nan_stat.data<int64_t>();

    KernelNanCounts<T><<<GET_BLOCKS(numel),
                         PADDLE_CUDA_NUM_THREADS,
                         pre_dim * sizeof(int64_t),
                         stream>>>(x_ptr,
                                   numel,
                                   pre_dim,
                                   stride,
                                   std::numeric_limits<T>::min(),
                                   nan_stat_ptr,
                                   nan_counts_ptr);

    auto nan_stat_mem_cpu =
        paddle::memory::Alloc(phi::CPUPlace(), sizeof(int64_t) * 2);
    int64_t* nan_stat_cpu_ptr =
        reinterpret_cast<int64_t*>(nan_stat_mem_cpu->ptr());
    paddle::memory::Copy(phi::CPUPlace(),
                         nan_stat_cpu_ptr,
                         dev_ctx.GetPlace(),
                         nan_stat_mem,
                         sizeof(int64_t) * 2,
                         stream);

    // all elements are nan values
    T nan_val = std::numeric_limits<T>::quiet_NaN();
    if (nan_stat_cpu_ptr[0] == numel) {
      FullLikeKernel<T, Context>(dev_ctx, x, nan_val, x.dtype(), out);
      return;
    }

    should_ignore_nan = nan_stat_cpu_ptr[0] > 0;
    max_valid_num = nan_stat_cpu_ptr[1];
  }

  int64_t sort_k = should_ignore_nan ? max_valid_num : ((stride >> 1) + 1);
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
  if (should_ignore_nan) {
    CalcNanmedianKernel<T>
        <<<GET_BLOCKS(pre_dim), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
            sort_out_ptr,
            sort_indices_ptr,
            nan_counts_ptr,
            m_ptr,
            o_ptr,
            is_ori_odd,
            pre_dim,
            max_valid_num,
            stride,
            div_factor,
            nan_val);
  } else {
    CalcMedianKernel<T>
        <<<GET_BLOCKS(pre_dim), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
            sort_out_ptr,
            sort_indices_ptr,
            m_ptr,
            o_ptr,
            div_factor,
            is_ori_odd,
            pre_dim,
            sort_k);
  }
}

template <typename T, typename Context>
void BaseMedianKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      const IntArray& axes,
                      bool ignore_nan,
                      DenseTensor* out,
                      DenseTensor* median_index) {
  DenseTensor x;
  auto rank = input.dims().size();
  if ((axes.size() == 0) || rank <= 1) {
    x = input;
    x.Resize({input.numel()});
  } else {
    PreprocessMedianKernel<T, Context>(dev_ctx, input, axes, &x);
  }

  int64_t* m_ptr = dev_ctx.template Alloc<int64_t>(median_index);
  ProcessMedianKernel<T, Context>(dev_ctx, x, ignore_nan, out, m_ptr);
  out->Resize(out->dims());
}

template <typename T, typename Context>
void NanmedianKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const IntArray& axes,
                     bool keepdim,
                     DenseTensor* out,
                     DenseTensor* median_index) {
  BaseMedianKernel<T, Context>(dev_ctx, x, axes, true, out, median_index);
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
                   phi::dtype::float16) {}
