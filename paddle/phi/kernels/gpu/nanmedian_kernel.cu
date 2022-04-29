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
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
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
                                int64_t* nan_counts,
                                T* output) {
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
    if (isnan(x)) {
      auto bin = static_cast<int64_t>(index / stride);
      paddle::platform::CudaAtomicAdd(&buf[bin], 1);
      // NOTE: at this moment paddle.sort does not suppert nan values
      output[index] = min_val;
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
__global__ void CalcMedianKernel(const T* sort_out,
                                 T* median_val,
                                 T* output,
                                 const bool is_odd,
                                 const int64_t pre_dim,
                                 const int64_t stride) {
  T div_factor = static_cast<T>(2.0);
  CUDA_KERNEL_LOOP(index, pre_dim) {
    int64_t pos = static_cast<int64_t>((index + 1) * stride) - 1;
    if (is_odd) {
      median_val[index * 2] = sort_out[pos];
      median_val[index * 2 + 1] = sort_out[pos];
      output[index] = sort_out[pos];
    } else {
      median_val[index * 2] = pos > 1 ? sort_out[pos - 1] : sort_out[pos];
      median_val[index * 2 + 1] = sort_out[pos];
      output[index] =
          (median_val[index * 2] + median_val[index * 2 + 1]) / div_factor;
    }
  }
}

template <typename T>
__global__ void CalcNanmedianKernel(const T* sort_out,
                                    int64_t* nan_counts,
                                    T* median_val,
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
      median_val[index * 2] = nan_val;
      median_val[index * 2 + 1] = nan_val;
      output[index] = nan_val;
    } else {
      bool check_odd = is_odd;
      int64_t nan_k =
          nan_cnt > 0 ? static_cast<int64_t>(stride - nan_cnt) : max_valid_num;
      pos += static_cast<int64_t>(nan_k >> 1);
      check_odd = nan_k & 1;

      if (check_odd) {
        median_val[index * 2] = sort_out[pos];
        median_val[index * 2 + 1] = sort_out[pos];
        output[index] = sort_out[pos];
      } else {
        median_val[index * 2] = pos > 0 ? sort_out[pos - 1] : sort_out[pos];
        median_val[index * 2 + 1] = sort_out[pos];
        output[index] =
            (median_val[index * 2] + median_val[index * 2 + 1]) / div_factor;
      }
    }
  }
}

template <typename T, typename Context>
void NanmedianKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     bool ignore_nan,
                     DenseTensor* out,
                     DenseTensor* medians) {
  auto stream = dev_ctx.stream();
  auto* ctx =
      reinterpret_cast<const paddle::platform::CUDADeviceContext*>(&dev_ctx);

  const T* x_ptr = x.data<T>();
  T* o_ptr = dev_ctx.template Alloc<T>(out);
  T* m_ptr = dev_ctx.template Alloc<T>(medians);

  int64_t numel = x.numel();
  auto x_dim = x.dims();
  int64_t x_rank = x_dim.size();
  int64_t stride = x_dim[x_rank - 1];
  int64_t pre_dim = numel / stride;
  int64_t i = 0;

  int64_t half_stride = (stride >> 1) + 1;
  bool is_ori_odd = stride & 1;

  DenseTensor sort_out;
  auto sort_dim = x.dims();
  sort_dim[x_rank - 1] = half_stride;

  sort_out.Resize(sort_dim);
  dev_ctx.template Alloc<T>(&sort_out);
  T* sort_out_ptr = sort_out.data<T>();

  std::vector<int64_t> out_dim_vec = vectorize<int64_t>(sort_dim);
  DenseTensor indices = phi::Empty<T, Context>(dev_ctx, IntArray(out_dim_vec));

  if (ignore_nan) {
    DenseTensor nan_counts, nan_stat, nonnan_x;

    nan_counts.Resize(phi::make_ddim({pre_dim}));
    dev_ctx.template Alloc<int64_t>(&nan_counts);
    int64_t* nan_counts_ptr = nan_counts.data<int64_t>();

    nan_stat.Resize(phi::make_ddim({2}));
    int64_t* nan_stat_mem = dev_ctx.template Alloc<int64_t>(&nan_stat);
    int64_t* nan_stat_ptr = nan_stat.data<int64_t>();

    nonnan_x.Resize(x.dims());
    dev_ctx.template Alloc<T>(&nonnan_x);
    T* nonnan_x_ptr = nonnan_x.data<T>();

    KernelNanCounts<T><<<GET_BLOCKS(numel),
                         PADDLE_CUDA_NUM_THREADS,
                         pre_dim * sizeof(int64_t),
                         stream>>>(x_ptr,
                                   numel,
                                   pre_dim,
                                   stride,
                                   std::numeric_limits<T>::min(),
                                   nan_stat_ptr,
                                   nan_counts_ptr,
                                   nonnan_x_ptr);

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

    // all elements are nan
    T nan_val = std::numeric_limits<T>::quiet_NaN();
    if (nan_stat_cpu_ptr[0] == numel) {
      FullLikeKernel<T, Context>(dev_ctx, x, nan_val, x.dtype(), out);
      return;
    }

    if (nan_stat_cpu_ptr[0] > 0) {
      int64_t max_valid_num = nan_stat_cpu_ptr[1];
      T div_factor = static_cast<T>(2.0);

      TopkKernel<T, Context>(dev_ctx,
                             x,
                             Scalar(max_valid_num),
                             -1,
                             true,
                             true,
                             &sort_out,
                             &indices);

      CalcNanmedianKernel<
          T><<<GET_BLOCKS(pre_dim), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
          sort_out_ptr,
          nan_counts_ptr,
          m_ptr,
          o_ptr,
          is_ori_odd,
          pre_dim,
          max_valid_num,
          stride,
          div_factor,
          nan_val);

      return;
    }
  }

  TopkKernel<T, Context>(
      dev_ctx, x, Scalar(half_stride), -1, true, true, &sort_out, &indices);

  CalcMedianKernel<
      T><<<GET_BLOCKS(pre_dim), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
      sort_out_ptr, m_ptr, o_ptr, is_ori_odd, pre_dim, half_stride);
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
