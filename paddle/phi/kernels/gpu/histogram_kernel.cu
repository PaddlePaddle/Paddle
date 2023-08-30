// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/histogram_kernel.h"

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

using IndexType = int64_t;
using phi::PADDLE_CUDA_NUM_THREADS;

inline int GET_BLOCKS(const int N) {
  return (N + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS;
}

template <typename T, typename IndexType>
__device__ static IndexType GetBin(T input_value,
                                   T min_value,
                                   T max_value,
                                   int64_t nbins) {
  IndexType bin = static_cast<int>((input_value - min_value) * nbins /
                                   (max_value - min_value));
  IndexType output_index = bin < nbins - 1 ? bin : nbins - 1;
  return output_index;
}

template <typename T, typename IndexType>
__global__ void KernelHistogram(const T* input,
                                const int total_elements,
                                const bool has_weight,
                                const T* weight,
                                const int64_t nbins,
                                const T* min_value,
                                const T* max_value,
                                T* output) {
  extern __shared__ __align__(sizeof(T)) unsigned char buf_hist_tmp[];
  T* buf_hist = reinterpret_cast<T*>(buf_hist_tmp);
  for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
    buf_hist[i] = 0;
  }
  __syncthreads();

  CUDA_KERNEL_LOOP(input_index, total_elements) {
    // const IndexType input_index = threadIdx.x + blockIdx.x * blockDim.x;
    const auto input_value = input[input_index];
    const auto weight_value =
        has_weight ? weight[input_index] : static_cast<T>(1);
    if (input_value >= *min_value && input_value <= *max_value) {
      const IndexType output_index =
          GetBin<T, IndexType>(input_value, *min_value, *max_value, nbins);
      phi::CudaAtomicAdd(&buf_hist[output_index], weight_value);
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
    phi::CudaAtomicAdd(&output[i], buf_hist[i]);
  }
}

template <int BlockSize, typename T, typename IndexType>
__global__ void KernelHistogramDensity(const T* input,
                                       const int total_elements,
                                       const bool has_weight,
                                       const T* weight,
                                       const int64_t nbins,
                                       const T* min_value,
                                       const T* max_value,
                                       float* output) {
  T count_weight = 0;
  T total_weight;
  __shared__ T total[BlockSize];
  extern __shared__ __align__(sizeof(float)) unsigned char buf_histd_tmp[];
  float* buf_histd = reinterpret_cast<float*>(buf_histd_tmp);

  for (int i = threadIdx.x; i < (total_elements); i += BlockSize) {
    const auto input_value = input[i];
    const auto weight_value = has_weight ? weight[i] : static_cast<T>(1);
    if (input_value >= *min_value && input_value <= *max_value) {
      count_weight += weight_value;
    }
  }
  total[threadIdx.x] = count_weight;
  __syncthreads();

// reduce the count with init value 0, and output accuracy.
#ifdef PADDLE_WITH_CUDA
  total_weight = thrust::reduce(thrust::device, total, total + BlockSize, 0.0);
#else
  // HIP thrust::reduce not support __device__
  for (int s = BlockSize / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      total[threadIdx.x] += total[threadIdx.x + s];
    }
    __syncthreads();
  }
  total_weight = total[0];
#endif

  for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
    buf_histd[i] = 0;
  }
  __syncthreads();

  const float interval_len =
      static_cast<float>(*max_value - *min_value) / nbins;
  CUDA_KERNEL_LOOP(input_index, total_elements) {
    // const IndexType input_index = threadIdx.x + blockIdx.x * blockDim.x;
    const auto input_value = input[input_index];
    auto weight_value = has_weight ? weight[input_index] : static_cast<T>(1);
    if (input_value >= *min_value && input_value <= *max_value) {
      const IndexType output_index =
          GetBin<T, IndexType>(input_value, *min_value, *max_value, nbins);
      float prob_value = static_cast<float>(weight_value) /
                         static_cast<float>(total_weight) / interval_len;
      phi::CudaAtomicAdd(&buf_histd[output_index], prob_value);
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
    phi::CudaAtomicAdd(&output[i], buf_histd[i]);
  }
}

template <typename T>
__global__ void KernelMinMax(const T* input,
                             const int numel,
                             const int block_num,
                             T* min_ptr,
                             T* max_ptr) {
  int64_t index = threadIdx.x + blockIdx.x * blockDim.x;
  int64_t i = index;
  T min_value = static_cast<T>(i < numel ? input[i] : input[0]);
  T max_value = static_cast<T>(i < numel ? input[i] : input[0]);

  for (; i < numel; i += blockDim.x * gridDim.x) {
    T value = static_cast<T>(input[i]);
    min_value = value < min_value ? value : min_value;
    max_value = value > max_value ? value : max_value;
  }
  if (max_ptr && min_ptr) {
    __syncthreads();
    T block_min_value = phi::funcs::BlockReduceMin<T>(min_value, FINAL_MASK);
    T block_max_value = phi::funcs::BlockReduceMax<T>(max_value, FINAL_MASK);

    if (threadIdx.x == 0) {
      min_ptr[blockIdx.x] = block_min_value;
      max_ptr[blockIdx.x] = block_max_value;
    }
  }
  __syncthreads();
  if (index == 0) {
    if (min_ptr && max_ptr) {
      min_value = min_ptr[0];
      max_value = max_ptr[0];
      for (int64_t i = 1; i < block_num; i++) {
        min_ptr[0] = min_ptr[i] < min_value ? min_ptr[i] : min_value;
        max_ptr[0] = max_ptr[i] > max_value ? max_ptr[i] : max_value;
      }
      if (min_ptr[0] == max_ptr[0]) {
        min_ptr[0] = min_ptr[0] - 1;
        max_ptr[0] = max_ptr[0] + 1;
      }
    }
  }
}

template <typename T>
__global__ void KernelMinMax(const T min_value,
                             const T max_value,
                             T* min_ptr,
                             T* max_ptr) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    min_ptr[0] = min_value;
    max_ptr[0] = max_value;
  }
}

template <typename T, typename Context>
void HistogramKernel(const Context& dev_ctx,
                     const DenseTensor& input,
                     const paddle::optional<DenseTensor>& weight,
                     int64_t bins,
                     int min,
                     int max,
                     bool density,
                     DenseTensor* output) {
  auto& nbins = bins;
  auto& minval = min;
  auto& maxval = max;

  const T* input_data = input.data<T>();
  const int input_numel = input.numel();

  if (input_data == nullptr) {
    dev_ctx.template Alloc<T>(output);
    phi::funcs::SetConstant<Context, T>()(dev_ctx, output, static_cast<T>(0));
    return;
  }

  T output_min = static_cast<T>(minval);
  T output_max = static_cast<T>(maxval);
  DenseTensor min_max;
  int block_num = GET_BLOCKS(input_numel);
  min_max.Resize({2 * block_num});
  auto* min_block_ptr = dev_ctx.template Alloc<T>(&min_max);
  auto* max_block_ptr = min_block_ptr + block_num;
  if (output_min == output_max) {
    KernelMinMax<T><<<GET_BLOCKS(input_numel),
                      PADDLE_CUDA_NUM_THREADS,
                      0,
                      dev_ctx.stream()>>>(
        input_data, input_numel, block_num, min_block_ptr, max_block_ptr);
  } else {
    KernelMinMax<T><<<1, 1, 0, dev_ctx.stream()>>>(
        output_min, output_max, min_block_ptr, max_block_ptr);
  }

  PADDLE_ENFORCE_EQ((std::isinf(static_cast<float>(output_min)) ||
                     std::isnan(static_cast<float>(output_max)) ||
                     std::isinf(static_cast<float>(output_min)) ||
                     std::isnan(static_cast<float>(output_max))),
                    false,
                    phi::errors::OutOfRange("range of min, max is not finite"));
  PADDLE_ENFORCE_GE(
      output_max,
      output_min,
      phi::errors::InvalidArgument(
          "max must be larger or equal to min. If min and max are both zero, "
          "the minimum and maximum values of the data are used. "
          "But received max is %d, min is %d",
          maxval,
          minval));

  bool has_weight = weight.is_initialized();
  const T* weight_data = has_weight ? weight->data<T>() : nullptr;

  auto stream = dev_ctx.stream();
  if (!density) {
    T* out_data = dev_ctx.template Alloc<T>(output);
    phi::funcs::SetConstant<Context, T>()(dev_ctx, output, static_cast<T>(0));
    KernelHistogram<T, IndexType><<<GET_BLOCKS(input_numel),
                                    PADDLE_CUDA_NUM_THREADS,
                                    nbins * sizeof(int64_t),
                                    stream>>>(input_data,
                                              input_numel,
                                              has_weight,
                                              weight_data,
                                              nbins,
                                              min_block_ptr,
                                              max_block_ptr,
                                              out_data);
  } else {
    float* out_data = dev_ctx.template Alloc<float>(output);
    phi::funcs::SetConstant<Context, float>()(
        dev_ctx, output, static_cast<float>(0));
    KernelHistogramDensity<PADDLE_CUDA_NUM_THREADS, T, IndexType>
        <<<GET_BLOCKS(input_numel),
           PADDLE_CUDA_NUM_THREADS,
           nbins * sizeof(int64_t),
           stream>>>(input_data,
                     input_numel,
                     has_weight,
                     weight_data,
                     nbins,
                     min_block_ptr,
                     max_block_ptr,
                     out_data);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(histogram,
                   GPU,
                   ALL_LAYOUT,
                   phi::HistogramKernel,
                   float,
                   double,
                   int,
                   int64_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
