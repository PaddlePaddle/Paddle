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

#include "paddle/phi/kernels/histogram_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/functors.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

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
  IndexType bin = static_cast<IndexType>((input_value - min_value) * nbins /
                                         (max_value - min_value));
  IndexType output_index = bin < nbins - 1 ? bin : nbins - 1;
  return output_index;
}

template <typename T, typename IndexType, typename Out_T>
__global__ void KernelHistogram(const T* input,
                                const T* weight,
                                const int total_elements,
                                const int64_t nbins,
                                const T* min_value,
                                const T* max_value,
                                Out_T* output) {
  extern __shared__ float buf_hist[];
  for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
    buf_hist[i] = 0;
  }
  __syncthreads();

  CUDA_KERNEL_LOOP(input_index, total_elements) {
    // const IndexType input_index = threadIdx.x + blockIdx.x * blockDim.x;
    const auto input_value = input[input_index];
    if (input_value >= *min_value && input_value <= *max_value) {
      const IndexType output_index =
          GetBin<T, IndexType>(input_value, *min_value, *max_value, nbins);
      phi::CudaAtomicAdd(&buf_hist[output_index],
                         weight ? static_cast<float>(weight[input_index]) : 1);
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
    phi::CudaAtomicAdd(&output[i], buf_hist[i]);
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

__global__ void KernelMul(float* data, float* scale, int64_t numel) {
  size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < numel) {
    data[index] /= *scale;
  }
}

template <typename T, typename Context>
void HistogramKernel(const Context& dev_ctx,
                     const DenseTensor& input,
                     const paddle::optional<DenseTensor>& weight,
                     int64_t bins,
                     float min,
                     float max,
                     bool density,
                     DenseTensor* output) {
  auto& nbins = bins;
  auto& minval = min;
  auto& maxval = max;

  const T* input_data = input.data<T>();
  const int input_numel = input.numel();
  auto weight_data = weight.get_ptr() ? weight.get_ptr()->data<T>() : nullptr;

  if (input_data == nullptr) return;

  T output_min = static_cast<T>(minval);
  T output_max = static_cast<T>(maxval);
  DenseTensor min_max;
  int block_num = GET_BLOCKS(input_numel);
  min_max.Resize({2 * block_num});
  auto* min_block_ptr = dev_ctx.template Alloc<T>(&min_max);
  auto* max_block_ptr = min_block_ptr + block_num;
  if (min == max) {
    KernelMinMax<T><<<GET_BLOCKS(input_numel),
                      PADDLE_CUDA_NUM_THREADS,
                      0,
                      dev_ctx.stream()>>>(
        input_data, input_numel, block_num, min_block_ptr, max_block_ptr);
  } else {
    KernelMinMax<T><<<1, 1, 0, dev_ctx.stream()>>>(
        output_min, output_max, min_block_ptr, max_block_ptr);
  }

  // copy min max value from GPU to CPU
  std::vector<T> min_max_vec;
  phi::TensorToVector(min_max, dev_ctx, &min_max_vec);
  output_min = min_max_vec[0];
  output_max = min_max_vec[1];

  // check if out of range
  double range =
      static_cast<double>(output_max) - static_cast<double>(output_min);
  PADDLE_ENFORCE_LT(
      range,
      static_cast<double>(std::numeric_limits<T>::max()),
      common::errors::InvalidArgument(
          "The range of max - min is out of range for target type, "
          "current kernel type is %s, the range should less than %f "
          "but now min is %f, max is %f.",
          typeid(T).name(),
          std::numeric_limits<T>::max(),
          output_min,
          output_max));

  PADDLE_ENFORCE_EQ(
      (std::isinf(static_cast<float>(output_min)) ||
       std::isnan(static_cast<float>(output_max)) ||
       std::isinf(static_cast<float>(output_min)) ||
       std::isnan(static_cast<float>(output_max))),
      false,
      common::errors::OutOfRange("range of min, max is not finite"));
  PADDLE_ENFORCE_GE(
      output_max,
      output_min,
      common::errors::InvalidArgument(
          "max must be larger or equal to min. If min and max are both zero, "
          "the minimum and maximum values of the data are used. "
          "But received max is %d, min is %d",
          maxval,
          minval));

  auto stream = dev_ctx.stream();

  if (!density && !weight_data) {
    int64_t* out_data = dev_ctx.template Alloc<int64_t>(output);
    phi::funcs::SetConstant<Context, int64_t>()(dev_ctx, output, 0);
    KernelHistogram<T, IndexType, int64_t><<<GET_BLOCKS(input_numel),
                                             PADDLE_CUDA_NUM_THREADS,
                                             nbins * sizeof(int64_t),
                                             stream>>>(input_data,
                                                       weight_data,
                                                       input_numel,
                                                       nbins,
                                                       min_block_ptr,
                                                       max_block_ptr,
                                                       out_data);
    return;

  } else {
    float* out_data = dev_ctx.template Alloc<float>(output);
    phi::funcs::SetConstant<Context, float>()(
        dev_ctx, output, static_cast<float>(0));
    KernelHistogram<T, IndexType, float><<<GET_BLOCKS(input_numel),
                                           PADDLE_CUDA_NUM_THREADS,
                                           nbins * sizeof(int64_t),
                                           stream>>>(input_data,
                                                     weight_data,
                                                     input_numel,
                                                     nbins,
                                                     min_block_ptr,
                                                     max_block_ptr,
                                                     out_data);
    if (density) {
      DenseTensor sum = phi::Sum<float, Context>(
          dev_ctx, *output, phi::IntArray({0}), phi::DataType::FLOAT32, false);
      float gap = static_cast<float>(nbins) /
                  static_cast<float>(output_max - output_min);
      std::vector<const DenseTensor*> ins = {output};
      std::vector<DenseTensor*> outs = {output};
      auto functor = phi::funcs::ScaleFunctor<float>(gap);
      phi::funcs::ElementwiseKernel<float>(dev_ctx, ins, &outs, functor);
      KernelMul<<<GET_BLOCKS(static_cast<int>(bins)),
                  PADDLE_CUDA_NUM_THREADS>>>(out_data, sum.data<float>(), bins);
    }
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
                   int64_t) {}
