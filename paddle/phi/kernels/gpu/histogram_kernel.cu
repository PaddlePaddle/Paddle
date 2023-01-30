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

<<<<<<< HEAD
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
=======
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

using IndexType = int64_t;
<<<<<<< HEAD
using phi::PADDLE_CUDA_NUM_THREADS;
=======
using paddle::platform::PADDLE_CUDA_NUM_THREADS;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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
                                const int64_t nbins,
                                const T min_value,
                                const T max_value,
                                int64_t* output) {
  extern __shared__ int64_t buf_hist[];
  for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
    buf_hist[i] = 0;
  }
  __syncthreads();

  CUDA_KERNEL_LOOP(input_index, total_elements) {
    // const IndexType input_index = threadIdx.x + blockIdx.x * blockDim.x;
    const auto input_value = input[input_index];
    if (input_value >= min_value && input_value <= max_value) {
      const IndexType output_index =
          GetBin<T, IndexType>(input_value, min_value, max_value, nbins);
<<<<<<< HEAD
      phi::CudaAtomicAdd(&buf_hist[output_index], 1);
=======
      paddle::platform::CudaAtomicAdd(&buf_hist[output_index], 1);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
<<<<<<< HEAD
    phi::CudaAtomicAdd(&output[i], buf_hist[i]);
=======
    paddle::platform::CudaAtomicAdd(&output[i], buf_hist[i]);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  }
}

template <typename T, typename Context>
void HistogramKernel(const Context& dev_ctx,
                     const DenseTensor& input,
                     int64_t bins,
                     int min,
                     int max,
                     DenseTensor* output) {
  auto& nbins = bins;
  auto& minval = min;
  auto& maxval = max;

  const T* input_data = input.data<T>();
  const int input_numel = input.numel();

<<<<<<< HEAD
  int64_t* out_data = dev_ctx.template Alloc<int64_t>(output);
=======
  int64_t* out_data = output->mutable_data<int64_t>(dev_ctx.GetPlace());
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  phi::funcs::SetConstant<Context, int64_t>()(
      dev_ctx, output, static_cast<int64_t>(0));

  if (input_data == nullptr) return;

  T output_min = static_cast<T>(minval);
  T output_max = static_cast<T>(maxval);

  if (output_min == output_max) {
    auto input_x = phi::EigenVector<T>::Flatten(input);

    DenseTensor input_min_t, input_max_t;
<<<<<<< HEAD
    input_min_t.Resize({1});
    input_max_t.Resize({1});
    auto* input_min_data = dev_ctx.template Alloc<T>(&input_min_t);
    auto* input_max_data = dev_ctx.template Alloc<T>(&input_max_t);
=======
    auto* input_min_data = input_min_t.mutable_data<T>({1}, dev_ctx.GetPlace());
    auto* input_max_data = input_max_t.mutable_data<T>({1}, dev_ctx.GetPlace());
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    auto input_min_scala = phi::EigenScalar<T>::From(input_min_t);
    auto input_max_scala = phi::EigenScalar<T>::From(input_max_t);

    auto* place = dev_ctx.eigen_device();
    input_min_scala.device(*place) = input_x.minimum();
    input_max_scala.device(*place) = input_x.maximum();

    DenseTensor input_min_cpu, input_max_cpu;
    paddle::framework::TensorCopySync(
        input_min_t, phi::CPUPlace(), &input_min_cpu);
    paddle::framework::TensorCopySync(
        input_max_t, phi::CPUPlace(), &input_max_cpu);

    output_min = input_min_cpu.data<T>()[0];
    output_max = input_max_cpu.data<T>()[0];
  }
  if (output_min == output_max) {
    output_min = output_min - 1;
    output_max = output_max + 1;
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

  auto stream = dev_ctx.stream();
  KernelHistogram<T, IndexType><<<GET_BLOCKS(input_numel),
                                  PADDLE_CUDA_NUM_THREADS,
                                  nbins * sizeof(int64_t),
                                  stream>>>(
      input_data, input_numel, nbins, output_min, output_max, out_data);
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
