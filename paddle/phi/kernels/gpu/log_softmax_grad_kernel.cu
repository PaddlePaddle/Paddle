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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/log_softmax_functor.h"
#include "paddle/phi/kernels/gpu/log_softmax_funcs.h"
#include "paddle/phi/kernels/log_softmax_grad_kernel.h"

namespace phi {

int GetNearGreaterPowerOfTwoGrad(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) {
    ++log2_value;
  }
  return 1 << log2_value;
}

// Backward below
#define LAUNCH_WARP_BACKWARD_COMPUTE(near_greater_power_of_two)     \
  case near_greater_power_of_two:                                   \
    ComputeLogSoftmaxBackwardInWarp<                                \
        T,                                                          \
        AccT,                                                       \
        near_greater_power_of_two><<<blocks, threads, 0, stream>>>( \
        output, grad_output, grad_input, outer_size, dim_size);     \
    break;

template <typename T, typename AccT, int NearGreaterPowerOfTwo>
__global__ void ComputeLogSoftmaxBackwardInWarp(const T *output,
                                                const T *grad_output,
                                                T *grad_input,
                                                int batch_size,
                                                int element_count) {
  constexpr int near_greater_power_of_two = NearGreaterPowerOfTwo;
  constexpr int kernel_warp_size =
      (near_greater_power_of_two < 32) ? near_greater_power_of_two : 32;
  constexpr int warp_iter = near_greater_power_of_two / kernel_warp_size;
  int batch_id = blockDim.y * blockIdx.x + threadIdx.y;

  int thread_in_warp_idx = threadIdx.x;

  // 1.read data from global memory to registers
  AccT output_register[warp_iter];       // NOLINT
  AccT grad_output_register[warp_iter];  // NOLINT
  int effective_element_count = (batch_id < batch_size) ? element_count : 0;
  for (int iter = 0; iter < warp_iter; ++iter) {
    int element_index = thread_in_warp_idx + iter * kernel_warp_size;
    if (element_index < effective_element_count) {
      output_register[iter] =
          static_cast<AccT>(output[batch_id * element_count + element_index]);
      grad_output_register[iter] = static_cast<AccT>(
          grad_output[batch_id * element_count + element_index]);
    } else {
      output_register[iter] = static_cast<AccT>(0);
      grad_output_register[iter] = static_cast<AccT>(0);
    }
  }

  // 2. For each warp, accumulate all thread registers
  AccT sum = grad_output_register[0];
#pragma unroll
  for (int iter = 1; iter < warp_iter; ++iter) {
    sum += grad_output_register[iter];
  }
  sum = WarpReduceSum<AccT, kernel_warp_size>(sum);

// 3. write result in grad_input
#pragma unroll
  for (int iter = 0; iter < warp_iter; ++iter) {
    int element_index = thread_in_warp_idx + iter * kernel_warp_size;
    if (element_index < effective_element_count) {
      grad_input[batch_id * element_count + element_index] = static_cast<T>(
          (grad_output_register[iter] - std::exp(output_register[iter]) * sum));
    }
  }
}

template <typename T, typename AccT>
void LaunchSoftmaxBackwardForLastAxis(T *grad_input,
                                      const T *grad_output,
                                      const T *output,
                                      int dim_size,
                                      int outer_size,
                                      gpuStream_t stream) {
  int threads_per_block = 128;
  int near_greater_power_of_two = GetNearGreaterPowerOfTwoGrad(dim_size);
  int kernel_warp_size =
      (near_greater_power_of_two < 32) ? near_greater_power_of_two : 32;
  int warps_per_block = (threads_per_block / kernel_warp_size);
  int blocks = (outer_size + warps_per_block - 1) / warps_per_block;
  dim3 threads(kernel_warp_size, warps_per_block, 1);

  switch (near_greater_power_of_two) {
    LAUNCH_WARP_BACKWARD_COMPUTE(1);     // dim_size: 1
    LAUNCH_WARP_BACKWARD_COMPUTE(2);     // dim_size: 2
    LAUNCH_WARP_BACKWARD_COMPUTE(4);     // dim_size: 3~4
    LAUNCH_WARP_BACKWARD_COMPUTE(8);     // dim_size: 5~8
    LAUNCH_WARP_BACKWARD_COMPUTE(16);    // dim_size: 9~16
    LAUNCH_WARP_BACKWARD_COMPUTE(32);    // dim_size: 17~32
    LAUNCH_WARP_BACKWARD_COMPUTE(64);    // dim_size: 33~64
    LAUNCH_WARP_BACKWARD_COMPUTE(128);   // dim_size: 65~128
    LAUNCH_WARP_BACKWARD_COMPUTE(256);   // dim_size: 129~256
    LAUNCH_WARP_BACKWARD_COMPUTE(512);   // dim_size: 257~512
    LAUNCH_WARP_BACKWARD_COMPUTE(1024);  // dim_size: 513~1024

    default:
      break;
  }
}

template <typename T, typename Context>
void LogSoftmaxGradKernel(const Context &dev_ctx,
                          const DenseTensor &x,
                          const DenseTensor &out,
                          const DenseTensor &out_grad,
                          int axis,
                          DenseTensor *x_grad) {
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;
  const auto *out_data = out.data<T>();
  const auto *out_grad_data = out_grad.data<T>();
  auto *x_grad_data = dev_ctx.template Alloc<T>(x_grad);

  const int rank = out.dims().size();
  const int canonical_axis = CanonicalAxis(axis, rank);

  int dim_size = out.dims()[canonical_axis];
  int inner_size = 1;
  for (int i = canonical_axis + 1; i < out.dims().size(); ++i) {
    inner_size *= out.dims()[i];
  }
  int outer_size = SizeToAxis(canonical_axis, out.dims());
  gpuStream_t stream = dev_ctx.stream();

  if (inner_size == 1 && dim_size <= 1024 && dim_size * sizeof(T) <= 4096) {
    LaunchSoftmaxBackwardForLastAxis<T, MPDType>(
        x_grad_data, out_grad_data, out_data, dim_size, outer_size, stream);
  } else {
    LogSoftmaxGradFunctor<Context, T>()(
        dev_ctx, &out, &out_grad, x_grad, canonical_axis);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(log_softmax_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::LogSoftmaxGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
