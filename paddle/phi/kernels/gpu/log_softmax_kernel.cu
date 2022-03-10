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
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/functors.h"
#include "paddle/phi/kernels/funcs/log_softmax_functor.h"
#include "paddle/phi/kernels/gpu/log_softmax_funcs.h"
#include "paddle/phi/kernels/log_softmax_kernel.h"

namespace phi {

int GetNearGreaterPowerOfTwo(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) {
    ++log2_value;
  }
  return 1 << log2_value;
}

#define LAUNCH_WARP_FORWAR_COMPUTE(near_greater_power_of_two)       \
  case near_greater_power_of_two:                                   \
    ComputeLogSoftmaxForwardInWarp<                                 \
        T,                                                          \
        AccT,                                                       \
        near_greater_power_of_two><<<blocks, threads, 0, stream>>>( \
        dst, src, outer_size, dim_size);                            \
    break;

template <typename T, typename AccT>
void LaunchSoftmaxForwardForLastAxis(
    T *dst, const T *src, int dim_size, int outer_size, gpuStream_t stream) {
  int threads_per_block = 128;
  int near_greater_power_of_two = GetNearGreaterPowerOfTwo(dim_size);
  int kernel_warp_size =
      (near_greater_power_of_two < 32) ? near_greater_power_of_two : 32;
  int warps_per_block = (threads_per_block / kernel_warp_size);
  int blocks = (outer_size + warps_per_block - 1) / warps_per_block;
  dim3 threads(kernel_warp_size, warps_per_block, 1);

  switch (near_greater_power_of_two) {
    LAUNCH_WARP_FORWAR_COMPUTE(1);
    LAUNCH_WARP_FORWAR_COMPUTE(2);
    LAUNCH_WARP_FORWAR_COMPUTE(4);     // dim_size: 3~4
    LAUNCH_WARP_FORWAR_COMPUTE(8);     // dim_size: 5~8
    LAUNCH_WARP_FORWAR_COMPUTE(16);    // dim_size: 9~16
    LAUNCH_WARP_FORWAR_COMPUTE(32);    // dim_size: 17~32
    LAUNCH_WARP_FORWAR_COMPUTE(64);    // dim_size: 33~64
    LAUNCH_WARP_FORWAR_COMPUTE(128);   // dim_size 65~128
    LAUNCH_WARP_FORWAR_COMPUTE(256);   // dim_size 129~256
    LAUNCH_WARP_FORWAR_COMPUTE(512);   // dim_size 257~512
    LAUNCH_WARP_FORWAR_COMPUTE(1024);  // dim_size 513~1024

    default:
      break;
  }
}

template <typename T, typename AccT>
__global__ void LogSoftmaxForwardCUDAKernelNotLastAxis(
    T *output, const T *input, int outer_size, int dim_size, int inner_size) {
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<AccT *>(smem);

  const int outer_stride = inner_size * dim_size;
  const int dim_stride = inner_size;

  for (int x_id = blockIdx.x; x_id < outer_size; x_id += gridDim.x) {
    for (int y_id = blockIdx.y * blockDim.y + threadIdx.y; y_id < inner_size;
         y_id += blockDim.y * gridDim.y) {
      const int data_offset = x_id * outer_stride + y_id;
      // When blockDim.x==1, no block.x-reduction opetaions are needed.
      // And threadIdx.x is 0 all the time, so the for-loops below are literally
      // loops (No parallel executions). Loop all elements along axis and
      // calculate the Max, Sum and (input[id]-Max-log(Sum)) to get the final
      // log_softmax values along that axis.
      // 1. reduce max
      AccT max_value = -std::numeric_limits<AccT>::infinity();
      // For one thread, iterate all items it responsable for, and get
      // max_value.
      // If there are N threads, N max_value will be returned.
      for (int d = threadIdx.x; d < dim_size; d += blockDim.x) {
        const AccT value =
            static_cast<AccT>(input[data_offset + d * dim_stride]);
        max_value = phi::funcs::MaxFunctor<AccT>()(max_value, value);
      }
      // If there are more than 1 threads along block x, reduce all max_values
      // and get the global max_value, which is the max value along "axis".
      // If there is only one thread along block x, no need to reduce, as the
      // 'max_value' is the global max_value.
      if (blockDim.x > 1) {
        max_value = BlockReduceAlongDimX<AccT, phi::funcs::MaxFunctor>(
            sdata, max_value);
      }

      // 2. reduce sum
      AccT sum = 0;
      // Below is the same execution as '1. reduce max'
      for (int d = threadIdx.x; d < dim_size; d += blockDim.x) {
        sum += std::exp(static_cast<AccT>(input[data_offset + d * dim_stride]) -
                        max_value);
      }
      if (blockDim.x > 1) {
        sum = BlockReduceAlongDimX<AccT, phi::funcs::AddFunctor>(sdata, sum);
      }

      // 3. input-max-log_sum and write to output
      for (int d = threadIdx.x; d < dim_size; d += blockDim.x) {
        output[data_offset + d * dim_stride] = static_cast<T>(
            static_cast<AccT>(input[data_offset + d * dim_stride]) - max_value -
            std::log(sum));
      }
    }
  }
}

template <typename T, typename MPDType>
void LaunchLogSoftmaxForwardCUDAKernelNotLastAxis(T *output_data,
                                                  const T *input_data,
                                                  int outer_size,
                                                  int dim_size,
                                                  int inner_size,
                                                  int num_sm,
                                                  gpuStream_t stream) {
  int shared_mem;
  dim3 grid;
  dim3 block;

  ComputeLaunchConfigure<MPDType>(
      &LogSoftmaxForwardCUDAKernelNotLastAxis<T, MPDType>,
      outer_size,
      dim_size,
      inner_size,
      grid,
      block,
      shared_mem,
      num_sm);

  LogSoftmaxForwardCUDAKernelNotLastAxis<
      T,
      MPDType><<<grid, block, shared_mem, stream>>>(
      output_data, input_data, outer_size, dim_size, inner_size);
}

template <typename T, typename Context>
void LogSoftmaxKernel(const Context &dev_ctx,
                      const DenseTensor &x,
                      int axis,
                      DenseTensor *out) {
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;
  const auto *input_data = x.data<T>();
  auto *output_data = dev_ctx.template Alloc<T>(out);

  const int rank = x.dims().size();
  const int canonical_axis = CanonicalAxis(axis, rank);
  int dim_size = x.dims()[canonical_axis];
  int inner_size = 1;
  for (int i = canonical_axis + 1; i < x.dims().size(); ++i) {
    inner_size *= x.dims()[i];
  }
  int outer_size = SizeToAxis(canonical_axis, x.dims());
  gpuStream_t stream = dev_ctx.stream();
  int num_sm = dev_ctx.GetSMCount();

  if (inner_size == 1 && dim_size <= 1024 && dim_size * sizeof(T) <= 4096) {
    LaunchSoftmaxForwardForLastAxis<T, MPDType>(
        output_data, input_data, dim_size, outer_size, stream);
  } else {
    LaunchLogSoftmaxForwardCUDAKernelNotLastAxis<T, MPDType>(output_data,
                                                             input_data,
                                                             outer_size,
                                                             dim_size,
                                                             inner_size,
                                                             num_sm,
                                                             stream);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(log_softmax,
                   GPU,
                   ALL_LAYOUT,
                   phi::LogSoftmaxKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
