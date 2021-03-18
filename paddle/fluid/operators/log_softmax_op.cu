// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda_runtime.h>
#include <limits>
#include "paddle/fluid/operators/log_softmax_op.h"
#include "paddle/fluid/platform/cuda_device_function.h"

namespace paddle {
namespace operators {

#define WARP_SIZE 32

#define LAUNCH_SOFTMAX_WARP_FORWARD(L2E)                                   \
  case L2E:                                                                \
    WarpLogSoftmaxForward<T, double, L2E><<<blocks, threads, 0>>>(         \
        dst, src, batch_count, softmax_elements_stride, softmax_elements); \
    break;

int LogTwoCeil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

template <typename T, int NumBatch, int KernelWarpSize>
__device__ __forceinline__ void ReduceSumForWarpBatch(T* sum) {
#pragma unroll
  for (int offset = KernelWarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < NumBatch; ++i) {
      T sum_val = platform::CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = sum[i] + sum_val;
    }
  }
}

template <typename T, int NumBatch, int KernelWarpSize>
__device__ __forceinline__ void ReduceMaxForWarpBatch(T* sum) {
#pragma unroll
  for (int offset = KernelWarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < NumBatch; ++i) {
      T max_val = platform::CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = max(sum[i], max_val);
    }
  }
}

template <typename T, typename AccT, int log2_elements>
__global__ void WarpLogSoftmaxForward(T* dst, const T* src, int batch_size,
                                      int stride, int element_count) {
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int kernel_warp_size =
      (next_power_of_two < WARP_SIZE) ? next_power_of_two : WARP_SIZE;
  constexpr int warp_iterations = next_power_of_two / kernel_warp_size;
  constexpr int num_batch = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * num_batch;
  int local_batches = batch_size - first_batch;
  if (local_batches > num_batch) local_batches = num_batch;

  int local_idx = threadIdx.x;
  src += first_batch * stride + local_idx;
  dst += first_batch * stride + local_idx;

  // 1.load data from global memory
  AccT elements[num_batch][warp_iterations];
  int idx = threadIdx.x + blockDim.x * threadIdx.y;

  for (int i = 0; i < num_batch; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < warp_iterations; ++it) {
      int element_index = local_idx + it * kernel_warp_size;
      if (element_index < batch_element_count) {
        elements[i][it] =
            static_cast<double>(src[i * element_count + it * kernel_warp_size]);
      } else {
        elements[i][it] = -std::numeric_limits<AccT>::infinity();
      }
    }
  }

  // 2.compute max_value
  AccT max_value[num_batch];
#pragma unroll
  for (int i = 0; i < num_batch; ++i) {
    max_value[i] = elements[i][0];
#pragma unroll
    for (int it = 1; it < warp_iterations; ++it) {
      max_value[i] =
          (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }
  ReduceMaxForWarpBatch<AccT, num_batch, kernel_warp_size>(max_value);

  AccT sum[num_batch]{0.0f};
#pragma unroll
  for (int i = 0; i < num_batch; ++i) {
#pragma unroll
    for (int it = 0; it < warp_iterations; ++it) {
      sum[i] += std::exp(elements[i][it] - max_value[i]);
    }
  }
  ReduceSumForWarpBatch<AccT, num_batch, kernel_warp_size>(sum);

// 3.store result
#pragma unroll
  for (int i = 0; i < num_batch; ++i) {
    if (i >= local_batches) break;
    sum[i] = std::log(sum[i]);
#pragma unroll
    for (int it = 0; it < warp_iterations; ++it) {
      int element_index = local_idx + it * kernel_warp_size;
      if (element_index < element_count) {
        dst[i * element_count + it * kernel_warp_size] =
            elements[i][it] - max_value[i] - sum[i];
      } else {
        break;
      }
    }
  }
}

template <typename T>
void LaunchSoftmaxForwardForLastAxis(T* dst, const T* src, int softmax_elements,
                                     int softmax_elements_stride,
                                     int batch_count) {
  int log2_elements = LogTwoCeil(softmax_elements);
  const int next_power_of_two = 1 << log2_elements;
  int warp_size =
      (next_power_of_two < WARP_SIZE) ? next_power_of_two : WARP_SIZE;
  int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

  // use 128 threads per block to maximimize gpu utilization
  constexpr int threads_per_block = 128;
  int warps_per_block = (threads_per_block / warp_size);
  int batches_per_block = warps_per_block * batches_per_warp;
  int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
  dim3 threads(warp_size, warps_per_block, 1);

  switch (log2_elements) {
    LAUNCH_SOFTMAX_WARP_FORWARD(0);   // 1
    LAUNCH_SOFTMAX_WARP_FORWARD(1);   // 2
    LAUNCH_SOFTMAX_WARP_FORWARD(2);   // 4
    LAUNCH_SOFTMAX_WARP_FORWARD(3);   // 8
    LAUNCH_SOFTMAX_WARP_FORWARD(4);   // 16
    LAUNCH_SOFTMAX_WARP_FORWARD(5);   // 32
    LAUNCH_SOFTMAX_WARP_FORWARD(6);   // 64
    LAUNCH_SOFTMAX_WARP_FORWARD(7);   // 128
    LAUNCH_SOFTMAX_WARP_FORWARD(8);   // 256
    LAUNCH_SOFTMAX_WARP_FORWARD(9);   // 512
    LAUNCH_SOFTMAX_WARP_FORWARD(10);  // 1024
    default:
      break;
  }
}

template <typename T>
class LogSoftmaxKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* X = context.Input<framework::Tensor>("X");
    auto* Out = context.Output<framework::Tensor>("Out");
    const auto* input_data = X->data<T>();
    auto* output_data = Out->mutable_data<T>(context.GetPlace());

    PADDLE_ENFORCE_GT(X->numel(), 0, platform::errors::InvalidArgument(
                                         "Expected number of elements > 0. But "
                                         "received number of elements is %d.",
                                         X->numel()));
    const int rank = X->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);

    int dim_size = X->dims()[axis];
    int inner_size = 1;
    for (int i = axis + 1; i < X->dims().size(); i++)
      inner_size *= X->dims()[i];
    int outer_size = 1;
    outer_size = SizeToAxis(axis, X->dims());

    if (inner_size == 1 && dim_size <= 1024 && dim_size * sizeof(T) <= 4096) {
      // execute CUDA kernel
      LaunchSoftmaxForwardForLastAxis<T>(output_data, input_data, dim_size,
                                         dim_size, outer_size);
    } else {
      // execute Eigen kernel
      LogSoftmaxFunctor<platform::CUDADeviceContext, T>()(
          context.template device_context<platform::CUDADeviceContext>(), X,
          Out, axis);
    }
  }
};

}  // operators
}  // paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    log_softmax, ops::LogSoftmaxKernel<plat::CUDADeviceContext, float>,
    ops::LogSoftmaxKernel<plat::CUDADeviceContext, double>,
    ops::LogSoftmaxKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    log_softmax_grad, ops::LogSoftmaxGradKernel<plat::CUDADeviceContext, float>,
    ops::LogSoftmaxGradKernel<plat::CUDADeviceContext, double>,
    ops::LogSoftmaxGradKernel<plat::CUDADeviceContext, plat::float16>);
