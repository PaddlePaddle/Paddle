/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/mean_iou_op.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {

template <typename T>
struct GenConfusionMatrix<paddle::platform::CUDADeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx,
                  const int64_t num_classes, const int64_t count,
                  const T* predictions, const T* labels, float* out_cm) {
    int block = 512;
    int grid = (count + block - 1) / block;
    MeanIoUCudaKernel<
        T><<<grid, block, 0, ctx.cuda_device_context().stream()>>>(
        num_classes, count, predictions, labels, out_cm);
  }
};

template <typename T>
struct Replace<paddle::platform::CUDADeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx, const int64_t n,
                  T* data, T target, T value) {
    int block = 512;
    int grid = (n + block - 1) / block;
    ReplaceCUDAKernel<
        T><<<grid, block, 0, ctx.cuda_device_context().stream()>>>(
        n, data, target, value);
  }
};

template <typename T>
struct Diagonal<paddle::platform::CUDADeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx, int64_t n, T* matrix,
                  T* diagonal) {
    int block = 512;
    int grid = (n + block - 1) / block;
    DiagonalCUDAKernel<
        T><<<grid, block, 0, ctx.cuda_device_context().stream()>>>(n, matrix,
                                                                   diagonal);
  }
};

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <typename T>
__global__ void MeanIoUCudaKernel(const int64_t num_classes,
                                  const int64_t count, const T* predictions,
                                  const T* labels, float* out_cm) {
  CUDA_1D_KERNEL_LOOP(i, count) {
    int64_t index = predictions[i] * num_classes + labels[i];
    out_cm[index] += 1.0f;
  }
}

template <typename T>
__global__ void ReplaceCudaKernel(const int64_t n, T* data, T target, T value) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    if (data[i] == target) {
      data[i] = value;
    }
  }
}

template <typename T>
__global__ DiagonalCUDAKernel(int64_t n, T* matrix, T* diagonal) {
  CUDA_1D_KERNEL_LOOP(i, n) { diagonal[i] = matrix[i * (n + 1)]; }
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    mean_iou, ops::MeanIoUKernel<paddle::platform::CUDADeviceContext, int>,
    ops::MeanIoUKernel<paddle::platform::CUDADeviceContext, int64_t>);
