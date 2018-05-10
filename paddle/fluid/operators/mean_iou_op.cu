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

#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/mean_iou_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <typename T, int block>
__global__ void MeanIoUCUDAKernel(const int64_t num_classes,
                                  const int64_t count, const T* predictions,
                                  const T* labels, float* out_cm,
                                  float* matrixes_data) {
  int64_t out_cm_size = num_classes * num_classes;
  for (int64_t i = threadIdx.x; i < count; i += block) {
    int64_t index =
        threadIdx.x * out_cm_size + predictions[i] * num_classes + labels[i];
    matrixes_data[index] += 1.0f;
  }
  __syncthreads();
  float result;
  for (int64_t i = threadIdx.x; i < out_cm_size; i += block) {
    result = 0.0f;
    for (int64_t j = i; j < block * out_cm_size; j += out_cm_size) {
      result += matrixes_data[j];
    }
    out_cm[i] += result;
  }
}

template <typename T>
__global__ void ReplaceCUDAKernel(const int64_t n, T* data, T target, T value) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    if (data[i] == target) {
      data[i] = value;
    }
  }
}

template <typename T>
__global__ void DiagonalCUDAKernel(int64_t n, T* matrix, T* diagonal) {
  CUDA_1D_KERNEL_LOOP(i, n) { diagonal[i] = matrix[i * (n + 1)]; }
}

template <typename T>
struct GenConfusionMatrix<paddle::platform::CUDADeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx,
                  const int64_t num_classes, const int64_t count,
                  const T* predictions, const T* labels, float* out_cm) {
    int block = PADDLE_CUDA_NUM_THREADS;
    Tensor matrixes;
    float* matrixes_data = matrixes.mutable_data<float>(
        {block, num_classes, num_classes}, ctx.GetPlace());
    math::SetConstant<paddle::platform::CUDADeviceContext, float> constant;
    constant(ctx.template device_context<paddle::platform::CUDADeviceContext>(),
             &matrixes, 0.0f);

    MeanIoUCUDAKernel<T, PADDLE_CUDA_NUM_THREADS><<<
        1, block, 0, ctx.cuda_device_context().stream()>>>(
        num_classes, count, predictions, labels, out_cm, matrixes_data);
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

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    mean_iou, ops::MeanIoUKernel<paddle::platform::CUDADeviceContext, int>,
    ops::MeanIoUKernel<paddle::platform::CUDADeviceContext, int64_t>);
