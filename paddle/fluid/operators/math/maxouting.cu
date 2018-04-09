/* Copyright (c) 2016 paddlepaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "hip/hip_runtime.h"
#include "paddle/fluid/operators/math/maxouting.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
__global__ void KernelMaxOut(const int nthreads, const T* input_data,
                             const int channels, const int input_height,
                             const int input_width, int groups,
                             T* output_data) {
  const int size = input_height * input_width * channels / groups;
  const int feat_len = input_height * input_width;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (int i = index; i < nthreads; i += offset) {
    int batch_idx = i / size;
    int batch_offset = i % size;
    int channel_idx = batch_offset / feat_len;
    int feat_idx = batch_offset % feat_len;
    int data_idx =
        (batch_idx * size + channel_idx * feat_len) * groups + feat_idx;
    T ele = static_cast<T>(-FLT_MAX);
    for (int g = 0; g < groups; ++g) {
      T x = input_data[data_idx + g * feat_len];
      ele = ele > x ? ele : x;
    }
    output_data[i] = ele;
  }
}
template <typename T>
__global__ void KernelMaxoutGrad(const int nthreads, const T* input_data,
                                 const T* output_data, const T* output_grad,
                                 T* input_grad, const int channels,
                                 const int input_height, const int input_width,
                                 int groups) {
  const int size = input_height * input_width * channels / groups;
  const int feat_len = input_height * input_width;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (int i = index; i < nthreads; i += offset) {
    int batch_idx = i / size;
    int batch_offset = i % size;
    int channel_idx = batch_offset / feat_len;
    int feat_idx = batch_offset % feat_len;
    int data_idx =
        (batch_idx * size + channel_idx * feat_len) * groups + feat_idx;
    int max_index = -1;
    bool continue_match = true;
    for (int g = 0; g < groups && continue_match; ++g) {
      if (input_data[data_idx + g * feat_len] == output_data[i]) {
        max_index = data_idx + g * feat_len;
        continue_match = false;
        break;
      }
    }
    if (max_index != -1) {
      input_grad[max_index] += output_grad[index];
    }
  }
}
/*
 * All tensors are in NCHW format.
 */
template <typename T>
class MaxOutFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input, framework::Tensor* output,
                  int groups) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output->dims()[1];
    const int output_height = output->dims()[2];
    const int output_width = output->dims()[3];

    const T* input_data = input.data<T>();
    T* output_data = output->mutable_data<T>(context.GetPlace());
    int nthreads = output->numel();
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    hipLaunchKernelGGL((KernelMaxOut<
        T>), dim3(grid), dim3(threads), 0,
                 context.stream(), nthreads, input_data, input_channels,
                              input_height, input_width, groups, output_data);
  }
};
/*
 * All tensors are in NCHW format.
 */
template <typename T>
class MaxOutGradFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input, framework::Tensor* input_grad,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad, int groups) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output.dims()[1];
    const int output_height = output.dims()[2];
    const int output_width = output.dims()[3];

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());
    int nthreads = output.numel();
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    hipLaunchKernelGGL((KernelMaxoutGrad<
        T>), dim3(grid), dim3(threads), 0,
                 context.stream(), nthreads, input_data, output_data,
                              output_grad_data, input_grad_data, input_channels,
                              input_height, input_width, groups);
  }
};

template class MaxOutGradFunctor<platform::CUDADeviceContext, float>;
template class MaxOutGradFunctor<platform::CUDADeviceContext, double>;

template class MaxOutFunctor<platform::CUDADeviceContext, float>;
template class MaxOutFunctor<platform::CUDADeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
