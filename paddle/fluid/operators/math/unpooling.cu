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
#include "paddle/fluid/operators/math/unpooling.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {
namespace math {
template <typename T>
__global__ void KernelUnpool2dMax(const int nthreads, const T* input_data,
                                  const int* indices_data,
                                  const int input_height, const int input_width,
                                  const int channels, T* output_data,
                                  const int output_height,
                                  const int output_width) {
  int in_n_stride = input_height * input_width * channels;
  int in_c_stride = input_height * input_width;
  int out_n_stride = output_height * output_width * channels;
  int out_c_stride = output_height * output_width;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (int i = index; i < nthreads; i += offset) {
    int bidx = i / in_n_stride;
    int boffset = i % in_n_stride;
    int cidx = boffset / in_c_stride;
    int out_offset = bidx * out_n_stride + cidx * out_c_stride;
    int out_index = indices_data[i];
    PADDLE_ASSERT(out_index < out_c_stride);
    output_data[out_offset + out_index] = input_data[i];
  }
}
template <typename T>
__global__ void KernelUnpool2dMaxGrad(
    const int nthreads, const T* input_data, const int* indices_data,
    const int input_height, const int input_width, const int channels,
    const T* output_data, const T* output_grad, const int output_height,
    const int output_width, T* input_grad) {
  int in_n_stride = input_height * input_width * channels;
  int in_c_stride = input_height * input_width;
  int out_n_stride = output_height * output_width * channels;
  int out_c_stride = output_height * output_width;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (int i = index; i < nthreads; i += offset) {
    int bidx = i / in_n_stride;
    int boffset = i % in_n_stride;
    int cidx = boffset / in_c_stride;
    int out_offset = bidx * out_n_stride + cidx * out_c_stride;
    int out_index = indices_data[i];
    PADDLE_ASSERT(out_index < out_c_stride);
    input_grad[i] = output_grad[out_offset + out_index];
  }
}
/*
 * All tensors are in NCHW format.
 */
template <typename T>
class Unpool2dMaxFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& indices, framework::Tensor* output) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output->dims()[1];
    const int output_height = output->dims()[2];
    const int output_width = output->dims()[3];
    const T* input_data = input.data<T>();
    const int* indices_data = indices.data<int>();
    T* output_data = output->mutable_data<T>(context.GetPlace());
    int threads = 1024;
    int grid = (input.numel() + threads - 1) / threads;
    hipLaunchKernelGGL((KernelUnpool2dMax<
        T>), dim3(grid), dim3(threads), 0,
                 context.stream(), input.numel(), input_data, indices_data,
                              input_height, input_width, output_channels,
                              output_data, output_height, output_width);
  }
};
/*
 * All tensors are in NCHW format.
 */
template <typename T>
class Unpool2dMaxGradFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& indices,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad,
                  framework::Tensor* input_grad) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output.dims()[1];
    const int output_height = output.dims()[2];
    const int output_width = output.dims()[3];
    const T* input_data = input.data<T>();
    const int* indices_data = indices.data<int>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());
    int threads = 1024;
    int grid = (input.numel() + threads - 1) / threads;
    hipLaunchKernelGGL((KernelUnpool2dMaxGrad<
        T>), dim3(grid), dim3(threads), 0,
                 context.stream(), input.numel(), input_data, indices_data,
                              input_height, input_width, output_channels,
                              output_data, output_grad_data, output_height,
                              output_width, input_grad_data);
  }
};
template class Unpool2dMaxGradFunctor<platform::CUDADeviceContext, float>;
template class Unpool2dMaxGradFunctor<platform::CUDADeviceContext, double>;
template class Unpool2dMaxFunctor<platform::CUDADeviceContext, float>;
template class Unpool2dMaxFunctor<platform::CUDADeviceContext, double>;
}  // namespace math
}  // namespace operators
}  // namespace paddle
