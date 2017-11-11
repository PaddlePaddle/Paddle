/* Copyright (c) 2016 paddlepaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/math/maxouting.h"
#include "paddle/platform/cuda_helper.h"

namespace paddle {
namespace operators {
namespace math {

template <typename MaxOutProcess, typename T>
__global__ void KernelMaxOut(const int nthreads, const T* input_data,
                             T* output_data, const int channels,
                             const int input_height, const int input_width,
                             int groups, MaxOutProcess maxout_process) {
  int size = input_height * input_width * channels / groups;
  int featLen = input_height * input_width;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
              index += blockDim.x * gridDim.x) {
    int batch_idx = index / size;
    int i = index % size;
    int channel_idx = i / featLen;
    int feat_idx = i % featLen;
    int data_idx =
      (batch_idx * size + channel_idx * featLen) * groups + feat_idx;
    T ele = maxout_process.initial();
    for (int g = 0; g < groups; g++) {
      maxout_process.compute(ele, input_data[data_idx + g * featLen]);
    }
    maxout_process.finalize(ele, (static_cast<T>(groups)));
    output_data[index] = ele;
  }
}
template <typename T>
__global__ void KernelMaxoutGrad(
    const int nthreads, const T* input_data, const T* output_data,
    const T* output_grad, T* input_grad, const int channels,
    const int input_height, const int input_width, int groups) {
    int size = input_height * input_width * channels / groups;
    int featLen = input_height * input_width;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
         index += blockDim.x * gridDim.x) {
      int batch_idx = index / size;
      int i = index % size;
      int channel_idx = i / featLen;
      int feat_idx = i % featLen;
      int data_idx =
        (batch_idx * size + channel_idx * featLen) * groups + feat_idx;
      int maxIndex = -1;
      bool stop = false;
      for (int g = 0; g < groups && !stop; g++) {
        if (input_data[data_idx + g * featLen] == output_data[index]) {
          maxIndex = data_idx + g * featLen;
          stop = true;
        }
      }
      if (maxIndex != -1) {
        // atomic add
        platform::CudaAtomicAdd(input_grad + maxIndex, output_grad[index]);
      }
    }
}
/*
 * All tensors are in NCHW format.
 * Ksize, strides, paddings are two elements. These two elements represent
 * height and width, respectively.
 */
template <typename MaxOutProcess, typename T>
class MaxOutFunctor<platform::GPUPlace, MaxOutProcess, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& output,
                  int groups, int num_channels,
                  MaxOutProcess maxout_process) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = num_channels / groups;
    const int output_height = output.dims()[2];
    const int output_width = output.dims()[3];

    const T* input_data = input.data<T>();
    T* output_data = output.mutable_data<T>(context.GetPlace());

    int nthreads = batch_size * output_channels * output_height * output_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelMaxOut<
        MaxOutProcess,
        T><<<grid, threads, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(context)
                 .stream()>>>(nthreads, input_data, output_data, input_channels,
                              input_height, input_width, groups,
                              maxout_process);
  }
};
/*
 * All tensors are in NCHW format.
 * Ksize, strides, paddings are two elements. These two elements represent
 * height and width, respectively.
 */
template <typename T>
class MaxOutGradFunctor<platform::GPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& input_grad,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad,
                  int groups, int num_channels) {
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
    T* input_grad_data = input_grad.mutable_data<T>(context.GetPlace());

    int nthreads = batch_size * output_channels * output_height * output_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelMaxoutGrad<
        T><<<grid, threads, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(context)
                 .stream()>>>(
        nthreads, input_data, output_data, output_grad_data, input_grad_data,
        input_channels, input_height, input_width, groups);
  }
};

template class MaxOutGradFunctor<platform::GPUPlace, float>;
template class MaxOutGradFunctor<platform::GPUPlace, double>;

template class MaxOutFunctor<platform::GPUPlace,
                             paddle::operators::math::MaxOut<float>, float>;
template class MaxOutFunctor<platform::GPUPlace,
                             paddle::operators::math::MaxOut<double>, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
