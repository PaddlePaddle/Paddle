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

#include "paddle/operators/math/unpooling.h"
#include "paddle/platform/cuda_helper.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
__global__ void KernelUnpool2dMax(const int nthreads,
                                  const T* input_data,
                                  const int* indices_data,
                                  const int input_height,
                                  const int input_width,
                                  T* output_data,
                                  const int output_height,
                                  const int output_width) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (int i = index; i < nthreads; i += offset) {
    int out_offset =  i / (input_height * input_width) \
                      * output_height * output_width;
    int out_index = indices_data[i];
    PADDLE_ASSERT(out_index < (output_height * output_width));
    output_data[out_offset + out_index] = input_data[i];
  }
}
template <typename T>
__global__ void KernelUnpool2dMaxGrad(const int nthreads,
                                      const T* input_data,
                                      const int* indices_data,
                                      const int input_height,
                                      const int input_width,
                                      const T* output_data,
                                      const T* output_grad,
                                      const int output_height,
                                      const int output_width,
                                      T* input_grad) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    for (int i = index; i < nthreads; i += offset) {
        int out_offset =  i / (input_height * input_width) \
                          * output_height * output_width;
        int out_index = indices_data[i];
        PADDLE_ASSERT(out_index < (output_height * output_width));
        input_grad[i] = output_grad[out_offset + out_index];
    }
}
/*
 * All tensors are in NCHW format.
 */
template <typename T>
class Unpool2dMaxFunctor<platform::GPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& indices,
                  framework::Tensor * output) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output->dims()[1];
    const int output_height = output->dims()[2];
    const int output_width = output->dims()[3];
    const T* input_data = input.data<T>();
    const int* indices_data = indices.data<int>();
    T* output_data = output->mutable_data<T>(context.GetPlace());

    int nthreads =  output->numel();
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelUnpool2dMax<
        T><<<grid, threads, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(context)
                 .stream()>>>(nthreads, input_data, indices_data,
                              input_height, input_width,
                              output_data, output_height, output_width);
  }
};
/*
 * All tensors are in NCHW format.
 */
template <typename T>
class Unpool2dMaxGradFunctor<platform::GPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& indices,
                  framework::Tensor * input_grad,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad) {
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
    int nthreads =  output.numel();
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelUnpool2dMaxGrad<
        T><<<grid, threads, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(context)
                 .stream()>>>(
                              nthreads, input_data, indices_data,
                              input_height, input_width,
                              output_data, output_grad_data,
                              output_height, output_width,
                              input_grad_data);
  }
};

template class Unpool2dMaxGradFunctor<platform::GPUPlace, float>;
template class Unpool2dMaxGradFunctor<platform::GPUPlace, double>;

template class Unpool2dMaxFunctor<platform::GPUPlace, float>;
template class Unpool2dMaxFunctor<platform::GPUPlace, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
