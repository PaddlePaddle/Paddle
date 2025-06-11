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

#include "paddle/phi/kernels/funcs/maxouting.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"

namespace phi {
namespace funcs {

/*
 * All tensors are in NCHW or NHWC format.
 */
template <typename T>
__global__ void KernelMaxOut(const int nthreads,
                             const T* input_data,
                             const int channels,
                             const int input_height,
                             const int input_width,
                             const int groups,
                             const int axis,
                             T* output_data) {
  const int64_t size =
      static_cast<int64_t>(input_height) * input_width * channels / groups;
  const int64_t feat_len = static_cast<int64_t>(input_height) * input_width;
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t offset = blockDim.x * gridDim.x;
  for (int64_t i = index; i < nthreads; i += offset) {
    int64_t batch_idx = i / size;
    int64_t batch_offset = i % size;
    int64_t channel_idx, feat_idx, data_idx;
    if (axis == 1) {
      channel_idx = batch_offset / feat_len;
      feat_idx = batch_offset % feat_len;
      data_idx =
          (batch_idx * size + channel_idx * feat_len) * groups + feat_idx;
    } else {
      channel_idx = batch_offset % channels;
      feat_idx = batch_offset / channels;
      data_idx =
          (batch_idx * size + feat_idx * channels + channel_idx) * groups;
    }
    T ele = static_cast<T>(-FLT_MAX);
    for (int g = 0; g < groups; ++g) {
      int64_t idx_offset = (axis == 1 ? g * feat_len : g);
      T x = input_data[data_idx + idx_offset];
      ele = ele > x ? ele : x;
    }
    output_data[i] = ele;
  }
}

/*
 * All tensors are in NCHW or NHWC format.
 */
template <typename T>
__global__ void KernelMaxoutGrad(const int nthreads,
                                 const T* input_data,
                                 const T* output_data,
                                 const T* output_grad,
                                 T* input_grad,
                                 const int channels,
                                 const int input_height,
                                 const int input_width,
                                 const int groups,
                                 const int axis) {
  const int64_t size =
      static_cast<int64_t>(input_height) * input_width * channels / groups;
  const int64_t feat_len = static_cast<int64_t>(input_height) * input_width;
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t offset = blockDim.x * gridDim.x;
  for (int64_t i = index; i < nthreads; i += offset) {
    int64_t batch_idx = i / size;
    int64_t batch_offset = i % size;
    int64_t channel_idx, feat_idx, data_idx;
    if (axis == 1) {
      channel_idx = batch_offset / feat_len;
      feat_idx = batch_offset % feat_len;
      data_idx =
          (batch_idx * size + channel_idx * feat_len) * groups + feat_idx;
    } else {
      channel_idx = batch_offset % channels;
      feat_idx = batch_offset / channels;
      data_idx =
          (batch_idx * size + feat_idx * channels + channel_idx) * groups;
    }
    int64_t max_index = -1;
    bool continue_match = true;
    for (int g = 0; g < groups && continue_match; ++g) {
      int64_t idx_offset = (axis == 1 ? g * feat_len : g);
      if (input_data[data_idx + idx_offset] == output_data[i]) {
        max_index = data_idx + idx_offset;
        continue_match = false;
      }
    }
    if (max_index != -1) {
      input_grad[max_index] += output_grad[index];
    }
  }
}

template <typename DeviceContext, typename T>
void MaxOutFunctor<DeviceContext, T>::operator()(const DeviceContext& context,
                                                 const phi::DenseTensor& input,
                                                 phi::DenseTensor* output,
                                                 const int groups,
                                                 const int axis) {
  const int batch_size = input.dims()[0];
  const int input_channels = input.dims()[axis];
  const int input_height = (axis == 1 ? input.dims()[2] : input.dims()[1]);
  const int input_width = (axis == 1 ? input.dims()[3] : input.dims()[2]);

  const T* input_data = input.data<T>();
  T* output_data = context.template Alloc<T>(output);
  int64_t nthreads = static_cast<int64_t>(output->numel());
  int blocks = static_cast<int>((nthreads + 1024 - 1) / 1024);
  dim3 threads(1024, 1);
  dim3 grid(blocks, 1);

  KernelMaxOut<T><<<grid, threads, 0, context.stream()>>>(nthreads,
                                                          input_data,
                                                          input_channels,
                                                          input_height,
                                                          input_width,
                                                          groups,
                                                          axis,
                                                          output_data);
}

template <typename DeviceContext, typename T>
void MaxOutGradFunctor<DeviceContext, T>::operator()(
    const DeviceContext& context,
    const phi::DenseTensor& input,
    phi::DenseTensor* input_grad,
    const phi::DenseTensor& output,
    const phi::DenseTensor& output_grad,
    const int groups,
    const int axis) {
  const int input_channels = input.dims()[axis];
  const int input_height = (axis == 1 ? input.dims()[2] : input.dims()[1]);
  const int input_width = (axis == 1 ? input.dims()[3] : input.dims()[2]);

  const T* input_data = input.data<T>();
  const T* output_data = output.data<T>();
  const T* output_grad_data = output_grad.data<T>();
  T* input_grad_data = context.template Alloc<T>(input_grad);
  int64_t nthreads = static_cast<int64_t>(output.numel());
  int blocks = static_cast<int>((nthreads + 1024 - 1) / 1024);
  dim3 threads(1024, 1);
  dim3 grid(blocks, 1);

  KernelMaxoutGrad<T><<<grid, threads, 0, context.stream()>>>(nthreads,
                                                              input_data,
                                                              output_data,
                                                              output_grad_data,
                                                              input_grad_data,
                                                              input_channels,
                                                              input_height,
                                                              input_width,
                                                              groups,
                                                              axis);
}

template class MaxOutGradFunctor<phi::GPUContext, float>;
template class MaxOutGradFunctor<phi::GPUContext, phi::dtype::float16>;
template class MaxOutGradFunctor<phi::GPUContext, double>;

template class MaxOutFunctor<phi::GPUContext, float>;
template class MaxOutFunctor<phi::GPUContext, phi::dtype::float16>;
template class MaxOutFunctor<phi::GPUContext, double>;

}  // namespace funcs
}  // namespace phi
