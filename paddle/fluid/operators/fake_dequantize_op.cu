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

#include "paddle/fluid/operators/fake_dequantize_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void KeDequantize(const T* in, const T* scale, T max_range, int num,
                             T* out) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num) {
    out[idx] = in[idx] * scale[0] / max_range;
  }
}

template <typename T>
struct DequantizeFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& dev_ctx,
                  const framework::Tensor* in, const framework::Tensor* scale,
                  T max_range, framework::Tensor* out) {
    const T* in_data = in->data<T>();
    const T* scale_factor = scale->data<T>();
    T* out_data = out->mutable_data<T>(dev_ctx.GetPlace());

    int num = in->numel();
    int block = 512;
    int grid = (num + block - 1) / block;

    KeDequantize<T><<<grid, block, 0, dev_ctx.stream()>>>(
        in_data, scale_factor, max_range, num, out_data);
  }
};

template <typename T>
__global__ void DequantizeOneScale(const T* in, const T* scale, T max_range,
                                   int num, int channel, T* out) {
  int tid = threadIdx.x;
  int channel_size = num / channel;
  const T* in_c = in + blockIdx.x * channel_size;
  T* out_c = out + blockIdx.x * channel_size;
  for (int i = tid; i < channel_size; i += blockDim.x) {
    out_c[i] = in_c[i] * scale[blockIdx.x] / max_range;
  }
}

template <typename T>
__global__ void DequantizeTwoScale(const T* in, const T* scale_one,
                                   const T* scale_two, T max_range, int num,
                                   int batch_size, int channel, T* out) {
  int tid = threadIdx.x;
  int channel_size = num / (batch_size * channel);
  int scale_index = blockIdx.x % channel;
  const T* in_c = in + blockIdx.x * channel_size;
  T* out_c = out + blockIdx.x * channel_size;
  for (int i = tid; i < channel_size; i += blockDim.x) {
    out_c[i] = in_c[i] * scale_one[scale_index] * scale_two[0] / max_range;
  }
}

template <typename T>
struct ChannelDequantizeFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& dev_ctx,
                  const framework::Tensor* in, const framework::Tensor** scales,
                  const int scale_num, T max_range, framework::Tensor* out) {
    const T* in_data = in->data<T>();
    T* out_data = out->mutable_data<T>(dev_ctx.GetPlace());
    if (scale_num == 1) {
      int num = in->numel();
      int channel = in->dims()[0];
      const T* scale_factor = scales[0]->data<T>();
      int block = 1024;
      int grid = channel;
      DequantizeOneScale<T><<<grid, block, 0, dev_ctx.stream()>>>(
          in_data, scale_factor, max_range, num, channel, out_data);
    } else if (scale_num == 2) {
      int num = in->numel();
      int batch_size = in->dims()[0];
      int channel = in->dims()[1];
      const T* scale_one = scales[0]->data<T>();
      const T* scale_two = scales[1]->data<T>();
      int block = 1024;
      int grid = batch_size * channel;
      DequantizeTwoScale<T><<<grid, block, 0, dev_ctx.stream()>>>(
          in_data, scale_one, scale_two, max_range, num, batch_size, channel,
          out_data);
    }
  }
};

template struct DequantizeFunctor<platform::CUDADeviceContext, float>;
template struct DequantizeFunctor<platform::CUDADeviceContext, double>;
template struct ChannelDequantizeFunctor<platform::CUDADeviceContext, float>;
template struct ChannelDequantizeFunctor<platform::CUDADeviceContext, double>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(fake_dequantize_max_abs,
                        ops::FakeDequantizeMaxAbsKernel<CUDA, float>,
                        ops::FakeDequantizeMaxAbsKernel<CUDA, double>);
REGISTER_OP_CUDA_KERNEL(
    fake_channel_wise_dequantize_max_abs,
    ops::FakeChannelWiseDequantizeMaxAbsKernel<CUDA, float>,
    ops::FakeChannelWiseDequantizeMaxAbsKernel<CUDA, double>);
