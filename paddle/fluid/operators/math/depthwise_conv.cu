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

#include <algorithm>
#include <vector>
#include "paddle/fluid/operators/math/depthwise_conv.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#define FULL_MASK 0xffffffff
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);
  return val;
}
__forceinline__ __device__ unsigned lane_id() {
  unsigned ret;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
  return ret;
}

__forceinline__ __device__ unsigned warp_id() {
  unsigned ret;
  asm volatile("mov.u32 %0, %warpid;" : "=r"(ret));
  return ret;
}

// A Cuda kernel to compute the depthwise convolution forward pass
// in NCHW format.
template <typename T, int c_filter_multiplier, int c_stride>
__global__ void KernelDepthwiseConv(
    const int nthreads, const T* const input_data, const T* const filter_data,
    const int batch_size, const int output_channels, const int output_height,
    const int output_width, const int input_channels, const int input_height,
    const int input_width, const int filter_multiplier, const int filter_height,
    const int filter_width, const int stride_height, const int stride_width,
    const int padding_height, const int padding_width, const int dilate_height,
    const int dilate_width, T* const output_data) {
  int real_stride_height = c_stride;
  int real_stride_width = c_stride;
  int real_filter_multiplier = c_filter_multiplier;
  if (c_stride == 0) {
    real_stride_height = stride_height;
    real_stride_width = stride_width;
  }
  if (c_filter_multiplier == 0) {
    real_stride_height = filter_multiplier;
  }

  for (int w_out = threadIdx.x; w_out < output_width; w_out += blockDim.x) {
    for (int h_out = threadIdx.y; h_out < output_height; h_out += blockDim.y) {
      const int batch = blockIdx.y;
      const int c_out = blockIdx.x;

      const int c_in = c_out / real_filter_multiplier;
      const T* weight = filter_data + c_out * filter_height * filter_width;
      T value = 0;
      const int h_in_start = -padding_height + h_out * real_stride_height;
      const int w_in_start = -padding_width + w_out * real_stride_width;
      const int h_in_end = h_in_start + filter_height * dilate_height;
      const int w_in_end = w_in_start + filter_width * dilate_width;

      const int in_offset =
          ((batch * input_channels + c_in) * input_height) * input_width;

      const int h_end = h_in_end < input_height ? h_in_end : input_height;
      const int w_end = w_in_end < input_width ? w_in_end : input_width;
      const int h_start = h_in_start > 0 ? h_in_start : 0;
      const int w_start = w_in_start > 0 ? w_in_start : 0;
      int weight_offset = 0;

      for (int h_in = h_in_start; h_in < h_in_end; h_in += dilate_height) {
        for (int w_in = w_in_start; w_in < w_in_end; w_in += dilate_width) {
          if (h_in >= h_start && h_in < h_end && w_in >= w_start &&
              w_in < w_end) {
            const int offset = in_offset + h_in * input_width + w_in;
            value += weight[weight_offset] * input_data[offset];
          }
          weight_offset++;
        }
      }
      int index =
          ((batch * gridDim.x + c_out) * output_height + h_out) * output_width +
          w_out;
      output_data[index] = value;
    }
  }
}

// CUDA kernel to compute the depthwise convolution backprop w.r.t input.
template <typename T, int c_filter_multiplier, int c_stride>
__global__ void KernelDepthwiseConvInputGrad(
    const int nthreads, const T* const output_grad_data,
    const T* const filter_data, const int batch_size, const int output_channels,
    const int output_height, const int output_width, const int input_channels,
    const int input_height, const int input_width, const int filter_multiplier,
    const int filter_height, const int filter_width, const int stride_height,
    const int stride_width, const int padding_height, const int padding_width,
    const int dilate_height, const int dilate_width, T* const input_grad_data) {
  int real_stride_height = c_stride;
  int real_stride_width = c_stride;
  int real_filter_multiplier = c_filter_multiplier;
  if (c_stride == 0) {
    real_stride_height = stride_height;
    real_stride_width = stride_width;
  }
  if (c_filter_multiplier == 0) {
    real_stride_height = filter_multiplier;
  }
  for (int w_in = threadIdx.x; w_in < input_width; w_in += blockDim.x) {
    for (int h_in = threadIdx.y; h_in < input_height; h_in += blockDim.y) {
      const int batch = blockIdx.y;
      const int c_in = blockIdx.x;

      const int c_out_start = c_in * real_filter_multiplier;

      int h_out_start =
          (h_in - filter_height + padding_height + real_stride_height) /
          real_stride_height;
      int _h_out_start = 0 > h_out_start ? 0 : h_out_start;

      int h_out_end = (h_in + padding_height) / real_stride_height;
      int _h_out_end =
          output_height - 1 < h_out_end ? output_height - 1 : h_out_end;

      int w_out_start =
          (w_in - filter_width + padding_width + real_stride_width) /
          real_stride_width;
      int _w_out_start = 0 > w_out_start ? 0 : w_out_start;

      int w_out_end = (w_in + padding_width) / real_stride_width;
      int _w_out_end =
          output_width - 1 < w_out_end ? output_width - 1 : w_out_end;

      T value = 0;

      for (int c_out = c_out_start;
           c_out < c_out_start + real_filter_multiplier; c_out++) {
        int filter_offset = (c_out + 1) * filter_height * filter_width;
        for (int h_out = h_out_start; h_out <= h_out_end;
             h_out += dilate_height) {
          for (int w_out = w_out_start; w_out <= w_out_end;
               w_out += dilate_width) {
            filter_offset--;
            if (h_out >= _h_out_start && h_out < _h_out_end &&
                w_out >= _w_out_start && w_out < _w_out_end) {
              const int output_grad_offset =
                  ((batch * output_channels + c_out) * output_height + h_out) *
                      output_width +
                  w_out;
              value += output_grad_data[output_grad_offset] *
                       filter_data[filter_offset];
            }
          }
        }
      }
      int index =
          ((batch * gridDim.x + c_in) * input_height + h_in) * input_width +
          w_in;
      input_grad_data[index] = value;
    }
  }
}

// Cuda kernel to compute the depthwise convolution backprop w.r.t. filter.
template <typename T>
__global__ void KernelDepthwiseConvFilterGrad(
    const T* output_grad_data, const T* input_data, const int num,
    const int output_channels, const int output_height, const int output_width,
    const int input_channels, const int input_height, const int input_width,
    const int filter_multiplier, const int filter_height,
    const int filter_width, const int stride_height, const int stride_width,
    const int padding_height, const int padding_width, const int dilate_height,
    const int dilate_width, T* filter_grad_data) {
  T s = 0;

  int gbid = ((blockIdx.z * gridDim.y) + blockIdx.y) * gridDim.x + blockIdx.x;
  // int gtid = threadIdx.y * blockDim.x + threadIdx.x;
  int lid = lane_id();

  for (int image_w = threadIdx.x; image_w < output_width;
       image_w += blockDim.x) {
    for (int bid = 0; bid < num; bid++) {
      for (int image_h = threadIdx.y; image_h < output_height;
           image_h += blockDim.y) {
        int kernel_id = blockIdx.z;
        int kernel_h = blockIdx.y - padding_height;
        int kernel_w = blockIdx.x - padding_width;

        int image_hk = image_h * stride_height + kernel_h * dilate_height;
        int image_wk = image_w * stride_width + kernel_w * dilate_width;
        if (image_hk < 0 || image_hk >= input_height) continue;
        if (image_wk < 0 || image_wk >= input_width) continue;
#define gaid(N, C, H, W) \
  ((((N)*gridDim.z + (C)) * output_height + (H)) * output_width + (W))
#define gaid2(N, C, H, W) \
  ((N * gridDim.z + C / filter_multiplier) * input_height + H) * input_width + W

        s += output_grad_data[gaid(bid, kernel_id, image_h, image_w)] *
             input_data[gaid2(bid, kernel_id, image_hk, image_wk)];

#undef gaid
#undef gaid2
      }
    }
  }
  s = warpReduceSum<T>(s);
  if (lid == 0) paddle::platform::CudaAtomicAdd(&filter_grad_data[gbid], s);
}

/*
 * All tensors are in NCHW format.
 * Ksize, strides, paddings are two elements. These two elements represent
 * height and width, respectively.
 */
template <class T>
class DepthwiseConvFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& filter,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::vector<int>& dilations,
                  framework::Tensor* output) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output->dims()[1];
    const int output_height = output->dims()[2];
    const int output_width = output->dims()[3];
    const int ksize_height = filter.dims()[2];
    const int ksize_width = filter.dims()[3];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];
    const int dilate_height = dilations[0];
    const int dilate_width = dilations[1];

    const T* input_data = input.data<T>();
    const T* filter_data = filter.data<T>();
    T* output_data = output->mutable_data<T>(context.GetPlace());

    int nthreads = batch_size * output_channels * output_height * output_width;
    int thread = 512;
    int blocks = std::min(std::max(thread / output_width, 1), output_height);
    dim3 threads(std::min(output_width, thread), blocks, 1);
    dim3 grid(output_channels, batch_size, 1);
    int filter_multiplier = output_channels / input_channels;

    if (filter_multiplier == 1 && stride_height == 1 && stride_width == 1) {
      KernelDepthwiseConv<T, 1, 1><<<grid, threads, 0, context.stream()>>>(
          nthreads, input_data, filter_data, batch_size, output_channels,
          output_height, output_width, input_channels, input_height,
          input_width, output_channels / input_channels, ksize_height,
          ksize_width, stride_height, stride_width, padding_height,
          padding_width, dilate_height, dilate_width, output_data);
    } else {
      KernelDepthwiseConv<T, 0, 0><<<grid, threads, 0, context.stream()>>>(
          nthreads, input_data, filter_data, batch_size, output_channels,
          output_height, output_width, input_channels, input_height,
          input_width, output_channels / input_channels, ksize_height,
          ksize_width, stride_height, stride_width, padding_height,
          padding_width, dilate_height, dilate_width, output_data);
    }
  }
};

template <typename T>
class DepthwiseConvInputGradFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& filter,
                  const framework::Tensor& output_grad,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::vector<int>& dilations,
                  framework::Tensor* input_grad) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output_grad.dims()[1];
    const int output_height = output_grad.dims()[2];
    const int output_width = output_grad.dims()[3];
    const int ksize_height = filter.dims()[2];
    const int ksize_width = filter.dims()[3];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];
    const int dilate_height = dilations[0];
    const int dilate_width = dilations[1];

    const T* filter_data = filter.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    int nthreads = batch_size * input_channels * input_height * input_width;
    int thread = 512;
    int blocks = std::min(std::max(thread / input_width, 1), input_height);
    dim3 threads(std::min(input_width, thread), blocks, 1);
    dim3 grid(input_channels, batch_size, 1);
    int filter_multiplier = output_channels / input_channels;

    if (filter_multiplier == 1 && stride_height == 1 && stride_width == 1) {
      KernelDepthwiseConvInputGrad<T, 1,
                                   1><<<grid, threads, 0, context.stream()>>>(
          nthreads, output_grad_data, filter_data, batch_size, output_channels,
          output_height, output_width, input_channels, input_height,
          input_width, output_channels / input_channels, ksize_height,
          ksize_width, stride_height, stride_width, padding_height,
          padding_width, dilate_height, dilate_width, input_grad_data);
    } else {
      KernelDepthwiseConvInputGrad<T, 0,
                                   0><<<grid, threads, 0, context.stream()>>>(
          nthreads, output_grad_data, filter_data, batch_size, output_channels,
          output_height, output_width, input_channels, input_height,
          input_width, output_channels / input_channels, ksize_height,
          ksize_width, stride_height, stride_width, padding_height,
          padding_width, dilate_height, dilate_width, input_grad_data);
    }
  }
};

template <typename T>
class DepthwiseConvFilterGradFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& output_grad,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::vector<int>& dilations,
                  framework::Tensor* filter_grad) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output_grad.dims()[1];
    const int output_height = output_grad.dims()[2];
    const int output_width = output_grad.dims()[3];
    const int ksize_height = filter_grad->dims()[2];
    const int ksize_width = filter_grad->dims()[3];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];
    const int dilate_height = dilations[0];
    const int dilate_width = dilations[1];

    const T* input_data = input.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* filter_grad_data = filter_grad->mutable_data<T>(context.GetPlace());

    int nthreads = batch_size * output_channels * output_height * output_width;

    int block_size = 512;
    int crop_output_height =
        std::min(std::max(block_size / output_width, 1), output_height);
    dim3 grid(ksize_width, ksize_height, output_channels);
    dim3 threads(std::min(output_width, block_size), crop_output_height, 1);

    KernelDepthwiseConvFilterGrad<T><<<grid, threads, 0, context.stream()>>>(
        output_grad_data, input_data, batch_size, output_channels,
        output_height, output_width, input_channels, input_height, input_width,
        output_channels / input_channels, ksize_height, ksize_width,
        stride_height, stride_width, padding_height, padding_width,
        dilate_height, dilate_width, filter_grad_data);
  }
};

template class DepthwiseConvFunctor<platform::CUDADeviceContext, float>;
template class DepthwiseConvFunctor<platform::CUDADeviceContext, double>;

template class DepthwiseConvInputGradFunctor<platform::CUDADeviceContext,
                                             float>;
template class DepthwiseConvInputGradFunctor<platform::CUDADeviceContext,
                                             double>;

template class DepthwiseConvFilterGradFunctor<platform::CUDADeviceContext,
                                              float>;
template class DepthwiseConvFilterGradFunctor<platform::CUDADeviceContext,
                                              double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
