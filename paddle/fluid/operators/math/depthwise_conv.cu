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
#if CUDA_VERSION < 9000
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down(val, offset);
  return val;
#else
#define FULL_MASK 0xffffffff
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);
  return val;
#endif
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

#define ARG_DEFINE_KernelDepthwiseConv                                         \
  const T *const input_data, const T *const filter_data, const int batch_size, \
      const int output_channels, const int output_height,                      \
      const int output_width, const int input_channels,                        \
      const int input_height, const int input_width,                           \
      const int filter_multiplier, const int filter_height,                    \
      const int filter_width, const int stride_height, const int stride_width, \
      const int padding_height, const int padding_width,                       \
      const int dilate_height, const int dilate_width, T *const output_data

// A Cuda kernel to compute the depthwise convolution forward pass
// in NCHW format.
template <typename T>
__device__ __inline__ void KernelDepthwiseConv(ARG_DEFINE_KernelDepthwiseConv) {
  for (int w_out = threadIdx.x; w_out < output_width; w_out += blockDim.x) {
    for (int h_out = threadIdx.y; h_out < output_height; h_out += blockDim.y) {
      const int batch = blockIdx.y;
      const int c_out = blockIdx.x;

      const int c_in = c_out / filter_multiplier;
      const T* weight = filter_data + c_out * filter_height * filter_width;
      T value = 0;
      const int h_in_start = -padding_height + h_out * stride_height;
      const int w_in_start = -padding_width + w_out * stride_width;
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

template <typename T, int c_filter>
__device__ __inline__ void KernelDepthwiseConvCFilter(
    ARG_DEFINE_KernelDepthwiseConv) {
  const int kWeghtSize = c_filter * c_filter;
  T r_weight[kWeghtSize];
  const int batch = blockIdx.y;
  const int c_out = blockIdx.x;
  const T* weight = filter_data + c_out * c_filter * c_filter;
  for (int i = 0; i < c_filter * c_filter; i++) r_weight[i] = weight[i];

  for (int w_out = threadIdx.x; w_out < output_width; w_out += blockDim.x) {
    for (int h_out = threadIdx.y; h_out < output_height; h_out += blockDim.y) {
      const int batch = blockIdx.y;
      const int c_out = blockIdx.x;

      const int c_in = c_out / filter_multiplier;
      T value = 0;
      const int h_in_start = -padding_height + h_out * stride_height;
      const int w_in_start = -padding_width + w_out * stride_width;
      const int h_in_end = h_in_start + c_filter * dilate_height;
      const int w_in_end = w_in_start + c_filter * dilate_width;

      const int in_offset =
          ((batch * input_channels + c_in) * input_height) * input_width;

      const int h_end = h_in_end < input_height ? h_in_end : input_height;
      const int w_end = w_in_end < input_width ? w_in_end : input_width;
      const int h_start = h_in_start > 0 ? h_in_start : 0;
      const int w_start = w_in_start > 0 ? w_in_start : 0;

      for (int h_in = h_in_start, h_f = 0; h_f < c_filter;
           h_in += dilate_height, h_f++) {
        for (int w_in = w_in_start, w_f = 0; w_f < c_filter;
             w_in += dilate_width, w_f++) {
          if (h_in >= 0 && h_in < input_height && w_in >= 0 &&
              w_in < input_width) {
            const int offset = in_offset + h_in * input_width + w_in;
            value += r_weight[h_f * c_filter + w_f] * input_data[offset];
          }
        }
      }
      int index =
          ((batch * gridDim.x + c_out) * output_height + h_out) * output_width +
          w_out;
      output_data[index] = value;
    }
  }
}

template <typename T, int c_filter_multiplier, int c_stride, int c_filter>
__global__ void KernelDepthwiseConvSp(ARG_DEFINE_KernelDepthwiseConv) {
  if (c_filter_multiplier == 0) {
    if (c_filter == -1)
      KernelDepthwiseConv<T>(
          input_data, filter_data, batch_size, output_channels, output_height,
          output_width, input_channels, input_height, input_width,
          filter_multiplier, filter_height, filter_width, stride_height,
          stride_width, padding_height, padding_width, dilate_height,
          dilate_width, output_data);
    else
      KernelDepthwiseConvCFilter<T, c_filter>(
          input_data, filter_data, batch_size, output_channels, output_height,
          output_width, input_channels, input_height, input_width,
          filter_multiplier, filter_height, filter_width, stride_height,
          stride_width, padding_height, padding_width, dilate_height,
          dilate_width, output_data);
  } else {
    if (c_filter == -1)
      KernelDepthwiseConv<T>(input_data, filter_data, batch_size,
                             output_channels, output_height, output_width,
                             input_channels, input_height, input_width,
                             c_filter_multiplier, filter_height, filter_height,
                             c_stride, c_stride, padding_height, padding_width,
                             dilate_height, dilate_width, output_data);
    else
      KernelDepthwiseConvCFilter<T, c_filter>(
          input_data, filter_data, batch_size, output_channels, output_height,
          output_width, input_channels, input_height, input_width,
          c_filter_multiplier, filter_height, filter_height, c_stride, c_stride,
          padding_height, padding_width, dilate_height, dilate_width,
          output_data);
  }
}

// CUDA kernel to compute the depthwise convolution backprop w.r.t input.
#define ARG_DEFINE_KernelDepthwiseConvInputGrad                                \
  const T *const output_grad_data, const T *const filter_data,                 \
      const int batch_size, const int output_channels,                         \
      const int output_height, const int output_width,                         \
      const int input_channels, const int input_height, const int input_width, \
      const int filter_multiplier, const int filter_height,                    \
      const int filter_width, const int stride_height, const int stride_width, \
      const int padding_height, const int padding_width,                       \
      const int dilate_height, const int dilate_width,                         \
      T *const input_grad_data

template <typename T>
__device__ __inline__ void KernelDepthwiseConvInputGrad(
    ARG_DEFINE_KernelDepthwiseConvInputGrad) {
  for (int w_in = threadIdx.x; w_in < input_width; w_in += blockDim.x) {
    for (int h_in = threadIdx.y; h_in < input_height; h_in += blockDim.y) {
      const int batch = blockIdx.y;
      const int c_in = blockIdx.x;

      const int c_out_start = c_in * filter_multiplier;

      int h_out_start =
          h_in - (filter_height - 1) * dilate_height + padding_height;

      int h_out_end = h_in + padding_height;

      int w_out_start =
          w_in - (filter_width - 1) * dilate_width + padding_width;

      int w_out_end = w_in + padding_width;

      T value = 0;

      for (int c_out = c_out_start; c_out < c_out_start + filter_multiplier;
           c_out++) {
        int filter_offset = (c_out + 1) * filter_height * filter_width;
        for (int h_out = h_out_start; h_out <= h_out_end;
             h_out += dilate_height) {
          for (int w_out = w_out_start; w_out <= w_out_end;
               w_out += dilate_width) {
            filter_offset--;
            int s_h_out = h_out / stride_height;
            int s_w_out = w_out / stride_width;
            if (h_out % stride_height == 0 && w_out % stride_width == 0 &&
                s_h_out >= 0 && s_h_out < output_height && s_w_out >= 0 &&
                s_w_out < output_width) {
              const int output_grad_offset =
                  ((batch * output_channels + c_out) * output_height +
                   s_h_out) *
                      output_width +
                  s_w_out;
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

template <typename T, int c_filter, int c_filter_multiplier>
__device__ __inline__ void KernelDepthwiseConvInputGradCFilter(
    ARG_DEFINE_KernelDepthwiseConvInputGrad) {
  const int kWeghtSize = c_filter * c_filter * c_filter_multiplier + 1;
  T r_weight[kWeghtSize];
  const int batch = blockIdx.y;
  const int c_in = blockIdx.x;

  for (int c_i = 0; c_i < filter_multiplier; c_i++) {
    int c_out = c_in * filter_multiplier + c_i;
    const T* weight = filter_data + c_out * c_filter * c_filter;
    for (int i = 0; i < c_filter * c_filter; i++)
      r_weight[i + c_i * c_filter * c_filter] =
          weight[c_filter * c_filter - i - 1];
  }

  for (int w_in = threadIdx.x; w_in < input_width; w_in += blockDim.x) {
    for (int h_in = threadIdx.y; h_in < input_height; h_in += blockDim.y) {
      const int batch = blockIdx.y;
      const int c_in = blockIdx.x;

      int h_out_start = h_in - (c_filter - 1) * dilate_height + padding_height;

      int w_out_start = w_in - (c_filter - 1) * dilate_width + padding_width;

      T value = 0;

      for (int c_i = 0; c_i < filter_multiplier; c_i++) {
        int c_out = c_in * filter_multiplier + c_i;
        for (int h_out = h_out_start, h_f = 0; h_f < c_filter;
             h_out += dilate_height, h_f++) {
          for (int w_out = w_out_start, w_f = 0; w_f < c_filter;
               w_out += dilate_width, w_f++) {
            int s_h_out = h_out / stride_height;
            int s_w_out = w_out / stride_width;
            if (h_out % stride_height == 0 && w_out % stride_width == 0 &&
                s_h_out >= 0 && s_h_out < output_height && s_w_out >= 0 &&
                s_w_out < output_width) {
              const int output_grad_offset =
                  ((batch * output_channels + c_out) * output_height +
                   s_h_out) *
                      output_width +
                  s_w_out;
              value +=
                  output_grad_data[output_grad_offset] *
                  r_weight[h_f * c_filter + w_f + c_i * c_filter * c_filter];
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

template <typename T, int c_filter_multiplier, int c_stride, int c_filter>
__global__ void KernelDepthwiseConvInputGradSp(
    ARG_DEFINE_KernelDepthwiseConvInputGrad) {
  if (c_filter_multiplier == 0)
    KernelDepthwiseConvInputGrad<T>(
        output_grad_data, filter_data, batch_size, output_channels,
        output_height, output_width, input_channels, input_height, input_width,
        filter_multiplier, filter_height, filter_width, stride_height,
        stride_width, padding_height, padding_width, dilate_height,
        dilate_width, input_grad_data);
  else if (c_filter == -1)
    KernelDepthwiseConvInputGrad<T>(
        output_grad_data, filter_data, batch_size, output_channels,
        output_height, output_width, input_channels, input_height, input_width,
        c_filter_multiplier, filter_height, filter_width, c_stride, c_stride,
        padding_height, padding_width, dilate_height, dilate_width,
        input_grad_data);
  else
    KernelDepthwiseConvInputGradCFilter<T, c_filter, c_filter_multiplier>(
        output_grad_data, filter_data, batch_size, output_channels,
        output_height, output_width, input_channels, input_height, input_width,
        c_filter_multiplier, filter_height, filter_width, c_stride, c_stride,
        padding_height, padding_width, dilate_height, dilate_width,
        input_grad_data);
}

// Cuda kernel to compute the depthwise convolution backprop w.r.t. filter.
template <typename T>
__device__ __inline__ void KernelDepthwiseConvFilterGrad(
    const T* output_grad_data, const T* input_data, const int num,
    const int output_channels, const int output_height, const int output_width,
    const int input_channels, const int input_height, const int input_width,
    const int filter_multiplier, const int filter_height,
    const int filter_width, const int stride_height, const int stride_width,
    const int padding_height, const int padding_width, const int dilate_height,
    const int dilate_width, T* filter_grad_data) {
  T s = 0;

  int gbid = ((blockIdx.z * gridDim.y) + blockIdx.y) * gridDim.x + blockIdx.x;
  int lid = lane_id();

  for (int image_w = threadIdx.x; image_w < output_width;
       image_w += blockDim.x) {
    for (int bid = 0; bid < num; bid++) {
      for (int image_h = threadIdx.y; image_h < output_height;
           image_h += blockDim.y) {
        int kernel_id = blockIdx.z;
        int kernel_h = blockIdx.y * dilate_height - padding_height;
        int kernel_w = blockIdx.x * dilate_width - padding_width;

        int image_hk = image_h * stride_height + kernel_h;
        int image_wk = image_w * stride_width + kernel_w;
        if (image_hk < 0 || image_hk >= input_height) continue;
        if (image_wk < 0 || image_wk >= input_width) continue;
#define gaid(N, C, H, W) \
  ((((N)*gridDim.z + (C)) * output_height + (H)) * output_width + (W))

        s += output_grad_data[gaid(bid, kernel_id, image_h, image_w)] *
             input_data[((bid * (gridDim.z / filter_multiplier) +
                          kernel_id / filter_multiplier) *
                             input_height +
                         image_hk) *
                            input_width +
                        image_wk];

#undef gaid
      }
    }
  }
#if __CUDA_ARCH__ >= 530
  s = warpReduceSum<T>(s);
  if (lid == 0) paddle::platform::CudaAtomicAdd(&filter_grad_data[gbid], s);
#else
  paddle::platform::CudaAtomicAdd(&filter_grad_data[gbid], s);
#endif
}

template <typename T, int c_filter_multiplier>
__global__ void KernelDepthwiseConvFilterGradSp(
    const T* output_grad_data, const T* input_data, const int num,
    const int output_channels, const int output_height, const int output_width,
    const int input_channels, const int input_height, const int input_width,
    const int filter_multiplier, const int filter_height,
    const int filter_width, const int stride_height, const int stride_width,
    const int padding_height, const int padding_width, const int dilate_height,
    const int dilate_width, T* filter_grad_data) {
  if (c_filter_multiplier == 0)
    KernelDepthwiseConvFilterGrad<T>(
        output_grad_data, input_data, num, output_channels, output_height,
        output_width, input_channels, input_height, input_width,
        filter_multiplier, filter_height, filter_width, stride_height,
        stride_width, padding_height, padding_width, dilate_height,
        dilate_width, filter_grad_data);
  else
    KernelDepthwiseConvFilterGrad<T>(
        output_grad_data, input_data, num, output_channels, output_height,
        output_width, input_channels, input_height, input_width,
        c_filter_multiplier, filter_height, filter_width, stride_height,
        stride_width, padding_height, padding_width, dilate_height,
        dilate_width, filter_grad_data);
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

    int thread = 512;
    int blocks = std::min(std::max(thread / output_width, 1), output_height);
    dim3 threads(std::min(output_width, thread), blocks, 1);
    dim3 grid(output_channels, batch_size, 1);
    int filter_multiplier = output_channels / input_channels;
#define check_case(c_filter_multiplier, c_stride, c_filter)                  \
  if (c_filter_multiplier == 0 ||                                            \
      filter_multiplier == c_filter_multiplier &&                            \
          stride_height == stride_width && stride_height == c_stride &&      \
          (ksize_height == ksize_width && ksize_height == c_filter ||        \
           c_filter == -1)) {                                                \
    KernelDepthwiseConvSp<T, c_filter_multiplier, c_stride,                  \
                          c_filter><<<grid, threads, 0, context.stream()>>>( \
        input_data, filter_data, batch_size, output_channels, output_height, \
        output_width, input_channels, input_height, input_width,             \
        filter_multiplier, ksize_height, ksize_width, stride_height,         \
        stride_width, padding_height, padding_width, dilate_height,          \
        dilate_width, output_data);                                          \
    return;                                                                  \
  }
    check_case(1, 1, 3);
    check_case(1, 1, 5);
    check_case(1, 1, -1);
    check_case(1, 2, 3);
    check_case(1, 2, 5);
    check_case(1, 2, -1);
    check_case(0, 0, 3);
    check_case(0, 0, 5);
    check_case(0, 0, -1);
// NOTE(liangdun): 0,0 for other case
// add other case if needed, e.g. check_case(2^n,1)
#undef check_case
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

    int thread = 512;
    int blocks = std::min(std::max(thread / input_width, 1), input_height);
    dim3 threads(std::min(input_width, thread), blocks, 1);
    dim3 grid(input_channels, batch_size, 1);
    int filter_multiplier = output_channels / input_channels;

#define check_case(c_filter_multiplier, c_stride, c_filter)             \
  if (c_filter_multiplier == 0 ||                                       \
      filter_multiplier == c_filter_multiplier &&                       \
          stride_height == stride_width && stride_height == c_stride && \
          (ksize_height == ksize_width && ksize_height == c_filter ||   \
           c_filter == -1)) {                                           \
    KernelDepthwiseConvInputGradSp<                                     \
        T, c_filter_multiplier, c_stride,                               \
        c_filter><<<grid, threads, 0, context.stream()>>>(              \
        output_grad_data, filter_data, batch_size, output_channels,     \
        output_height, output_width, input_channels, input_height,      \
        input_width, filter_multiplier, ksize_height, ksize_width,      \
        stride_height, stride_width, padding_height, padding_width,     \
        dilate_height, dilate_width, input_grad_data);                  \
    return;                                                             \
  }
    check_case(1, 1, 3);
    check_case(1, 1, 5);
    check_case(1, 1, -1);
    check_case(1, 2, 3);
    check_case(1, 2, 5);
    check_case(1, 2, -1);
    check_case(2, 1, 3);
    check_case(2, 1, 5);
    check_case(2, 1, -1);
    check_case(2, 2, 3);
    check_case(2, 2, 5);
    check_case(2, 2, -1);
    check_case(0, 0, -1);
// NOTE(liangdun): 0,0 for other case
// add other case if needed, e.g. check_case(2^n,1)
#undef check_case
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

    int block_size = 512;
    int crop_output_height =
        std::min(std::max(block_size / output_width, 1), output_height);
    dim3 grid(ksize_width, ksize_height, output_channels);
    dim3 threads(std::min(output_width, block_size), crop_output_height, 1);
    int filter_multiplier = output_channels / input_channels;

#define check_case(c_filter_multiplier)                                       \
  if (c_filter_multiplier == 0 || c_filter_multiplier == filter_multiplier) { \
    KernelDepthwiseConvFilterGradSp<                                          \
        T, c_filter_multiplier><<<grid, threads, 0, context.stream()>>>(      \
        output_grad_data, input_data, batch_size, output_channels,            \
        output_height, output_width, input_channels, input_height,            \
        input_width, filter_multiplier, ksize_height, ksize_width,            \
        stride_height, stride_width, padding_height, padding_width,           \
        dilate_height, dilate_width, filter_grad_data);                       \
    return;                                                                   \
  }
    check_case(1);
    check_case(0);
#undef check_case
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
