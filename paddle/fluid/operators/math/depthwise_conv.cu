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
#ifdef __NVCC__
#include <cub/cub.cuh>
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/fluid/operators/math/depthwise_conv.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
static __forceinline__ __device__ T WarpReduceSum(T val, int warp_size) {
  typedef cub::WarpReduce<T> WarpReduce;
  typename WarpReduce::TempStorage temp_storage;
  val = WarpReduce(temp_storage).Sum(val, warp_size);
  return val;
}

template <typename T>
__forceinline__ __device__ T BlockReduceSum(T val) {
  static __shared__ T shared[32];
  int thread_id = threadIdx.x + threadIdx.y * blockDim.x +
                  threadIdx.z * blockDim.x * blockDim.y;
  int warp_size = min(blockDim.x * blockDim.y * blockDim.z, warpSize);
  int lane = thread_id % warp_size;
  int wid = thread_id / warp_size;

  val = WarpReduceSum(val, warp_size);  // Each warp performs partial reduction

  if (lane == 0) shared[wid] = val;  // Write reduced value to shared memory
  __syncthreads();                   // Wait for all partial reductions

  // read from shared memory only if that warp existed
  int block_size = blockDim.x * blockDim.y * blockDim.z;
  if (thread_id < (block_size - 1) / warp_size + 1) {
    val = shared[lane];
  } else {
    val = static_cast<T>(0);
  }

  if (wid == 0) {
    val = WarpReduceSum(val, warp_size);  // Final reduce within first warp
  }
  __syncthreads();
  if (thread_id != 0) {
    val = static_cast<T>(0);
  }
  return val;
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
template <typename T, bool fuse_relu_before_conv>
__device__ __inline__ void KernelDepthwiseConvNCHW(
    ARG_DEFINE_KernelDepthwiseConv) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= (output_channels * batch_size * output_height * output_width))
    return;

  const int w_out = idx % output_width;
  const int h_out = (idx / output_width) % output_height;
  const int c_out = (idx / output_width / output_height) % output_channels;
  const int batch = idx / output_width / output_height / output_channels;

  const int c_in = c_out / filter_multiplier;
  const T* weight = filter_data + c_out * filter_height * filter_width;
  T value = 0;
  const int h_in_start = -padding_height + h_out * stride_height;
  const int w_in_start = -padding_width + w_out * stride_width;
  const int h_in_end = h_in_start + filter_height * dilate_height;
  const int w_in_end = w_in_start + filter_width * dilate_width;

  int in_offset =
      ((batch * input_channels + c_in) * input_height) * input_width;

  const int h_end = h_in_end < input_height ? h_in_end : input_height;
  const int w_end = w_in_end < input_width ? w_in_end : input_width;
  const int h_start = h_in_start > 0 ? h_in_start : 0;
  const int w_start = w_in_start > 0 ? w_in_start : 0;
  int weight_offset = 0;

#pragma unroll
  for (int h_in = h_in_start; h_in < h_in_end; h_in += dilate_height) {
#pragma unroll
    for (int w_in = w_in_start; w_in < w_in_end; w_in += dilate_width) {
      if (h_in >= h_start && h_in < h_end && w_in >= w_start && w_in < w_end) {
        int offset = in_offset + h_in * input_width + w_in;
        T in_data = input_data[offset];
        if (fuse_relu_before_conv) {
          value += weight[weight_offset] * max(0.0f, in_data);
        } else {
          value += weight[weight_offset] * in_data;
        }
      }
      weight_offset++;
    }
  }
  int index = batch * output_channels * output_height * output_width +
              c_out * output_height * output_width + h_out * output_width +
              w_out;
  output_data[index] = value;
}

// A Cuda kernel to compute the depthwise convolution forward pass
// in NHWC format.
template <typename T, bool fuse_relu_before_conv>
__device__ __inline__ void KernelDepthwiseConvNHWC(
    ARG_DEFINE_KernelDepthwiseConv) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= (output_channels * batch_size * output_height * output_width))
    return;

  const int c_out = idx % output_channels;
  const int w_out = (idx / output_channels) % output_width;
  const int h_out = (idx / output_channels / output_width) % output_height;
  const int batch = idx / output_width / output_height / output_channels;

  const int c_in = c_out / filter_multiplier;
  T value = 0;
  const int h_in_start = -padding_height + h_out * stride_height;
  const int w_in_start = -padding_width + w_out * stride_width;
  const int h_in_end = h_in_start + filter_height * dilate_height;
  const int w_in_end = w_in_start + filter_width * dilate_width;

  const int h_end = h_in_end < input_height ? h_in_end : input_height;
  const int w_end = w_in_end < input_width ? w_in_end : input_width;
  const int h_start = h_in_start > 0 ? h_in_start : 0;
  const int w_start = w_in_start > 0 ? w_in_start : 0;
  int weight_offset = 0;

#pragma unroll
  for (int h_in = h_in_start; h_in < h_in_end; h_in += dilate_height) {
#pragma unroll
    for (int w_in = w_in_start; w_in < w_in_end; w_in += dilate_width) {
      if (h_in >= h_start && h_in < h_end && w_in >= w_start && w_in < w_end) {
        int offset = ((batch * input_height + h_in) * input_width + w_in) *
                         input_channels +
                     c_in;
        T in_data = input_data[offset];
        const T* weight = filter_data + weight_offset * output_channels + c_out;
        if (fuse_relu_before_conv) {
          value += weight[0] * max(0.0f, in_data);
        } else {
          value += weight[0] * in_data;
        }
      }
      weight_offset++;
    }
  }
  int index = batch * output_channels * output_height * output_width +
              h_out * output_width * output_channels + w_out * output_channels +
              c_out;
  output_data[index] = value;
}

template <typename T, int c_filter, bool fuse_relu_before_conv>
__device__ __inline__ void KernelDepthwiseConvCFilterNCHW(
    ARG_DEFINE_KernelDepthwiseConv) {
  const int kWeightSize = c_filter * c_filter;
  T r_weight[kWeightSize];
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

      int in_offset =
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
            int offset = in_offset + h_in * input_width + w_in;
            if (fuse_relu_before_conv) {
              value += r_weight[h_f * c_filter + w_f] *
                       max(0.0f, input_data[offset]);
            } else {
              value += r_weight[h_f * c_filter + w_f] * input_data[offset];
            }
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

template <typename T, int c_filter, bool fuse_relu_before_conv>
__device__ __inline__ void KernelDepthwiseConvCFilterNHWC(
    ARG_DEFINE_KernelDepthwiseConv) {
  const int batch = blockIdx.z;
  int h_out = blockIdx.x * dilate_height + blockIdx.y;
  if (h_out >= output_height) {
    return;
  }
  int in_offset = batch * input_height * input_width * input_channels;
  int out_offset =
      (batch * output_height + h_out) * output_width * output_channels;
  const int h_in_start = -padding_height + h_out * stride_height;
  const int wi_size = (output_width + dilate_width - 1) / dilate_width;
  const int kWeightSize = c_filter * c_filter;
  T r_weight[kWeightSize];

  for (int c_out = threadIdx.x; c_out < output_channels; c_out += blockDim.x) {
    for (int i = 0; i < c_filter * c_filter; i++) {
      const T* weight = filter_data + i * output_channels + c_out;
      r_weight[i] = weight[0];
    }
    const int c_in = c_out / filter_multiplier;
    for (int i = threadIdx.y; i < wi_size * dilate_width; i += blockDim.y) {
      int i_dw = i / wi_size;
      int i_wi = i - i_dw * wi_size;
      int w_out = i_wi * dilate_width + i_dw;
      if (w_out >= output_width) {
        continue;
      }
      T value = 0;
      const int w_in_start = -padding_width + w_out * stride_width;
      for (int h_in = h_in_start, h_f = 0; h_f < c_filter;
           h_in += dilate_height, h_f++) {
        for (int w_in = w_in_start, w_f = 0; w_f < c_filter;
             w_in += dilate_width, w_f++) {
          if (h_in >= 0 && h_in < input_height && w_in >= 0 &&
              w_in < input_width) {
            int offset =
                in_offset + (h_in * input_width + w_in) * input_channels + c_in;
            if (fuse_relu_before_conv) {
              value += r_weight[h_f * c_filter + w_f] *
                       max(0.0f, input_data[offset]);
            } else {
              value += r_weight[h_f * c_filter + w_f] * input_data[offset];
            }
          }
        }
      }
      int index = out_offset + w_out * output_channels + c_out;
      output_data[index] = value;
    }
  }
}

template <typename T, int c_filter_multiplier, int c_stride, int c_filter,
          DataLayout data_layout, bool fuse_relu_before_conv>
__global__ void KernelDepthwiseConvSp(ARG_DEFINE_KernelDepthwiseConv) {
  int final_filter_multiplier = filter_multiplier;
  int h_stride = stride_height;
  int w_stride = stride_width;
  if (c_filter_multiplier != 0) {
    final_filter_multiplier = c_filter_multiplier;
    h_stride = c_stride;
    w_stride = c_stride;
  }
  if (c_filter == -1) {
    if (data_layout != DataLayout::kNHWC) {
      KernelDepthwiseConvNCHW<T, fuse_relu_before_conv>(
          input_data, filter_data, batch_size, output_channels, output_height,
          output_width, input_channels, input_height, input_width,
          final_filter_multiplier, filter_height, filter_width, h_stride,
          w_stride, padding_height, padding_width, dilate_height, dilate_width,
          output_data);
    } else {
      KernelDepthwiseConvNHWC<T, fuse_relu_before_conv>(
          input_data, filter_data, batch_size, output_channels, output_height,
          output_width, input_channels, input_height, input_width,
          final_filter_multiplier, filter_height, filter_width, h_stride,
          w_stride, padding_height, padding_width, dilate_height, dilate_width,
          output_data);
    }
  } else {
    if (data_layout != DataLayout::kNHWC) {
      KernelDepthwiseConvCFilterNCHW<T, c_filter, fuse_relu_before_conv>(
          input_data, filter_data, batch_size, output_channels, output_height,
          output_width, input_channels, input_height, input_width,
          final_filter_multiplier, filter_height, filter_width, h_stride,
          w_stride, padding_height, padding_width, dilate_height, dilate_width,
          output_data);
    } else {
      KernelDepthwiseConvCFilterNHWC<T, c_filter, fuse_relu_before_conv>(
          input_data, filter_data, batch_size, output_channels, output_height,
          output_width, input_channels, input_height, input_width,
          final_filter_multiplier, filter_height, filter_width, h_stride,
          w_stride, padding_height, padding_width, dilate_height, dilate_width,
          output_data);
    }
  }
}

// CUDA kernel to compute the depthwise convolution backprop w.r.t input.
#define ARG_DEFINE_KernelDepthwiseConvInputGrad                                \
  const T *const input_data, const T *const output_grad_data,                  \
      const T *const filter_data, const int batch_size,                        \
      const int output_channels, const int output_height,                      \
      const int output_width, const int input_channels,                        \
      const int input_height, const int input_width,                           \
      const int filter_multiplier, const int filter_height,                    \
      const int filter_width, const int stride_height, const int stride_width, \
      const int padding_height, const int padding_width,                       \
      const int dilate_height, const int dilate_width,                         \
      T *const input_grad_data

template <typename T, bool fuse_relu_before_conv>
__device__ __inline__ void KernelDepthwiseConvInputGradNCHW(
    ARG_DEFINE_KernelDepthwiseConvInputGrad) {
  const int batch = blockIdx.y;
  const int c_in = blockIdx.x;
  for (int w_in = threadIdx.x; w_in < input_width; w_in += blockDim.x) {
    for (int h_in = threadIdx.y; h_in < input_height; h_in += blockDim.y) {
      const int c_out_start = c_in * filter_multiplier;
      int h_out_start =
          h_in - (filter_height - 1) * dilate_height + padding_height;
      int h_out_end = h_in + padding_height;
      int w_out_start =
          w_in - (filter_width - 1) * dilate_width + padding_width;
      int w_out_end = w_in + padding_width;

      T value = 0;
      int index =
          ((batch * gridDim.x + c_in) * input_height + h_in) * input_width +
          w_in;

      if (fuse_relu_before_conv) {
        if (input_data[index] <= 0) {
          input_grad_data[index] = 0;
          continue;
        }
      }

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
              int output_grad_offset =
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
      input_grad_data[index] = value;
    }
  }
}

template <typename T, bool fuse_relu_before_conv>
__device__ __inline__ void KernelDepthwiseConvInputGradNHWC(
    ARG_DEFINE_KernelDepthwiseConvInputGrad) {
  const int batch = blockIdx.z;
  int h_in = blockIdx.x * dilate_height + blockIdx.y;
  if (h_in >= input_height) {
    return;
  }

  for (int c_in = threadIdx.x; c_in < input_channels; c_in += blockDim.x) {
    for (int w_in = threadIdx.y; w_in < input_width; w_in += blockDim.y) {
      int h_out_start =
          h_in - (filter_height - 1) * dilate_height + padding_height;
      int w_out_start =
          w_in - (filter_width - 1) * dilate_width + padding_width;

      T value = 0;
      int index = ((batch * input_height + h_in) * input_width + w_in) *
                      input_channels +
                  c_in;
      if (fuse_relu_before_conv) {
        if (input_data[index] <= 0) {
          input_grad_data[index] = 0;
          continue;
        }
      }

      for (int c_i = 0; c_i < filter_multiplier; c_i++) {
        int c_out = c_in * filter_multiplier + c_i;
        int weight_offset = filter_height * filter_width;
        for (int h_out = h_out_start, h_f = 0; h_f < filter_height;
             h_out += dilate_height, h_f++) {
          for (int w_out = w_out_start, w_f = 0; w_f < filter_width;
               w_out += dilate_width, w_f++) {
            weight_offset--;
            int s_h_out = h_out / stride_height;
            int s_w_out = w_out / stride_width;
            if (h_out % stride_height == 0 && w_out % stride_width == 0 &&
                s_h_out >= 0 && s_h_out < output_height && s_w_out >= 0 &&
                s_w_out < output_width) {
              int output_grad_offset =
                  ((batch * output_height + s_h_out) * output_width + s_w_out) *
                      output_channels +
                  c_out;
              int filter_offset = weight_offset * output_channels + c_out;
              value += output_grad_data[output_grad_offset] *
                       filter_data[filter_offset];
            }
          }
        }
      }
      input_grad_data[index] = value;
    }
  }
}

template <typename T, int c_filter, int c_filter_multiplier,
          bool fuse_relu_before_conv>
__device__ __inline__ void KernelDepthwiseConvInputGradCFilterNCHW(
    ARG_DEFINE_KernelDepthwiseConvInputGrad) {
  const int kWeightSize = c_filter * c_filter * c_filter_multiplier + 1;
  T r_weight[kWeightSize];
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
      int h_out_start = h_in - (c_filter - 1) * dilate_height + padding_height;
      int w_out_start = w_in - (c_filter - 1) * dilate_width + padding_width;

      T value = 0;
      int index =
          ((batch * gridDim.x + c_in) * input_height + h_in) * input_width +
          w_in;
      if (fuse_relu_before_conv) {
        if (input_data[index] <= 0) {
          input_grad_data[index] = 0;
          continue;
        }
      }

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
              int output_grad_offset =
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
      input_grad_data[index] = value;
    }
  }
}

template <typename T, int c_filter, int c_filter_multiplier,
          bool fuse_relu_before_conv>
__device__ __inline__ void KernelDepthwiseConvInputGradCFilterNHWC(
    ARG_DEFINE_KernelDepthwiseConvInputGrad) {
  int h_in = blockIdx.x * dilate_height + blockIdx.y;
  if (h_in >= input_height) {
    return;
  }
  const int kWeightSize = c_filter * c_filter * c_filter_multiplier + 1;
  T r_weight[kWeightSize];
  const int batch = blockIdx.z;
  const int wi_size = (input_width + dilate_width - 1) / dilate_width;
  const int h_out_start =
      h_in - (c_filter - 1) * dilate_height + padding_height;

  for (int c_in = threadIdx.x; c_in < input_channels; c_in += blockDim.x) {
    for (int c_i = 0; c_i < c_filter_multiplier; c_i++) {
      int c_out = c_in * c_filter_multiplier + c_i;
      for (int i = 0; i < c_filter * c_filter; i++)
        r_weight[i + c_i * c_filter * c_filter] =
            filter_data[(c_filter * c_filter - i - 1) * output_channels +
                        c_out];
    }
    for (int i = threadIdx.y; i < wi_size * dilate_width; i += blockDim.y) {
      int i_dw = i / wi_size;
      int i_wi = i - i_dw * wi_size;
      int w_in = i_wi * dilate_width + i_dw;
      if (w_in >= input_width) {
        continue;
      }
      int w_out_start = w_in - (c_filter - 1) * dilate_width + padding_width;

      T value = 0;
      int index = ((batch * input_height + h_in) * input_width + w_in) *
                      input_channels +
                  c_in;
      if (fuse_relu_before_conv) {
        if (input_data[index] <= 0) {
          input_grad_data[index] = 0;
          continue;
        }
      }

      for (int c_i = 0; c_i < c_filter_multiplier; c_i++) {
        int c_out = c_in * c_filter_multiplier + c_i;
        for (int h_out = h_out_start, h_f = 0; h_f < c_filter;
             h_out += dilate_height, h_f++) {
          for (int w_out = w_out_start, w_f = 0; w_f < c_filter;
               w_out += dilate_width, w_f++) {
            int s_h_out = h_out / stride_height;
            int s_w_out = w_out / stride_width;
            if (h_out % stride_height == 0 && w_out % stride_width == 0 &&
                s_h_out >= 0 && s_h_out < output_height && s_w_out >= 0 &&
                s_w_out < output_width) {
              int output_grad_offset =
                  ((batch * output_height + s_h_out) * output_width + s_w_out) *
                      output_channels +
                  c_out;
              value +=
                  output_grad_data[output_grad_offset] *
                  r_weight[h_f * c_filter + w_f + c_i * c_filter * c_filter];
            }
          }
        }
      }
      input_grad_data[index] = value;
    }
  }
}

template <typename T, int c_filter_multiplier, int c_stride, int c_filter,
          DataLayout data_layout, bool fuse_relu_before_conv>
__global__ void KernelDepthwiseConvInputGradSp(
    ARG_DEFINE_KernelDepthwiseConvInputGrad) {
  int final_filter_multiplier = filter_multiplier;
  int h_stride = stride_height;
  int w_stride = stride_width;
  if (c_filter_multiplier != 0) {
    final_filter_multiplier = c_filter_multiplier;
    h_stride = c_stride;
    w_stride = c_stride;
  }

  if (c_filter_multiplier == 0 || c_filter == -1) {
    if (data_layout != DataLayout::kNHWC) {
      KernelDepthwiseConvInputGradNCHW<T, fuse_relu_before_conv>(
          input_data, output_grad_data, filter_data, batch_size,
          output_channels, output_height, output_width, input_channels,
          input_height, input_width, final_filter_multiplier, filter_height,
          filter_width, h_stride, w_stride, padding_height, padding_width,
          dilate_height, dilate_width, input_grad_data);
    } else {
      KernelDepthwiseConvInputGradNHWC<T, fuse_relu_before_conv>(
          input_data, output_grad_data, filter_data, batch_size,
          output_channels, output_height, output_width, input_channels,
          input_height, input_width, final_filter_multiplier, filter_height,
          filter_width, h_stride, w_stride, padding_height, padding_width,
          dilate_height, dilate_width, input_grad_data);
    }
  } else {
    if (data_layout != DataLayout::kNHWC) {
      KernelDepthwiseConvInputGradCFilterNCHW<T, c_filter, c_filter_multiplier,
                                              fuse_relu_before_conv>(
          input_data, output_grad_data, filter_data, batch_size,
          output_channels, output_height, output_width, input_channels,
          input_height, input_width, c_filter_multiplier, filter_height,
          filter_width, c_stride, c_stride, padding_height, padding_width,
          dilate_height, dilate_width, input_grad_data);
    } else {
      KernelDepthwiseConvInputGradCFilterNHWC<T, c_filter, c_filter_multiplier,
                                              fuse_relu_before_conv>(
          input_data, output_grad_data, filter_data, batch_size,
          output_channels, output_height, output_width, input_channels,
          input_height, input_width, c_filter_multiplier, filter_height,
          filter_width, c_stride, c_stride, padding_height, padding_width,
          dilate_height, dilate_width, input_grad_data);
    }
  }
}

// Cuda kernel to compute the depthwise convolution backprop w.r.t. filter.
template <typename T, bool fuse_relu_before_conv>
__device__ __inline__ void KernelDepthwiseConvFilterGradNCHW(
    const T* output_grad_data, const T* input_data, const int num,
    const int output_channels, const int output_height, const int output_width,
    const int input_channels, const int input_height, const int input_width,
    const int filter_multiplier, const int filter_height,
    const int filter_width, const int stride_height, const int stride_width,
    const int padding_height, const int padding_width, const int dilate_height,
    const int dilate_width, T* filter_grad_data) {
  T s = 0;
  int gbid = ((blockIdx.z * gridDim.y) + blockIdx.y) * gridDim.x + blockIdx.x;

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
        int input_id = ((bid * (gridDim.z / filter_multiplier) +
                         kernel_id / filter_multiplier) *
                            input_height +
                        image_hk) *
                           input_width +
                       image_wk;
        if (fuse_relu_before_conv) {
          s += output_grad_data[gaid(bid, kernel_id, image_h, image_w)] *
               max(0.0f, input_data[input_id]);
        } else {
          s += output_grad_data[gaid(bid, kernel_id, image_h, image_w)] *
               input_data[input_id];
        }
#undef gaid
      }
    }
  }

  T val = BlockReduceSum(s);
  platform::CudaAtomicAdd(&filter_grad_data[gbid], val);
}

template <typename T, bool fuse_relu_before_conv>
__device__ __inline__ void KernelDepthwiseConvFilterGradNHWC(
    const T* output_grad_data, const T* input_data, const int num,
    const int output_channels, const int output_height, const int output_width,
    const int input_channels, const int input_height, const int input_width,
    const int filter_multiplier, const int filter_height,
    const int filter_width, const int stride_height, const int stride_width,
    const int padding_height, const int padding_width, const int dilate_height,
    const int dilate_width, T* filter_grad_data) {
  int bid = blockIdx.z;
  int image_h = blockIdx.y;
  int kernel_iw = blockIdx.x % filter_width;
  int kernel_ih = blockIdx.x / filter_width;
  for (int kernel_id = threadIdx.x; kernel_id < output_channels;
       kernel_id += blockDim.x) {
    T s = 0;
    int gbid =
        ((kernel_id * filter_height) + kernel_ih) * filter_width + kernel_iw;
    for (int image_w = threadIdx.y; image_w < output_width;
         image_w += blockDim.y) {
      int kernel_h = kernel_ih * dilate_height - padding_height;
      int kernel_w = kernel_iw * dilate_width - padding_width;

      int image_hk = image_h * stride_height + kernel_h;
      int image_wk = image_w * stride_width + kernel_w;
      if (image_hk < 0 || image_hk >= input_height) continue;
      if (image_wk < 0 || image_wk >= input_width) continue;
#define gaid(N, H, W, C) \
  ((((N)*output_height + (H)) * output_width + (W)) * output_channels + (C))
      int input_id =
          ((bid * input_height + image_hk) * input_width + image_wk) *
              input_channels +
          kernel_id / filter_multiplier;
      if (fuse_relu_before_conv) {
        s += output_grad_data[gaid(bid, image_h, image_w, kernel_id)] *
             max(0.0f, input_data[input_id]);
      } else {
        s += output_grad_data[gaid(bid, image_h, image_w, kernel_id)] *
             input_data[input_id];
      }
#undef gaid
    }
    platform::CudaAtomicAdd(&filter_grad_data[gbid], s);
  }
}

template <typename T, int c_filter, bool fuse_relu_before_conv>
__device__ __inline__ void KernelDepthwiseConvFilterGradCFilterNHWC(
    const T* output_grad_data, const T* input_data, const int num,
    const int output_channels, const int output_height, const int output_width,
    const int input_channels, const int input_height, const int input_width,
    const int filter_multiplier, const int filter_height,
    const int filter_width, const int stride_height, const int stride_width,
    const int padding_height, const int padding_width, const int dilate_height,
    const int dilate_width, T* filter_grad_data) {
  const int bid = blockIdx.z;
  int image_h = blockIdx.x * dilate_height + blockIdx.y;
  if (image_h >= output_height) {
    return;
  }
  const int kWeightSize = c_filter * c_filter;
  T r_weight[kWeightSize];
  const int wi_size = (output_width + dilate_width - 1) / dilate_width;

  for (int kernel_id = threadIdx.x; kernel_id < output_channels;
       kernel_id += blockDim.x) {
    for (int i = 0; i < c_filter * c_filter; ++i) {
      r_weight[i] = 0;
    }
    for (int i = threadIdx.y; i < wi_size * dilate_width; i += blockDim.y) {
      int i_dw = i / wi_size;
      int i_wi = i - i_dw * wi_size;
      int image_w = i_wi * dilate_width + i_dw;
      if (image_w >= output_width) {
        continue;
      }
      for (int kernel_ih = 0; kernel_ih < c_filter; ++kernel_ih) {
        for (int kernel_iw = 0; kernel_iw < c_filter; ++kernel_iw) {
          int kernel_h = kernel_ih * dilate_height - padding_height;
          int kernel_w = kernel_iw * dilate_width - padding_width;
          int image_hk = image_h * stride_height + kernel_h;
          int image_wk = image_w * stride_width + kernel_w;
          if (image_hk < 0 || image_hk >= input_height) continue;
          if (image_wk < 0 || image_wk >= input_width) continue;
          int input_id =
              ((bid * input_height + image_hk) * input_width + image_wk) *
                  input_channels +
              kernel_id / filter_multiplier;
          int output_id =
              ((bid * output_height + image_h) * output_width + image_w) *
                  output_channels +
              kernel_id;
          T s = 0;
          if (fuse_relu_before_conv) {
            s = output_grad_data[output_id] * max(0.0f, input_data[input_id]);
          } else {
            s = output_grad_data[output_id] * input_data[input_id];
          }
          r_weight[kernel_ih * c_filter + kernel_iw] += s;
        }
      }
    }
    for (int i = 0; i < c_filter * c_filter; ++i) {
      T* weight = filter_grad_data + i * output_channels + kernel_id;
      platform::CudaAtomicAdd(&weight[0], r_weight[i]);
    }
  }
}

template <typename T, int c_filter_multiplier, int c_stride, int c_filter,
          DataLayout data_layout, bool fuse_relu_before_conv>
__global__ void KernelDepthwiseConvFilterGradSp(
    const T* output_grad_data, const T* input_data, const int num,
    const int output_channels, const int output_height, const int output_width,
    const int input_channels, const int input_height, const int input_width,
    const int filter_multiplier, const int filter_height,
    const int filter_width, const int stride_height, const int stride_width,
    const int padding_height, const int padding_width, const int dilate_height,
    const int dilate_width, T* filter_grad_data) {
  int final_filter_multiplier = filter_multiplier;
  int h_stride = stride_height;
  int w_stride = stride_width;
  if (c_filter_multiplier != 0) {
    final_filter_multiplier = c_filter_multiplier;
    h_stride = c_stride;
    w_stride = c_stride;
  }
  if (c_filter_multiplier == 0 || c_filter == -1) {
    if (data_layout != DataLayout::kNHWC) {
      KernelDepthwiseConvFilterGradNCHW<T, fuse_relu_before_conv>(
          output_grad_data, input_data, num, output_channels, output_height,
          output_width, input_channels, input_height, input_width,
          final_filter_multiplier, filter_height, filter_width, h_stride,
          w_stride, padding_height, padding_width, dilate_height, dilate_width,
          filter_grad_data);
    } else {
      KernelDepthwiseConvFilterGradNHWC<T, fuse_relu_before_conv>(
          output_grad_data, input_data, num, output_channels, output_height,
          output_width, input_channels, input_height, input_width,
          final_filter_multiplier, filter_height, filter_width, h_stride,
          w_stride, padding_height, padding_width, dilate_height, dilate_width,
          filter_grad_data);
    }
  } else {
    if (data_layout != DataLayout::kNHWC) {
      KernelDepthwiseConvFilterGradNCHW<T, fuse_relu_before_conv>(
          output_grad_data, input_data, num, output_channels, output_height,
          output_width, input_channels, input_height, input_width,
          final_filter_multiplier, filter_height, filter_width, h_stride,
          w_stride, padding_height, padding_width, dilate_height, dilate_width,
          filter_grad_data);
    } else {
      KernelDepthwiseConvFilterGradCFilterNHWC<T, c_filter,
                                               fuse_relu_before_conv>(
          output_grad_data, input_data, num, output_channels, output_height,
          output_width, input_channels, input_height, input_width,
          final_filter_multiplier, filter_height, filter_width, h_stride,
          w_stride, padding_height, padding_width, dilate_height, dilate_width,
          filter_grad_data);
    }
  }
}

/*
 * All tensors are in NCHW format.
 * Ksize, strides, paddings are two elements. These two elements represent
 * height and width, respectively.
 */
template <class T, bool fuse_relu_before_conv>
class DepthwiseConvFunctor<platform::CUDADeviceContext, T,
                           fuse_relu_before_conv> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& filter,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::vector<int>& dilations, framework::Tensor* output,
                  const DataLayout data_layout = DataLayout::kNCHW) {
    const int batch_size = input.dims()[0];
    const int input_channels =
        (data_layout != DataLayout::kNHWC ? input.dims()[1] : input.dims()[3]);
    const int input_height =
        (data_layout != DataLayout::kNHWC ? input.dims()[2] : input.dims()[1]);
    const int input_width =
        (data_layout != DataLayout::kNHWC ? input.dims()[3] : input.dims()[2]);
    const int output_channels =
        (data_layout != DataLayout::kNHWC ? output->dims()[1]
                                          : output->dims()[3]);
    const int output_height =
        (data_layout != DataLayout::kNHWC ? output->dims()[2]
                                          : output->dims()[1]);
    const int output_width =
        (data_layout != DataLayout::kNHWC ? output->dims()[3]
                                          : output->dims()[2]);
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

    framework::Tensor filter_hwc;
    if (data_layout == DataLayout::kNHWC) {
      framework::DDim filter_hwc_dims({filter.dims()[2], filter.dims()[3],
                                       filter.dims()[0], filter.dims()[1]});
      filter_hwc.Resize(filter_hwc_dims);
      filter_hwc.mutable_data<T>(context.GetPlace());
      std::vector<int> perm_axis({2, 3, 0, 1});
      math::TransposeNormal<platform::CUDADeviceContext, T> trans;
      trans(context, filter, &filter_hwc, perm_axis);
      filter_data = filter_hwc.data<T>();
    }

    int thread = 512;
    int blocks;
    dim3 threads;
    dim3 grid;

    if (data_layout != DataLayout::kNHWC) {
      if (output_width > 1024 && output_width <= 2048)
        thread = (output_width - 1) / 2 + 1;
      else if (output_width > 512 && output_width <= 1024)
        thread = output_width;
#ifdef __HIPCC__
      thread = std::min(thread, 256);
#endif
      blocks = std::min(std::max(thread / output_width, 1), output_height);
      threads = dim3(std::min(output_width, thread), blocks, 1);
      grid = dim3(output_channels, batch_size, 1);
    } else {
#ifdef __HIPCC__
      thread = std::min(thread, 256);
#endif
      blocks = std::min(
          std::max(thread / output_channels, 1),
          ((output_width + dilate_width - 1) / dilate_width) * dilate_width);
      threads = dim3(std::min(output_channels, thread), blocks, 1);
      grid = dim3((output_height + dilate_height - 1) / dilate_height,
                  dilate_height, batch_size);
    }
    int filter_multiplier = output_channels / input_channels;
    int nums_output =
        batch_size * output_channels * output_height * output_width;
#ifdef __HIPCC__
    int block_size = 256;
#else
    int block_size = 512;
#endif
    int grid_size = (nums_output + block_size - 1) / block_size;

#define check_case(c_filter_multiplier, c_stride, c_filter)                    \
  if (c_filter_multiplier == 0 ||                                              \
      filter_multiplier == c_filter_multiplier &&                              \
          stride_height == stride_width && stride_height == c_stride &&        \
          (ksize_height == ksize_width && ksize_height == c_filter ||          \
           c_filter == -1)) {                                                  \
    if (c_filter == -1) {                                                      \
      threads.x = block_size;                                                  \
      grid.x = grid_size;                                                      \
      threads.y = threads.z = grid.y = grid.z = 1;                             \
    }                                                                          \
    if (data_layout != DataLayout::kNHWC) {                                    \
      KernelDepthwiseConvSp<                                                   \
          T, c_filter_multiplier, c_stride, c_filter, DataLayout::kNCHW,       \
          fuse_relu_before_conv><<<grid, threads, 0, context.stream()>>>(      \
          input_data, filter_data, batch_size, output_channels, output_height, \
          output_width, input_channels, input_height, input_width,             \
          filter_multiplier, ksize_height, ksize_width, stride_height,         \
          stride_width, padding_height, padding_width, dilate_height,          \
          dilate_width, output_data);                                          \
    } else {                                                                   \
      KernelDepthwiseConvSp<                                                   \
          T, c_filter_multiplier, c_stride, c_filter, DataLayout::kNHWC,       \
          fuse_relu_before_conv><<<grid, threads, 0, context.stream()>>>(      \
          input_data, filter_data, batch_size, output_channels, output_height, \
          output_width, input_channels, input_height, input_width,             \
          filter_multiplier, ksize_height, ksize_width, stride_height,         \
          stride_width, padding_height, padding_width, dilate_height,          \
          dilate_width, output_data);                                          \
    }                                                                          \
    return;                                                                    \
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

template <typename T, bool fuse_relu_before_conv>
class DepthwiseConvInputGradFunctor<platform::CUDADeviceContext, T,
                                    fuse_relu_before_conv> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& filter,
                  const framework::Tensor& output_grad,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::vector<int>& dilations,
                  framework::Tensor* input_grad,
                  const DataLayout data_layout = DataLayout::kNCHW) {
    const int batch_size = input.dims()[0];
    const int input_channels =
        (data_layout != DataLayout::kNHWC ? input.dims()[1] : input.dims()[3]);
    const int input_height =
        (data_layout != DataLayout::kNHWC ? input.dims()[2] : input.dims()[1]);
    const int input_width =
        (data_layout != DataLayout::kNHWC ? input.dims()[3] : input.dims()[2]);
    const int output_channels =
        (data_layout != DataLayout::kNHWC ? output_grad.dims()[1]
                                          : output_grad.dims()[3]);
    const int output_height =
        (data_layout != DataLayout::kNHWC ? output_grad.dims()[2]
                                          : output_grad.dims()[1]);
    const int output_width =
        (data_layout != DataLayout::kNHWC ? output_grad.dims()[3]
                                          : output_grad.dims()[2]);
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
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    framework::Tensor filter_hwc;
    if (data_layout == DataLayout::kNHWC) {
      framework::DDim filter_hwc_dims({filter.dims()[2], filter.dims()[3],
                                       filter.dims()[0], filter.dims()[1]});
      filter_hwc.Resize(filter_hwc_dims);
      filter_hwc.mutable_data<T>(context.GetPlace());
      std::vector<int> perm_axis({2, 3, 0, 1});
      math::TransposeNormal<platform::CUDADeviceContext, T> trans;
      trans(context, filter, &filter_hwc, perm_axis);
      filter_data = filter_hwc.data<T>();
    }

    int thread = 512;
    int blocks;
    dim3 threads;
    dim3 grid;

    if (data_layout != DataLayout::kNHWC) {
      if (input_width > 1024 && input_width <= 2048) {
        thread = (input_width - 1) / 2 + 1;
      } else if (input_width > 512 && input_width <= 1024) {
        thread = input_width;
      }
      blocks = std::min(std::max(thread / input_width, 1), input_height);
      threads = dim3(std::min(input_width, thread), blocks, 1);
      grid = dim3(input_channels, batch_size, 1);
    } else {
      blocks = std::min(
          std::max(thread / input_channels, 1),
          ((input_width + dilate_width - 1) / dilate_width) * dilate_width);
      threads = dim3(std::min(input_channels, thread), blocks, 1);
      grid = dim3((input_height + dilate_height - 1) / dilate_height,
                  dilate_height, batch_size);
    }
    int filter_multiplier = output_channels / input_channels;

#define check_case(c_filter_multiplier, c_stride, c_filter)               \
  if (c_filter_multiplier == 0 ||                                         \
      filter_multiplier == c_filter_multiplier &&                         \
          stride_height == stride_width && stride_height == c_stride &&   \
          (ksize_height == ksize_width && ksize_height == c_filter ||     \
           c_filter == -1)) {                                             \
    if (data_layout != DataLayout::kNHWC) {                               \
      KernelDepthwiseConvInputGradSp<                                     \
          T, c_filter_multiplier, c_stride, c_filter, DataLayout::kNCHW,  \
          fuse_relu_before_conv><<<grid, threads, 0, context.stream()>>>( \
          input_data, output_grad_data, filter_data, batch_size,          \
          output_channels, output_height, output_width, input_channels,   \
          input_height, input_width, filter_multiplier, ksize_height,     \
          ksize_width, stride_height, stride_width, padding_height,       \
          padding_width, dilate_height, dilate_width, input_grad_data);   \
    } else {                                                              \
      KernelDepthwiseConvInputGradSp<                                     \
          T, c_filter_multiplier, c_stride, c_filter, DataLayout::kNHWC,  \
          fuse_relu_before_conv><<<grid, threads, 0, context.stream()>>>( \
          input_data, output_grad_data, filter_data, batch_size,          \
          output_channels, output_height, output_width, input_channels,   \
          input_height, input_width, filter_multiplier, ksize_height,     \
          ksize_width, stride_height, stride_width, padding_height,       \
          padding_width, dilate_height, dilate_width, input_grad_data);   \
    }                                                                     \
    return;                                                               \
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

template <typename T, bool fuse_relu_before_conv>
class DepthwiseConvFilterGradFunctor<platform::CUDADeviceContext, T,
                                     fuse_relu_before_conv> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& output_grad,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::vector<int>& dilations,
                  framework::Tensor* filter_grad,
                  const DataLayout data_layout = DataLayout::kNCHW) {
    const int batch_size = input.dims()[0];
    const int input_channels =
        (data_layout != DataLayout::kNHWC ? input.dims()[1] : input.dims()[3]);
    const int input_height =
        (data_layout != DataLayout::kNHWC ? input.dims()[2] : input.dims()[1]);
    const int input_width =
        (data_layout != DataLayout::kNHWC ? input.dims()[3] : input.dims()[2]);
    const int output_channels =
        (data_layout != DataLayout::kNHWC ? output_grad.dims()[1]
                                          : output_grad.dims()[3]);
    const int output_height =
        (data_layout != DataLayout::kNHWC ? output_grad.dims()[2]
                                          : output_grad.dims()[1]);
    const int output_width =
        (data_layout != DataLayout::kNHWC ? output_grad.dims()[3]
                                          : output_grad.dims()[2]);
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
    int blocks;
    dim3 threads;
    dim3 grid;
    if (data_layout != DataLayout::kNHWC) {
      if (output_width > 1024 && output_width <= 2048) {
        block_size = (output_width - 1) / 2 + 1;
      } else if (output_width > 512 && output_width <= 1024) {
        block_size = output_width;
      }
      blocks = std::min(std::max(block_size / output_width, 1), output_height);
      grid = dim3(ksize_width, ksize_height, output_channels);
      threads = dim3(std::min(output_width, block_size), blocks, 1);
    } else {
      blocks = std::min(
          std::max(block_size / output_channels, 1),
          ((output_width + dilate_width - 1) / dilate_width) * dilate_width);
      grid = dim3((output_height + dilate_height - 1) / dilate_height,
                  dilate_height, batch_size);
      threads = dim3(std::min(output_channels, block_size), blocks, 1);
    }
    int filter_multiplier = output_channels / input_channels;

#define check_case(c_filter_multiplier, c_stride, c_filter)                    \
  if (c_filter_multiplier == 0 ||                                              \
      filter_multiplier == c_filter_multiplier &&                              \
          stride_height == stride_width && stride_height == c_stride &&        \
          (ksize_height == ksize_width && ksize_height == c_filter ||          \
           c_filter == -1)) {                                                  \
    if (data_layout != DataLayout::kNHWC) {                                    \
      KernelDepthwiseConvFilterGradSp<                                         \
          T, c_filter_multiplier, c_stride, c_filter, DataLayout::kNCHW,       \
          fuse_relu_before_conv><<<grid, threads, 0, context.stream()>>>(      \
          output_grad_data, input_data, batch_size, output_channels,           \
          output_height, output_width, input_channels, input_height,           \
          input_width, filter_multiplier, ksize_height, ksize_width,           \
          stride_height, stride_width, padding_height, padding_width,          \
          dilate_height, dilate_width, filter_grad_data);                      \
    } else {                                                                   \
      framework::Tensor filter_grad_hwc;                                       \
      if (c_filter != -1) {                                                    \
        framework::DDim filter_grad_hwc_dims(                                  \
            {filter_grad->dims()[2], filter_grad->dims()[3],                   \
             filter_grad->dims()[0], filter_grad->dims()[1]});                 \
        filter_grad_hwc.Resize(filter_grad_hwc_dims);                          \
        filter_grad_hwc.mutable_data<T>(context.GetPlace());                   \
        math::SetConstant<platform::CUDADeviceContext, T> set_zero;            \
        set_zero(context, &filter_grad_hwc, static_cast<T>(0));                \
        filter_grad_data = filter_grad_hwc.data<T>();                          \
      } else {                                                                 \
        block_size = 512;                                                      \
        if (output_channels > 1024 && output_channels <= 2048) {               \
          block_size = (output_channels - 1) / 2 + 1;                          \
        } else if (output_channels > 512 && output_channels <= 1024) {         \
          block_size = output_channels;                                        \
        }                                                                      \
        blocks =                                                               \
            std::min(std::max(block_size / output_channels, 1), output_width); \
        grid = dim3(ksize_width * ksize_height, output_height, batch_size);    \
        threads = dim3(std::min(output_channels, block_size), blocks, 1);      \
      }                                                                        \
      KernelDepthwiseConvFilterGradSp<                                         \
          T, c_filter_multiplier, c_stride, c_filter, DataLayout::kNHWC,       \
          fuse_relu_before_conv><<<grid, threads, 0, context.stream()>>>(      \
          output_grad_data, input_data, batch_size, output_channels,           \
          output_height, output_width, input_channels, input_height,           \
          input_width, filter_multiplier, ksize_height, ksize_width,           \
          stride_height, stride_width, padding_height, padding_width,          \
          dilate_height, dilate_width, filter_grad_data);                      \
      if (c_filter != -1) {                                                    \
        std::vector<int> perm_axis({2, 3, 0, 1});                              \
        math::TransposeNormal<platform::CUDADeviceContext, T> trans;           \
        trans(context, filter_grad_hwc, filter_grad, perm_axis);               \
      }                                                                        \
    }                                                                          \
    return;                                                                    \
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
#undef check_case
  }
};

template class DepthwiseConvFunctor<platform::CUDADeviceContext, float, false>;
template class DepthwiseConvFunctor<platform::CUDADeviceContext, double, false>;

template class DepthwiseConvInputGradFunctor<platform::CUDADeviceContext, float,
                                             false>;
template class DepthwiseConvInputGradFunctor<platform::CUDADeviceContext,
                                             double, false>;

template class DepthwiseConvFilterGradFunctor<platform::CUDADeviceContext,
                                              float, false>;
template class DepthwiseConvFilterGradFunctor<platform::CUDADeviceContext,
                                              double, false>;

template class DepthwiseConvFunctor<platform::CUDADeviceContext, float, true>;
template class DepthwiseConvFunctor<platform::CUDADeviceContext, double, true>;

template class DepthwiseConvInputGradFunctor<platform::CUDADeviceContext, float,
                                             true>;
template class DepthwiseConvInputGradFunctor<platform::CUDADeviceContext,
                                             double, true>;

template class DepthwiseConvFilterGradFunctor<platform::CUDADeviceContext,
                                              float, true>;
template class DepthwiseConvFilterGradFunctor<platform::CUDADeviceContext,
                                              double, true>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
