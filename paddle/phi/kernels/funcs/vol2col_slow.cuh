/* Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <vector>

#include "paddle/common/errors.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/im2col_slow.cuh"

#include "paddle/phi/backends/gpu/gpu_context.h"

namespace phi {
namespace funcs {

#if defined(__CUDACC__) || defined(__HIPCC__)

template <typename T>
__global__ void Vol2colKernel(const int64_t n,
                              const T* data_vol,
                              const int depth,
                              const int height,
                              const int width,
                              const int ksize_t,
                              const int ksize_h,
                              const int ksize_w,
                              const int pad_t,
                              const int pad_h,
                              const int pad_w,
                              const int stride_t,
                              const int stride_h,
                              const int stride_w,
                              const int dilation_t,
                              const int dilation_h,
                              const int dilation_w,
                              const int depth_col,
                              const int height_col,
                              const int width_col,
                              T* data_col) {
  CUDA_KERNEL_LOOP_TYPE(index, n, int64_t) {
    auto w_out = index % width_col;
    index /= width_col;
    auto h_out = index % height_col;
    index /= height_col;
    auto t_out = index % depth_col;
    auto channel_in = index / depth_col;
    auto channel_out = channel_in * ksize_t * ksize_h * ksize_w;
    auto t_in = t_out * stride_t - pad_t;
    auto h_in = h_out * stride_h - pad_h;
    auto w_in = w_out * stride_w - pad_w;
    data_col +=
        ((channel_out * depth_col + t_out) * height_col + h_out) * width_col +
        w_out;
    data_vol += ((channel_in * depth + t_in) * height + h_in) * width + w_in;
    for (int i = 0; i < ksize_t; ++i) {
      for (int j = 0; j < ksize_h; ++j) {
        for (int k = 0; k < ksize_w; ++k) {
          auto t = t_in + i * dilation_t;
          auto h = h_in + j * dilation_h;
          auto w = w_in + k * dilation_w;
          *data_col = (t >= 0 && h >= 0 && w >= 0 && t < depth && h < height &&
                       w < width)
                          ? data_vol[i * dilation_t * height * width +
                                     j * dilation_h * width + k * dilation_w]
                          : static_cast<T>(0);
          data_col += depth_col * height_col * width_col;
        }
      }
    }
  }
}

template <typename T, typename accT>
__global__ void Vol2imKernel(const int64_t n,
                             const T* data_col,
                             const unsigned depth,
                             const unsigned height,
                             const unsigned width,
                             const unsigned channels,
                             const unsigned kernel_t,
                             const unsigned kernel_h,
                             const unsigned kernel_w,
                             const unsigned pad_t,
                             const unsigned pad_h,
                             const unsigned pad_w,
                             const unsigned stride_t,
                             const unsigned stride_h,
                             const unsigned stride_w,
                             const unsigned dilation_t,
                             const unsigned dilation_h,
                             const unsigned dilation_w,
                             const unsigned depth_col,
                             const unsigned height_col,
                             const unsigned width_col,
                             T* data_vol) {
  CUDA_KERNEL_LOOP(index, n) {
    accT val = static_cast<accT>(0);
    const auto w_im = index % width + pad_w;
    const auto h_im = (index / width) % height + pad_h;
    const auto t_im = (index / width / height) % depth + pad_t;
    const auto c_im = index / (width * height * depth);
    auto kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    auto kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    auto kernel_extent_t = (kernel_t - 1) * dilation_t + 1;
    const auto w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const auto w_col_end = std::min(w_im / stride_w + 1, width_col);
    const auto h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const auto h_col_end = std::min(h_im / stride_h + 1, height_col);
    const auto t_col_start =
        (t_im < kernel_extent_t) ? 0 : (t_im - kernel_extent_t) / stride_t + 1;
    const auto t_col_end = std::min(t_im / stride_t + 1, depth_col);
    for (unsigned t_col = t_col_start; t_col < t_col_end; t_col += 1) {
      for (unsigned h_col = h_col_start; h_col < h_col_end; h_col += 1) {
        for (unsigned w_col = w_col_start; w_col < w_col_end; w_col += 1) {
          uint64_t t_k = (t_im - t_col * stride_t);
          uint64_t h_k = (h_im - h_col * stride_h);
          uint64_t w_k = (w_im - w_col * stride_w);
          if (t_k % dilation_t == 0 && h_k % dilation_h == 0 &&
              w_k % dilation_w == 0) {
            t_k /= dilation_t;
            h_k /= dilation_h;
            w_k /= dilation_w;
            const int64_t idx_k =
                ((c_im * kernel_t + t_k) * kernel_h + h_k) * kernel_w + w_k;
            const int64_t data_col_index =
                ((idx_k * depth_col + t_col) * height_col + h_col) * width_col +
                w_col;
            val += data_col[data_col_index];
          }
        }
      }
    }
    data_vol[index] = static_cast<T>(val);
  }
}

template <typename T, typename Context>
void vol2col_slow(const Context& dev_ctx,
                  const T* data_vol,
                  const int channels,
                  const int depth,
                  const int height,
                  const int width,
                  const int depth_col,
                  const int height_col,
                  const int width_col,
                  const int ksize_t,
                  const int ksize_h,
                  const int ksize_w,
                  const int pad_t,
                  const int pad_h,
                  const int pad_w,
                  const int stride_t,
                  const int stride_h,
                  const int stride_w,
                  const int dilation_t,
                  const int dilation_h,
                  const int dilation_w,
                  T* data_col) {
  auto stream = dev_ctx.stream();
  const auto num_kernels =
      static_cast<int64_t>(channels) * depth_col * height_col * width_col;
  Vol2colKernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
      num_kernels,
      data_vol,
      depth,
      height,
      width,
      ksize_t,
      ksize_h,
      ksize_w,
      pad_t,
      pad_h,
      pad_w,
      stride_t,
      stride_h,
      stride_w,
      dilation_t,
      dilation_h,
      dilation_w,
      depth_col,
      height_col,
      width_col,
      data_col);
}

template <typename T, typename accT, typename Context>
void col2vol_slow(const Context& dev_ctx,
                  const T* data_col,
                  const int64_t channels,
                  const int64_t depth,
                  const int64_t height,
                  const int64_t width,
                  const int64_t output_depth,
                  const int64_t output_height,
                  const int64_t output_width,
                  const int64_t patch_t,
                  const int64_t patch_h,
                  const int64_t patch_w,
                  const int64_t pad_t,
                  const int64_t pad_h,
                  const int64_t pad_w,
                  const int64_t stride_t,
                  const int64_t stride_h,
                  const int64_t stride_w,
                  const int64_t dilation_t,
                  const int64_t dilation_h,
                  const int64_t dilation_w,
                  T* data_vol) {
  auto stream = dev_ctx.stream();
  const auto num_kernels = channels * depth * height * width;
  Vol2imKernel<T, accT>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(num_kernels,
                                                                 data_col,
                                                                 depth,
                                                                 height,
                                                                 width,
                                                                 channels,
                                                                 patch_t,
                                                                 patch_h,
                                                                 patch_w,
                                                                 pad_t,
                                                                 pad_h,
                                                                 pad_w,
                                                                 stride_t,
                                                                 stride_h,
                                                                 stride_w,
                                                                 dilation_t,
                                                                 dilation_h,
                                                                 dilation_w,
                                                                 output_depth,
                                                                 output_height,
                                                                 output_width,
                                                                 data_vol);
}
#endif  // __CUDACC__

template <typename T>
void vol2col_slow(cudaStream_t stream,
                  const T* data_vol,
                  const int channels,
                  const int depth,
                  const int height,
                  const int width,
                  const int depth_col,
                  const int height_col,
                  const int width_col,
                  const int ksize_t,
                  const int ksize_h,
                  const int ksize_w,
                  const int pad_t,
                  const int pad_h,
                  const int pad_w,
                  const int stride_t,
                  const int stride_h,
                  const int stride_w,
                  const int dilation_t,
                  const int dilation_h,
                  const int dilation_w,
                  T* data_col);

template <typename T, typename accT>
void col2vol_slow(cudaStream_t stream,
                  const T* data_col,
                  const int64_t channels,
                  const int64_t depth,
                  const int64_t height,
                  const int64_t width,
                  const int64_t output_depth,
                  const int64_t output_height,
                  const int64_t output_width,
                  const int64_t patch_t,
                  const int64_t patch_h,
                  const int64_t patch_w,
                  const int64_t pad_t,
                  const int64_t pad_h,
                  const int64_t pad_w,
                  const int64_t stride_t,
                  const int64_t stride_h,
                  const int64_t stride_w,
                  const int64_t dilation_t,
                  const int64_t dilation_h,
                  const int64_t dilation_w,
                  T* data_vol);
}  // namespace funcs
}  // namespace phi
