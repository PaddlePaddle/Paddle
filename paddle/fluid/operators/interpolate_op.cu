/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
#include <string>
#include "paddle/fluid/operators/interpolate_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

using framework::Tensor;
using DataLayout = framework::DataLayout;

template <typename T>
__global__ void KeNearestNeighborInterpFw(
    const T* in, const size_t in_img_h, const size_t in_img_w,
    const size_t input_h, const size_t input_w, T* out, const size_t out_img_h,
    const size_t out_img_w, const size_t output_h, const size_t output_w,
    const size_t num_channels, const float ratio_h, const float ratio_w,
    const bool align_corners, const DataLayout data_layout) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < nthreads; tid += stride) {
    int out_id_h = tid / output_w;
    int out_id_w = tid % output_w;
    int in_img_size = input_w / num_channels;
    int out_img_size = output_w / num_channels;

    int channel_id, out_img_idy, out_img_idx;
    if (data_layout == DataLayout::kNCHW) {
      channel_id = out_id_w / out_img_size;
      out_img_idy = (out_id_w % out_img_size) / out_img_w;
      out_img_idx = tid % out_img_w;
    } else {
      out_img_idy = out_id_w / (out_img_w * num_channels);
      out_img_idx = out_id_w % (out_img_w * num_channels) / num_channels;
      channel_id = tid % num_channels;
    }

    int in_img_idy = (align_corners)
                         ? static_cast<int>(ratio_h * out_img_idy + 0.5)
                         : static_cast<int>(ratio_h * out_img_idy);
    int in_img_idx = (align_corners)
                         ? static_cast<int>(ratio_w * out_img_idx + 0.5)
                         : static_cast<int>(ratio_w * out_img_idx);

    if (data_layout == DataLayout::kNCHW) {
      out[tid] = in[out_id_h * input_w + channel_id * in_img_size +
                    in_img_idy * in_img_w + in_img_idx];
    } else {
      out[tid] = in[out_id_h * input_w + in_img_idy * in_img_w * num_channels +
                    in_img_idx * num_channels + channel_id];
    }
  }
}

template <typename T>
__global__ void KeNearestNeighborInterpBw(
    T* in, const size_t in_img_h, const size_t in_img_w, const size_t input_h,
    const size_t input_w, const T* out, const size_t out_img_h,
    const size_t out_img_w, const size_t output_h, const size_t output_w,
    const size_t num_channels, const float ratio_h, const float ratio_w,
    const bool align_corners, const DataLayout data_layout) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < nthreads; tid += stride) {
    int out_id_h = tid / output_w;
    int out_id_w = tid % output_w;
    int in_img_size = input_w / num_channels;
    int out_img_size = output_w / num_channels;

    int channel_id, out_img_idy, out_img_idx;
    if (data_layout == DataLayout::kNCHW) {
      channel_id = out_id_w / out_img_size;
      out_img_idy = (out_id_w % out_img_size) / out_img_w;
      out_img_idx = tid % out_img_w;
    } else {
      out_img_idy = out_id_w / (out_img_w * num_channels);
      out_img_idx = out_id_w % (out_img_w * num_channels) / num_channels;
      channel_id = tid % num_channels;
    }

    int in_img_idy = (align_corners)
                         ? static_cast<int>(ratio_h * out_img_idy + 0.5)
                         : static_cast<int>(ratio_h * out_img_idy);
    int in_img_idx = (align_corners)
                         ? static_cast<int>(ratio_w * out_img_idx + 0.5)
                         : static_cast<int>(ratio_w * out_img_idx);

    T* in_pos;
    if (data_layout == DataLayout::kNCHW) {
      in_pos = &in[out_id_h * input_w + channel_id * in_img_size +
                   in_img_idy * in_img_w + in_img_idx];
    } else {
      in_pos = &in[out_id_h * input_w + in_img_idy * in_img_w * num_channels +
                   in_img_idx * num_channels + channel_id];
    }
    const T out_pos = out[out_id_h * output_w + out_id_w];
    platform::CudaAtomicAdd(in_pos, out_pos);
  }
}

template <typename T>
__global__ void KeLinearInterpFw(const T* in, const size_t in_img_w,
                                 const size_t input_w, T* out,
                                 const size_t out_img_w, const size_t output_h,
                                 const size_t output_w,
                                 const size_t num_channels, const float ratio_w,
                                 const bool align_corners, const int align_mode,
                                 const DataLayout data_layout) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  bool align_flag = (align_mode == 0 && !align_corners);
  for (; tid < nthreads; tid += stride) {
    int out_id_h = tid / output_w;
    int out_id_w = tid % output_w;
    int in_img_size = input_w / num_channels;
    int out_img_size = output_w / num_channels;

    int channel_id, out_img_idy, out_img_idx;
    if (data_layout == DataLayout::kNCHW) {
      channel_id = out_id_w / out_img_size;
      out_img_idx = tid % out_img_w;
    } else {
      out_img_idx = out_id_w % (out_img_w * num_channels) / num_channels;
      channel_id = tid % num_channels;
    }

    int in_img_idx = align_flag
                         ? static_cast<int>(ratio_w * (out_img_idx + 0.5) - 0.5)
                         : static_cast<int>(ratio_w * out_img_idx);
    in_img_idx = (in_img_idx > 0) ? in_img_idx : 0;  // w
    int w_id = (in_img_idx < in_img_w - 1) ? 1 : 0;  // w_id

    T src_w = ratio_w * (out_img_idx + 0.5) - 0.5;
    src_w = (src_w > 0) ? src_w : 0;
    T w1lambda =
        align_flag ? src_w - in_img_idx : ratio_w * out_img_idx - in_img_idx;
    T w2lambda = 1.f - w1lambda;

    if (data_layout == DataLayout::kNCHW) {
      const T* in_pos =
          &in[out_id_h * out_id_w + channel_id * in_img_size + in_img_idx];
      // linear interpolation
      out[out_id_h * output_w + out_id_w] =
          w2lambda * in_pos[0] + w1lambda * in_pos[w_id];

    } else {
      const T* in_pos =
          &in[out_id_h * input_w + in_img_idx * num_channels + channel_id];
      // linear interpolation
      out[out_id_h * output_w + out_id_w] =
          w2lambda * in_pos[0] + w1lambda * in_pos[w_id * num_channels];
    }
  }
}

template <typename T>
__global__ void KeLinearInterpBw(T* in, const size_t in_img_w,
                                 const size_t input_w, const T* out,
                                 const size_t out_img_w, const size_t output_h,
                                 const size_t output_w,
                                 const size_t num_channels, const T ratio_w,
                                 const bool align_corners, const int align_mode,
                                 const DataLayout data_layout) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  bool align_flag = (align_mode == 0 && !align_corners);
  for (; tid < nthreads; tid += stride) {
    int out_id_h = tid / output_w;
    int out_id_w = tid % output_w;
    int in_img_size = input_w / num_channels;
    int out_img_size = output_w / num_channels;

    int channel_id, out_img_idx;
    if (data_layout == DataLayout::kNCHW) {
      channel_id = out_id_w / out_img_size;
      out_img_idx = tid % out_img_w;
    } else {
      out_img_idx = out_id_w % (out_img_w * num_channels) / num_channels;
      channel_id = tid % num_channels;
    }

    int in_img_idx = align_flag ? ratio_w * (out_img_idx + 0.5) - 0.5
                                : ratio_w * out_img_idx;
    in_img_idx = (in_img_idx > 0) ? in_img_idx : 0;  // w
    int w_id = (in_img_idx < in_img_w - 1) ? 1 : 0;  // w_id

    T src_w = ratio_w * (out_img_idx + 0.5) - 0.5;
    src_w = (src_w > 0) ? src_w : 0;
    T w1lambda =
        align_flag ? src_w - in_img_idx : ratio_w * out_img_idx - in_img_idx;
    T w2lambda = 1.f - w1lambda;

    T* in_pos;
    if (data_layout == DataLayout::kNCHW) {
      in_pos = &in[out_id_h * input_w + channel_id * in_img_size + in_img_idx];
    } else {
      in_pos = &in[out_id_h * input_w + in_img_idx * num_channels + channel_id];
    }
    const T* out_pos = &out[out_id_w];

    if (data_layout == DataLayout::kNCHW) {
      platform::CudaAtomicAdd(&in_pos[0], w2lambda * out_pos[0]);
      platform::CudaAtomicAdd(&in_pos[w_id], w1lambda * out_pos[0]);
    } else {
      platform::CudaAtomicAdd(&in_pos[0], w2lambda * out_pos[0]);
      platform::CudaAtomicAdd(&in_pos[w_id * num_channels],
                              w1lambda * out_pos[0]);
    }
  }
}

template <typename T>
__global__ void KeBilinearInterpFw(
    const T* in, const size_t in_img_h, const size_t in_img_w,
    const size_t input_h, const size_t input_w, T* out, const size_t out_img_h,
    const size_t out_img_w, const size_t output_h, const size_t output_w,
    const size_t num_channels, const float ratio_h, const float ratio_w,
    const bool align_corners, const int align_mode,
    const DataLayout data_layout) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  bool align_flag = (align_mode == 0 && !align_corners);
  for (; tid < nthreads; tid += stride) {
    int out_id_h = tid / output_w;
    int out_id_w = tid % output_w;
    int in_img_size = input_w / num_channels;
    int out_img_size = output_w / num_channels;

    int channel_id, out_img_idy, out_img_idx;
    if (data_layout == DataLayout::kNCHW) {
      channel_id = out_id_w / out_img_size;
      out_img_idy = (out_id_w % out_img_size) / out_img_w;
      out_img_idx = tid % out_img_w;
    } else {
      out_img_idy = out_id_w / (out_img_w * num_channels);
      out_img_idx = out_id_w % (out_img_w * num_channels) / num_channels;
      channel_id = tid % num_channels;
    }

    int in_img_idy = align_flag
                         ? static_cast<int>(ratio_h * (out_img_idy + 0.5) - 0.5)
                         : static_cast<int>(ratio_h * out_img_idy);
    in_img_idy = (in_img_idy > 0) ? in_img_idy : 0;
    int h_id = (in_img_idy < in_img_h - 1) ? 1 : 0;
    T src_h = ratio_h * (out_img_idy + 0.5) - 0.5;
    src_h = (src_h > 0) ? src_h : 0;
    T h1lambda =
        align_flag ? src_h - in_img_idy : ratio_h * out_img_idy - in_img_idy;
    T h2lambda = 1.f - h1lambda;

    int in_img_idx = align_flag
                         ? static_cast<int>(ratio_w * (out_img_idx + 0.5) - 0.5)
                         : static_cast<int>(ratio_w * out_img_idx);
    in_img_idx = (in_img_idx > 0) ? in_img_idx : 0;
    int w_id = (in_img_idx < in_img_w - 1) ? 1 : 0;
    T src_w = ratio_w * (out_img_idx + 0.5) - 0.5;
    src_w = (src_w > 0) ? src_w : 0;
    T w1lambda =
        align_flag ? src_w - in_img_idx : ratio_w * out_img_idx - in_img_idx;
    T w2lambda = 1.f - w1lambda;

    if (data_layout == DataLayout::kNCHW) {
      const T* in_pos = &in[out_id_h * input_w + channel_id * in_img_size +
                            in_img_idy * in_img_w + in_img_idx];

      // bilinear interpolation
      out[out_id_h * output_w + out_id_w] =
          h2lambda * (w2lambda * in_pos[0] + w1lambda * in_pos[w_id]) +
          h1lambda * (w2lambda * in_pos[h_id * in_img_w] +
                      w1lambda * in_pos[h_id * in_img_w + w_id]);
    } else {
      const T* in_pos =
          &in[out_id_h * input_w + in_img_idy * in_img_w * num_channels +
              in_img_idx * num_channels + channel_id];

      // bilinear interpolation
      out[out_id_h * output_w + out_id_w] =
          h2lambda *
              (w2lambda * in_pos[0] + w1lambda * in_pos[w_id * num_channels]) +
          h1lambda * (w2lambda * in_pos[h_id * in_img_w * num_channels] +
                      w1lambda * in_pos[h_id * in_img_w * num_channels +
                                        w_id * num_channels]);
    }
  }
}

template <typename T>
__global__ void KeBilinearInterpBw(
    T* in, const size_t in_img_h, const size_t in_img_w, const size_t input_h,
    const size_t input_w, const T* out, const size_t out_img_h,
    const size_t out_img_w, const size_t output_h, const size_t output_w,
    const size_t num_channels, const T ratio_h, const T ratio_w,
    const bool align_corners, const int align_mode,
    const DataLayout data_layout) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  bool align_flag = (align_mode == 0 && !align_corners);
  for (; tid < nthreads; tid += stride) {
    int out_id_h = tid / output_w;
    int out_id_w = tid % output_w;
    int in_img_size = input_w / num_channels;
    int out_img_size = output_w / num_channels;

    int channel_id, out_img_idy, out_img_idx;
    if (data_layout == DataLayout::kNCHW) {
      channel_id = out_id_w / out_img_size;
      out_img_idy = (out_id_w % out_img_size) / out_img_w;
      out_img_idx = tid % out_img_w;
    } else {
      out_img_idy = out_id_w / (out_img_w * num_channels);
      out_img_idx = out_id_w % (out_img_w * num_channels) / num_channels;
      channel_id = tid % num_channels;
    }

    int in_img_idy = align_flag ? ratio_h * (out_img_idy + 0.5) - 0.5
                                : ratio_h * out_img_idy;
    in_img_idy = (in_img_idy > 0) ? in_img_idy : 0;
    int h_id = (in_img_idy < in_img_h - 1) ? 1 : 0;
    T src_h = ratio_h * (out_img_idy + 0.5) - 0.5;
    src_h = (src_h > 0) ? src_h : 0;
    T h1lambda =
        align_flag ? src_h - in_img_idy : ratio_h * out_img_idy - in_img_idy;
    T h2lambda = 1.f - h1lambda;

    int in_img_idx = align_flag ? ratio_w * (out_img_idx + 0.5) - 0.5
                                : ratio_w * out_img_idx;
    in_img_idx = (in_img_idx > 0) ? in_img_idx : 0;
    int w_id = (in_img_idx < in_img_w - 1) ? 1 : 0;
    T src_w = ratio_w * (out_img_idx + 0.5) - 0.5;
    src_w = (src_w > 0) ? src_w : 0;
    T w1lambda =
        align_flag ? src_w - in_img_idx : ratio_w * out_img_idx - in_img_idx;
    T w2lambda = 1.f - w1lambda;

    T* in_pos;
    if (data_layout == DataLayout::kNCHW) {
      in_pos = &in[out_id_h * input_w + channel_id * in_img_size +
                   in_img_idy * in_img_w + in_img_idx];
    } else {
      in_pos = &in[out_id_h * input_w + in_img_idy * in_img_w * num_channels +
                   in_img_idx * num_channels + channel_id];
    }

    const T* out_pos = &out[out_id_h * output_w + out_id_w];

    if (data_layout == DataLayout::kNCHW) {
      platform::CudaAtomicAdd(&in_pos[0], h2lambda * w2lambda * out_pos[0]);
      platform::CudaAtomicAdd(&in_pos[w_id], h2lambda * w1lambda * out_pos[0]);
      platform::CudaAtomicAdd(&in_pos[h_id * in_img_w],
                              h1lambda * w2lambda * out_pos[0]);
      platform::CudaAtomicAdd(&in_pos[h_id * in_img_w + w_id],
                              h1lambda * w1lambda * out_pos[0]);
    } else {
      platform::CudaAtomicAdd(&in_pos[0], h2lambda * w2lambda * out_pos[0]);
      platform::CudaAtomicAdd(&in_pos[w_id * num_channels],
                              h2lambda * w1lambda * out_pos[0]);
      platform::CudaAtomicAdd(&in_pos[h_id * in_img_w * num_channels],
                              h1lambda * w2lambda * out_pos[0]);
      platform::CudaAtomicAdd(
          &in_pos[h_id * in_img_w * num_channels + w_id * num_channels],
          h1lambda * w1lambda * out_pos[0]);
    }
  }
}

template <typename T>
__global__ void KeTrilinearInterpFw(
    const T* in, const size_t in_img_d, const size_t in_img_h,
    const size_t in_img_w, const size_t input_h, const size_t input_w, T* out,
    const size_t out_img_d, const size_t out_img_h, const size_t out_img_w,
    const size_t output_h, const size_t output_w, const size_t num_channels,
    const float ratio_d, const float ratio_h, const float ratio_w,
    const bool align_corners, const int align_mode,
    const DataLayout data_layout) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  bool align_flag = (align_mode == 0 && !align_corners);
  for (; tid < nthreads; tid += stride) {
    int out_id_h = tid / output_w;
    int out_id_w = tid % output_w;
    int in_img_size = input_w / num_channels;
    int out_img_size = output_w / num_channels;

    int channel_id, out_img_idt, out_img_idy, out_img_idx;
    if (data_layout == DataLayout::kNCHW) {
      channel_id = out_id_w / out_img_size;
      out_img_idt = (out_id_w % out_img_size) / out_img_h / out_img_w;
      out_img_idy = ((out_id_w % out_img_size) / out_img_w) % out_img_h;
      out_img_idx = tid % out_img_w;
    } else {
      out_img_idt = out_id_w / (out_img_h * out_img_w * num_channels);
      out_img_idy = out_id_w % (out_img_h * out_img_w * num_channels) /
                    (out_img_w * num_channels);
      out_img_idx = out_id_w % (out_img_w * num_channels) / num_channels;
      channel_id = tid % num_channels;
    }

    int in_img_idt = align_flag
                         ? static_cast<int>(ratio_d * (out_img_idt + 0.5) - 0.5)
                         : static_cast<int>(ratio_d * out_img_idt);
    in_img_idt = (in_img_idt > 0) ? in_img_idt : 0;
    int d_id = (in_img_idt < in_img_d - 1) ? 1 : 0;
    T src_d = ratio_d * (out_img_idt + 0.5) - 0.5;
    src_d = (src_d > 0) ? src_d : 0;
    T d1lambda =
        align_flag ? src_d - in_img_idt : ratio_d * out_img_idt - in_img_idt;
    T d2lambda = 1.f - d1lambda;

    int in_img_idy = align_flag
                         ? static_cast<int>(ratio_h * (out_img_idy + 0.5) - 0.5)
                         : static_cast<int>(ratio_h * out_img_idy);
    in_img_idy = (in_img_idy > 0) ? in_img_idy : 0;
    int h_id = (in_img_idy < in_img_h - 1) ? 1 : 0;
    T src_h = ratio_h * (out_img_idy + 0.5) - 0.5;
    src_h = (src_h > 0) ? src_h : 0;
    T h1lambda =
        align_flag ? src_h - in_img_idy : ratio_h * out_img_idy - in_img_idy;
    T h2lambda = 1.f - h1lambda;

    int in_img_idx = align_flag
                         ? static_cast<int>(ratio_w * (out_img_idx + 0.5) - 0.5)
                         : static_cast<int>(ratio_w * out_img_idx);
    in_img_idx = (in_img_idx > 0) ? in_img_idx : 0;
    int w_id = (in_img_idx < in_img_w - 1) ? 1 : 0;
    T src_w = ratio_w * (out_img_idx + 0.5) - 0.5;
    src_w = (src_w > 0) ? src_w : 0;
    T w1lambda =
        align_flag ? src_w - in_img_idx : ratio_w * out_img_idx - in_img_idx;
    T w2lambda = 1.f - w1lambda;

    if (data_layout == DataLayout::kNCHW) {
      int in_pos1_idx = out_id_h * input_w + channel_id * in_img_size +
                        (in_img_idt * in_img_h + in_img_idy) * in_img_w +
                        in_img_idx;
      const T* in_pos1 = &in[in_pos1_idx];
      int in_pos2_idx = in_pos1_idx + d_id * in_img_h * in_img_w;
      const T* in_pos2 = &in[in_pos2_idx];

      // trilinear interpolation
      out[out_id_h * output_w + out_id_w] =
          d2lambda *
              (h2lambda * (w2lambda * in_pos1[0] + w1lambda * in_pos1[w_id]) +
               h1lambda * (w2lambda * in_pos1[h_id * in_img_w] +
                           w1lambda * in_pos1[h_id * in_img_w + w_id])) +
          d1lambda *
              (h2lambda * (w2lambda * in_pos2[0] + w1lambda * in_pos2[w_id]) +
               h1lambda * (w2lambda * in_pos2[h_id * in_img_w] +
                           w1lambda * in_pos2[h_id * in_img_w + w_id]));

    } else {
      int in_pos1_idx = out_id_h * input_w +
                        in_img_idt * in_img_h * in_img_w * num_channels +
                        in_img_idy * in_img_w * num_channels +
                        in_img_idx * num_channels + channel_id;
      const T* in_pos1 = &in[in_pos1_idx];
      int in_pos2_idx = in_pos1_idx + d_id * in_img_h * in_img_w * num_channels;
      const T* in_pos2 = &in[in_pos2_idx];

      // trilinear interpolation
      out[out_id_h * output_w + out_id_w] =
          d2lambda *
              (h2lambda * (w2lambda * in_pos1[0] +
                           w1lambda * in_pos1[w_id * num_channels]) +
               h1lambda * (w2lambda * in_pos1[h_id * in_img_w * num_channels] +
                           w1lambda * in_pos1[h_id * in_img_w * num_channels +
                                              w_id * num_channels])) +
          d1lambda *
              (h2lambda * (w2lambda * in_pos2[0] +
                           w1lambda * in_pos2[w_id * num_channels]) +
               h1lambda * (w2lambda * in_pos2[h_id * in_img_w * num_channels] +
                           w1lambda * in_pos2[h_id * in_img_w * num_channels +
                                              w_id * num_channels]));
    }
  }
}

template <typename T>
__global__ void KeTrilinearInterpBw(
    T* in, const size_t in_img_d, const size_t in_img_h, const size_t in_img_w,
    const size_t input_h, const size_t input_w, const T* out,
    const size_t out_img_d, const size_t out_img_h, const size_t out_img_w,
    const size_t output_h, const size_t output_w, const size_t num_channels,
    const T ratio_d, const T ratio_h, const T ratio_w, const bool align_corners,
    const int align_mode, const DataLayout data_layout) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  bool align_flag = (align_mode == 0 && !align_corners);
  for (; tid < nthreads; tid += stride) {
    int out_id_h = tid / output_w;
    int out_id_w = tid % output_w;
    int in_img_size = input_w / num_channels;
    int out_img_size = output_w / num_channels;

    int channel_id, out_img_idt, out_img_idy, out_img_idx;
    if (data_layout == DataLayout::kNCHW) {
      channel_id = out_id_w / out_img_size;
      out_img_idt = (out_id_w % out_img_size) / out_img_h / out_img_w;
      out_img_idy = ((out_id_w % out_img_size) / out_img_w) % out_img_h;
      out_img_idx = tid % out_img_w;
    } else {
      out_img_idt = out_id_w / (out_img_h * out_img_w * num_channels);
      out_img_idy = out_id_w % (out_img_h * out_img_w * num_channels) /
                    (out_img_w * num_channels);
      out_img_idx = out_id_w % (out_img_w * num_channels) / num_channels;
      channel_id = tid % num_channels;
    }

    int in_img_idt = align_flag
                         ? static_cast<int>(ratio_d * (out_img_idt + 0.5) - 0.5)
                         : static_cast<int>(ratio_d * out_img_idt);
    in_img_idt = (in_img_idt > 0) ? in_img_idt : 0;
    int d_id = (in_img_idt < in_img_d - 1) ? 1 : 0;
    T src_d = ratio_d * (out_img_idt + 0.5) - 0.5;
    src_d = (src_d > 0) ? src_d : 0;
    T d1lambda =
        align_flag ? src_d - in_img_idt : ratio_d * out_img_idt - in_img_idt;
    T d2lambda = 1.f - d1lambda;

    int in_img_idy = align_flag
                         ? static_cast<int>(ratio_h * (out_img_idy + 0.5) - 0.5)
                         : static_cast<int>(ratio_h * out_img_idy);
    in_img_idy = (in_img_idy > 0) ? in_img_idy : 0;
    int h_id = (in_img_idy < in_img_h - 1) ? 1 : 0;
    T src_h = ratio_h * (out_img_idy + 0.5) - 0.5;
    src_h = (src_h > 0) ? src_h : 0;
    T h1lambda =
        align_flag ? src_h - in_img_idy : ratio_h * out_img_idy - in_img_idy;
    T h2lambda = 1.f - h1lambda;

    int in_img_idx = align_flag
                         ? static_cast<int>(ratio_w * (out_img_idx + 0.5) - 0.5)
                         : static_cast<int>(ratio_w * out_img_idx);
    in_img_idx = (in_img_idx > 0) ? in_img_idx : 0;
    int w_id = (in_img_idx < in_img_w - 1) ? 1 : 0;
    T src_w = ratio_w * (out_img_idx + 0.5) - 0.5;
    src_w = (src_w > 0) ? src_w : 0;
    T w1lambda =
        align_flag ? src_w - in_img_idx : ratio_w * out_img_idx - in_img_idx;
    T w2lambda = 1.f - w1lambda;

    if (data_layout == DataLayout::kNCHW) {
      int in_pos1_idx = out_id_h * input_w + channel_id * in_img_size +
                        (in_img_idt * in_img_h + in_img_idy) * in_img_w +
                        in_img_idx;
      T* in_pos1 = &in[in_pos1_idx];
      int in_pos2_idx = in_pos1_idx + d_id * in_img_h * in_img_w;
      T* in_pos2 = &in[in_pos2_idx];

      const T* out_pos = &out[out_id_h * output_w + out_id_w];

      // trilinear interpolation grad
      platform::CudaAtomicAdd(&in_pos1[0],
                              d2lambda * h2lambda * w2lambda * out_pos[0]);
      platform::CudaAtomicAdd(&in_pos1[w_id],
                              d2lambda * h2lambda * w1lambda * out_pos[0]);
      platform::CudaAtomicAdd(&in_pos1[h_id * in_img_w],
                              d2lambda * h1lambda * w2lambda * out_pos[0]);
      platform::CudaAtomicAdd(&in_pos1[h_id * in_img_w + w_id],
                              d2lambda * h1lambda * w1lambda * out_pos[0]);
      platform::CudaAtomicAdd(&in_pos2[0],
                              d1lambda * h2lambda * w2lambda * out_pos[0]);
      platform::CudaAtomicAdd(&in_pos2[w_id],
                              d1lambda * h2lambda * w1lambda * out_pos[0]);
      platform::CudaAtomicAdd(&in_pos2[h_id * in_img_w],
                              d1lambda * h1lambda * w2lambda * out_pos[0]);
      platform::CudaAtomicAdd(&in_pos2[h_id * in_img_w + w_id],
                              d1lambda * h1lambda * w1lambda * out_pos[0]);
    } else {
      int in_pos1_idx = out_id_h * input_w +
                        in_img_idt * in_img_h * in_img_w * num_channels +
                        in_img_idy * in_img_w * num_channels +
                        in_img_idx * num_channels + channel_id;
      T* in_pos1 = &in[in_pos1_idx];
      int in_pos2_idx = in_pos1_idx + d_id * in_img_h * in_img_w * num_channels;
      T* in_pos2 = &in[in_pos2_idx];

      const T* out_pos = &out[out_id_h * output_w + out_id_w];

      // trilinear interpolation grad
      platform::CudaAtomicAdd(&in_pos1[0],
                              d2lambda * h2lambda * w2lambda * out_pos[0]);
      platform::CudaAtomicAdd(&in_pos1[w_id * num_channels],
                              d2lambda * h2lambda * w1lambda * out_pos[0]);
      platform::CudaAtomicAdd(&in_pos1[h_id * in_img_w * num_channels],
                              d2lambda * h1lambda * w2lambda * out_pos[0]);
      platform::CudaAtomicAdd(
          &in_pos1[h_id * in_img_w * num_channels + w_id * num_channels],
          d2lambda * h1lambda * w1lambda * out_pos[0]);
      platform::CudaAtomicAdd(&in_pos2[0],
                              d1lambda * h2lambda * w2lambda * out_pos[0]);
      platform::CudaAtomicAdd(&in_pos2[w_id * num_channels],
                              d1lambda * h2lambda * w1lambda * out_pos[0]);
      platform::CudaAtomicAdd(&in_pos2[h_id * in_img_w * num_channels],
                              d1lambda * h1lambda * w2lambda * out_pos[0]);
      platform::CudaAtomicAdd(
          &in_pos2[h_id * in_img_w * num_channels + w_id * num_channels],
          d1lambda * h1lambda * w1lambda * out_pos[0]);
    }
  }
}

template <typename T>
__device__ __forceinline__ static T Kecubic_interp(const T x0, const T x1,
                                                   const T x2, const T x3,
                                                   T t) {
  T coeffs[4];
  T a = -0.75;
  T x_1 = t;
  T x_2 = 1.0 - t;
  coeffs[0] = cubic_convolution2<T>(x_1 + 1.0, a);
  coeffs[1] = cubic_convolution1<T>(x_1, a);
  coeffs[2] = cubic_convolution1<T>(x_2, a);
  coeffs[3] = cubic_convolution2<T>(x_2 + 1.0, a);
  return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

template <typename T>
__global__ void KeBicubicInterpFw(
    const T* in, const size_t in_img_h, const size_t in_img_w,
    const size_t input_h, const size_t input_w, T* out, const size_t out_img_h,
    const size_t out_img_w, const size_t output_h, const size_t output_w,
    const size_t num_channels, const float ratio_h, const float ratio_w,
    const bool align_corners, const DataLayout data_layout) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (; tid < nthreads; tid += stride) {
    int out_id_h = tid / output_w;
    int out_id_w = tid % output_w;
    int in_img_size = input_w / num_channels;
    int out_img_size = output_w / num_channels;

    int channel_id, out_img_idy, out_img_idx;

    if (data_layout == DataLayout::kNCHW) {
      channel_id = out_id_w / out_img_size;
      out_img_idy = (out_id_w % out_img_size) / out_img_w;
      out_img_idx = tid % out_img_w;
    } else {
      out_img_idy = out_id_w / (out_img_w * num_channels);
      out_img_idx = out_id_w % (out_img_w * num_channels) / num_channels;
      channel_id = tid % num_channels;
    }

    T in_img_idy = align_corners
                       ? static_cast<T>(ratio_h * out_img_idy)
                       : static_cast<T>(ratio_h * (out_img_idy + 0.5) - 0.5);
    int input_y = floorf(in_img_idy);
    const T y_t = in_img_idy - input_y;

    T in_img_idx = align_corners
                       ? static_cast<T>(ratio_w * out_img_idx)
                       : static_cast<T>(ratio_w * (out_img_idx + 0.5) - 0.5);
    int input_x = floorf(in_img_idx);
    const T x_t = in_img_idx - input_x;

    T coefficients[4];
    const T* in_pos_0;
    const T* in_pos_1;
    const T* in_pos_2;
    const T* in_pos_3;
    int access_x_0;
    if (data_layout == DataLayout::kNCHW) {
      for (int k = 0; k < 4; k++) {
        int access_y =
            max(min(input_y - 1 + k, static_cast<int>(in_img_h - 1)), 0);
        access_x_0 = max(min(input_x - 1, static_cast<int>(in_img_w - 1)), 0);
        int access_x_1 =
            max(min(input_x + 0, static_cast<int>(in_img_w - 1)), 0);
        int access_x_2 =
            max(min(input_x + 1, static_cast<int>(in_img_w - 1)), 0);
        int access_x_3 =
            max(min(input_x + 2, static_cast<int>(in_img_w - 1)), 0);

        in_pos_0 = &in[out_id_h * input_w + channel_id * in_img_size +
                       access_y * in_img_w + access_x_0];
        in_pos_1 = &in[out_id_h * input_w + channel_id * in_img_size +
                       access_y * in_img_w + access_x_1];
        in_pos_2 = &in[out_id_h * input_w + channel_id * in_img_size +
                       access_y * in_img_w + access_x_2];
        in_pos_3 = &in[out_id_h * input_w + channel_id * in_img_size +
                       access_y * in_img_w + access_x_3];

        coefficients[k] = Kecubic_interp<T>(in_pos_0[0], in_pos_1[0],
                                            in_pos_2[0], in_pos_3[0], x_t);
      }

      out[out_id_h * output_w + out_id_w] =
          Kecubic_interp<T>(coefficients[0], coefficients[1], coefficients[2],
                            coefficients[3], y_t);

    } else {
      for (int k = 0; k < 4; k++) {
        int access_y =
            max(min(input_y - 1 + k, static_cast<int>((in_img_h - 1))), 0);
        int access_x_0 =
            max(min(input_x - 1, static_cast<int>((in_img_w - 1))), 0);
        int access_x_1 =
            max(min(input_x + 0, static_cast<int>((in_img_w - 1))), 0);
        int access_x_2 =
            max(min(input_x + 1, static_cast<int>((in_img_w - 1))), 0);
        int access_x_3 =
            max(min(input_x + 2, static_cast<int>((in_img_w - 1))), 0);

        const T* in_pos_0 =
            &in[out_id_h * input_w + access_y * in_img_w * num_channels +
                access_x_0 * num_channels + channel_id];
        const T* in_pos_1 =
            &in[out_id_h * input_w + access_y * in_img_w * num_channels +
                access_x_1 * num_channels + channel_id];
        const T* in_pos_2 =
            &in[out_id_h * input_w + access_y * in_img_w * num_channels +
                access_x_2 * num_channels + channel_id];
        const T* in_pos_3 =
            &in[out_id_h * input_w + access_y * in_img_w * num_channels +
                access_x_3 * num_channels + channel_id];

        coefficients[k] = Kecubic_interp(in_pos_0[0], in_pos_1[0], in_pos_2[0],
                                         in_pos_3[0], x_t);
      }

      out[out_id_h * output_w + out_id_w] =
          static_cast<T>(Kecubic_interp(coefficients[0], coefficients[1],
                                        coefficients[2], coefficients[3], y_t));
    }
  }
}

template <typename T>
__global__ void KeBicubicInterpBw(
    T* in, const size_t in_img_h, const size_t in_img_w, const size_t input_h,
    const size_t input_w, const T* out, const size_t out_img_h,
    const size_t out_img_w, const size_t output_h, const size_t output_w,
    const size_t num_channels, const float ratio_h, const float ratio_w,
    const bool align_corners, const DataLayout data_layout) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (; tid < nthreads; tid += stride) {
    int out_id_h = tid / output_w;
    int out_id_w = tid % output_w;
    int in_img_size = input_w / num_channels;
    int out_img_size = output_w / num_channels;

    int channel_id, out_img_idy, out_img_idx;
    if (data_layout == DataLayout::kNCHW) {
      channel_id = out_id_w / out_img_size;
      out_img_idy = (out_id_w % out_img_size) / out_img_w;
      out_img_idx = tid % out_img_w;
    } else {
      out_img_idy = out_id_w / (out_img_w * num_channels);
      out_img_idx = out_id_w % (out_img_w * num_channels) / num_channels;
      channel_id = tid % num_channels;
    }

    T in_img_idy = align_corners
                       ? static_cast<T>(ratio_h * out_img_idy)
                       : static_cast<T>(ratio_h * (out_img_idy + 0.5) - 0.5);
    int input_y = floorf(in_img_idy);
    const T y_t = in_img_idy - input_y;

    T in_img_idx = align_corners
                       ? static_cast<T>(ratio_w * out_img_idx)
                       : static_cast<T>(ratio_w * (out_img_idx + 0.5) - 0.5);
    int input_x = floorf(in_img_idx);

    const T x_t = in_img_idx - input_x;

    T x_coeffs[4];
    T y_coeffs[4];

    get_cubic_upsample_coefficients(x_coeffs, x_t);
    get_cubic_upsample_coefficients(y_coeffs, y_t);

    const T* out_pos = &out[out_id_h * output_w + out_id_w];
    T* in_pos;

    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        int access_y = max(min(static_cast<int>(input_y - 1 + j),
                               static_cast<int>(in_img_h - 1)),
                           0);
        int access_x = max(min(static_cast<int>(input_x - 1 + i),
                               static_cast<int>(in_img_w - 1)),
                           0);
        if (data_layout == DataLayout::kNCHW) {
          in_pos = &in[out_id_h * input_w + channel_id * in_img_size +
                       access_y * in_img_w + access_x];
        } else {
          in_pos = &in[out_id_h * input_w + access_y * in_img_w * num_channels +
                       access_x * num_channels + channel_id];
        }
        platform::CudaAtomicAdd(&in_pos[0],
                                (out_pos[0] * y_coeffs[j] * x_coeffs[i]));
      }
    }
  }
}

template <typename T>
static void Interpolate1DCUDAFwd(const framework::ExecutionContext& ctx,
                                 const Tensor& input, Tensor* output) {
  auto* input_data = input.data<T>();

  const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
  const DataLayout data_layout = framework::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  auto interp_method = ctx.Attr<std::string>("interp_method");
  bool align_corners = ctx.Attr<bool>("align_corners");
  int align_mode = ctx.Attr<int>("align_mode");

  int out_w = ctx.Attr<int>("out_w");

  auto list_new_shape_tensor = ctx.MultiInput<framework::Tensor>("SizeTensor");
  if (list_new_shape_tensor.size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(list_new_shape_tensor);
    out_w = new_size[0];
  } else {
    float scale;
    auto scale_tensor = ctx.Input<Tensor>("Scale");
    if (scale_tensor != nullptr) {
      auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
      scale = scale_data[0];
    } else {
      scale = ctx.Attr<float>("scale");
    }
    if (scale > 0) {
      out_w = static_cast<int>(in_w * scale);
    }
    auto out_size = ctx.Input<Tensor>("OutSize");
    if (out_size != nullptr) {
      Tensor sizes;
      framework::TensorCopySync(*out_size, platform::CPUPlace(), &sizes);
      auto size_data = sizes.data<int>();
      out_w = size_data[0];
    }
  }
  PADDLE_ENFORCE_GT(out_w, 0, platform::errors::InvalidArgument(
                                  "out_w in Attr(out_shape) of Op(interpolate) "
                                  "should be greater than 0."));
  framework::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {n, c, out_w};
  } else {
    dim_out = {n, out_w, c};
  }
  auto output_data = output->mutable_data<T>(dim_out, ctx.GetPlace());

  if (in_w == out_w) {
    framework::TensorCopy(input, ctx.GetPlace(), output);
    return;
  }

  float ratio_w = 0.f;
  if (out_w > 1) {
    ratio_w = (align_corners) ? static_cast<float>(in_w - 1.0) / (out_w - 1.0)
                              : static_cast<float>(in_w) / out_w;
  }

  int in_cw = c * in_w;
  int out_cw = c * out_w;
  int pixelNum = n * out_cw;

  platform::GpuLaunchConfig config =
      platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), pixelNum);

  if ("linear" == interp_method) {
    KeLinearInterpFw<T><<<config.block_per_grid, config.thread_per_block, 0,
                          ctx.cuda_device_context().stream()>>>(
        input_data, in_w, in_cw, output_data, out_w, n, out_cw, c, ratio_w,
        align_corners, align_mode, data_layout);
  }
}

template <typename T>
static void Interpolate2DCUDAFwd(const framework::ExecutionContext& ctx,
                                 const Tensor& input, Tensor* output) {
  auto* input_data = input.data<T>();

  const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
  const DataLayout data_layout = framework::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  auto interp_method = ctx.Attr<std::string>("interp_method");
  bool align_corners = ctx.Attr<bool>("align_corners");
  int align_mode = ctx.Attr<int>("align_mode");

  int out_h = ctx.Attr<int>("out_h");
  int out_w = ctx.Attr<int>("out_w");

  auto list_new_shape_tensor = ctx.MultiInput<framework::Tensor>("SizeTensor");
  if (list_new_shape_tensor.size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(list_new_shape_tensor);
    out_h = new_size[0];
    out_w = new_size[1];
  } else {
    float scale;
    auto scale_tensor = ctx.Input<Tensor>("Scale");
    if (scale_tensor != nullptr) {
      auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
      scale = scale_data[0];
    } else {
      scale = ctx.Attr<float>("scale");
    }
    if (scale > 0) {
      out_h = static_cast<int>(in_h * scale);
      out_w = static_cast<int>(in_w * scale);
    }
    auto out_size = ctx.Input<Tensor>("OutSize");
    if (out_size != nullptr) {
      Tensor sizes;
      framework::TensorCopySync(*out_size, platform::CPUPlace(), &sizes);
      auto size_data = sizes.data<int>();
      out_h = size_data[0];
      out_w = size_data[1];
    }
  }
  PADDLE_ENFORCE_GT(out_h, 0, platform::errors::InvalidArgument(
                                  "out_h in Attr(out_shape) of Op(interpolate) "
                                  "should be greater than 0."));
  PADDLE_ENFORCE_GT(out_w, 0, platform::errors::InvalidArgument(
                                  "out_w in Attr(out_shape) of Op(interpolate) "
                                  "should be greater than 0."));

  framework::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {n, c, out_h, out_w};
  } else {
    dim_out = {n, out_h, out_w, c};
  }
  auto output_data = output->mutable_data<T>(dim_out, ctx.GetPlace());

  if (in_h == out_h && in_w == out_w) {
    framework::TensorCopy(input, ctx.GetPlace(), output);
    return;
  }

  float ratio_h = 0.f;
  float ratio_w = 0.f;
  if (out_h > 1) {
    ratio_h = (align_corners) ? static_cast<float>(in_h - 1) / (out_h - 1)
                              : static_cast<float>(in_h) / out_h;
  }
  if (out_w > 1) {
    ratio_w = (align_corners) ? static_cast<float>(in_w - 1) / (out_w - 1)
                              : static_cast<float>(in_w) / out_w;
  }

  int in_hw = in_h * in_w;
  int out_hw = out_h * out_w;
  int in_chw = c * in_hw;
  int out_chw = c * out_hw;

  int pixelNum = n * out_chw;

  platform::GpuLaunchConfig config =
      platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), pixelNum);

  if ("nearest" == interp_method) {
    KeNearestNeighborInterpFw<
        T><<<config.block_per_grid, config.thread_per_block, 0,
             ctx.cuda_device_context().stream()>>>(
        input_data, in_h, in_w, n, in_chw, output_data, out_h, out_w, n,
        out_chw, c, ratio_h, ratio_w, align_corners, data_layout);
  } else if ("bilinear" == interp_method) {
    KeBilinearInterpFw<T><<<config.block_per_grid, config.thread_per_block, 0,
                            ctx.cuda_device_context().stream()>>>(
        input_data, in_h, in_w, n, in_chw, output_data, out_h, out_w, n,
        out_chw, c, ratio_h, ratio_w, align_corners, align_mode, data_layout);
  } else if ("bicubic" == interp_method) {
    KeBicubicInterpFw<T><<<config.block_per_grid, 512, 0,
                           ctx.cuda_device_context().stream()>>>(
        input_data, in_h, in_w, n, in_chw, output_data, out_h, out_w, n,
        out_chw, c, ratio_h, ratio_w, align_corners, data_layout);
  }
}

template <typename T>
static void Interpolate3DCUDAFwd(const framework::ExecutionContext& ctx,
                                 const Tensor& input, Tensor* output) {
  auto* input_data = input.data<T>();

  const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
  const DataLayout data_layout = framework::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  auto interp_method = ctx.Attr<std::string>("interp_method");
  bool align_corners = ctx.Attr<bool>("align_corners");
  int align_mode = ctx.Attr<int>("align_mode");

  int out_d = ctx.Attr<int>("out_d");
  int out_h = ctx.Attr<int>("out_h");
  int out_w = ctx.Attr<int>("out_w");

  auto list_new_shape_tensor = ctx.MultiInput<framework::Tensor>("SizeTensor");
  if (list_new_shape_tensor.size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(list_new_shape_tensor);
    out_d = new_size[0];
    out_h = new_size[1];
    out_w = new_size[2];
  } else {
    float scale;
    auto scale_tensor = ctx.Input<Tensor>("Scale");
    if (scale_tensor != nullptr) {
      auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
      scale = scale_data[0];
    } else {
      scale = ctx.Attr<float>("scale");
    }
    if (scale > 0) {
      out_d = static_cast<int>(in_d * scale);
      out_h = static_cast<int>(in_h * scale);
      out_w = static_cast<int>(in_w * scale);
    }
    auto out_size = ctx.Input<Tensor>("OutSize");
    if (out_size != nullptr) {
      Tensor sizes;
      framework::TensorCopySync(*out_size, platform::CPUPlace(), &sizes);
      auto size_data = sizes.data<int>();
      out_d = size_data[0];
      out_h = size_data[1];
      out_w = size_data[2];
    }
  }
  PADDLE_ENFORCE_GT(out_d, 0, platform::errors::InvalidArgument(
                                  "out_d in Attr(out_shape) of Op(interpolate) "
                                  "should be greater than 0."));
  PADDLE_ENFORCE_GT(out_h, 0, platform::errors::InvalidArgument(
                                  "out_h in Attr(out_shape) of Op(interpolate) "
                                  "should be greater than 0."));
  PADDLE_ENFORCE_GT(out_w, 0, platform::errors::InvalidArgument(
                                  "out_w in Attr(out_shape) of Op(interpolate) "
                                  "should be greater than 0."));

  framework::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {n, c, out_d, out_h, out_w};
  } else {
    dim_out = {n, out_d, out_h, out_w, c};
  }
  auto output_data = output->mutable_data<T>(dim_out, ctx.GetPlace());

  if (in_d == out_d && in_h == out_h && in_w == out_w) {
    framework::TensorCopy(input, ctx.GetPlace(), output);
    return;
  }

  float ratio_d = 0.f;
  float ratio_h = 0.f;
  float ratio_w = 0.f;
  if (out_d > 1) {
    ratio_d = (align_corners) ? static_cast<float>(in_d - 1) / (out_d - 1)
                              : static_cast<float>(in_d) / out_d;
  }
  if (out_h > 1) {
    ratio_h = (align_corners) ? static_cast<float>(in_h - 1) / (out_h - 1)
                              : static_cast<float>(in_h) / out_h;
  }
  if (out_w > 1) {
    ratio_w = (align_corners) ? static_cast<float>(in_w - 1) / (out_w - 1)
                              : static_cast<float>(in_w) / out_w;
  }

  int in_dhw = in_d * in_h * in_w;
  int out_dhw = out_d * out_h * out_w;
  int in_cdhw = c * in_dhw;
  int out_cdhw = c * out_dhw;

  int pixelNum = n * out_cdhw;

  platform::GpuLaunchConfig config =
      platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), pixelNum);

  if ("trilinear" == interp_method) {
    KeTrilinearInterpFw<T><<<config.block_per_grid, config.thread_per_block, 0,
                             ctx.cuda_device_context().stream()>>>(
        input_data, in_d, in_h, in_w, n, in_cdhw, output_data, out_d, out_h,
        out_w, n, out_cdhw, c, ratio_d, ratio_h, ratio_w, align_corners,
        align_mode, data_layout);
  }
}

template <typename T>
static void Interpolate1DCUDABwd(const framework::ExecutionContext& ctx,
                                 Tensor* input_grad, const Tensor output_grad) {
  auto* input = ctx.Input<Tensor>("X");
  const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
  const DataLayout data_layout = framework::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  ExtractNCDWH(input->dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  auto interp_method = ctx.Attr<std::string>("interp_method");
  bool align_corners = ctx.Attr<bool>("align_corners");
  int align_mode = ctx.Attr<int>("align_mode");

  int out_w = ctx.Attr<int>("out_w");
  float scale;
  auto scale_tensor = ctx.Input<Tensor>("Scale");
  if (scale_tensor != nullptr) {
    auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
    scale = scale_data[0];
  } else {
    scale = ctx.Attr<float>("scale");
  }
  if (scale > 0) {
    out_w = static_cast<int>(in_w * scale);
  }

  auto out_size = ctx.Input<Tensor>("OutSize");
  if (out_size != nullptr) {
    Tensor sizes;
    framework::TensorCopySync(*out_size, platform::CPUPlace(), &sizes);
    auto size_data = sizes.data<int>();
    out_w = size_data[0];
  }
  auto list_new_size_tensor = ctx.MultiInput<framework::Tensor>("SizeTensor");
  if (list_new_size_tensor.size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(list_new_size_tensor);
    out_w = new_size[0];
  }

  auto* output_grad_data = output_grad.data<T>();
  framework::DDim dim_grad;
  if (data_layout == DataLayout::kNCHW) {
    dim_grad = {n, c, in_w};
  } else {
    dim_grad = {n, in_w, c};
  }
  input_grad->mutable_data<T>(dim_grad, ctx.GetPlace());
  auto* input_grad_data = input_grad->mutable_data<T>(dim_grad, ctx.GetPlace());
  auto& device_ctx = ctx.template device_context<platform::CUDADeviceContext>();
  math::SetConstant<platform::CUDADeviceContext, T> zero;
  zero(device_ctx, input_grad, static_cast<T>(0.0));

  if (in_w == out_w) {
    framework::TensorCopy(output_grad, ctx.GetPlace(), input_grad);
    return;
  }

  float ratio_w = 0.f;
  if (out_w > 1) {
    ratio_w = (align_corners) ? static_cast<float>(in_w - 1) / (out_w - 1)
                              : static_cast<float>(in_w) / out_w;
  }
  int in_cw = c * in_w;
  int out_cw = c * out_w;
  int pixelNum = n * out_cw;

  platform::GpuLaunchConfig config =
      platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), pixelNum);

  if ("linear" == interp_method) {
    KeLinearInterpBw<T><<<config.block_per_grid, config.thread_per_block, 0,
                          ctx.cuda_device_context().stream()>>>(
        input_grad_data, in_w, in_cw, output_grad_data, out_w, n, out_cw, c,
        ratio_w, align_corners, align_mode, data_layout);
  }
}

template <typename T>
static void Interpolate2DCUDABwd(const framework::ExecutionContext& ctx,
                                 Tensor* input_grad, const Tensor output_grad) {
  auto* input = ctx.Input<Tensor>("X");
  const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
  const DataLayout data_layout = framework::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  ExtractNCDWH(input->dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  auto interp_method = ctx.Attr<std::string>("interp_method");
  bool align_corners = ctx.Attr<bool>("align_corners");
  int align_mode = ctx.Attr<int>("align_mode");

  int out_h = ctx.Attr<int>("out_h");
  int out_w = ctx.Attr<int>("out_w");
  float scale;
  auto scale_tensor = ctx.Input<Tensor>("Scale");
  if (scale_tensor != nullptr) {
    auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
    scale = scale_data[0];
  } else {
    scale = ctx.Attr<float>("scale");
  }
  if (scale > 0) {
    out_h = static_cast<int>(in_h * scale);
    out_w = static_cast<int>(in_w * scale);
  }

  auto out_size = ctx.Input<Tensor>("OutSize");
  if (out_size != nullptr) {
    Tensor sizes;
    framework::TensorCopySync(*out_size, platform::CPUPlace(), &sizes);
    auto size_data = sizes.data<int>();
    out_h = size_data[0];
    out_w = size_data[1];
  }
  auto list_new_size_tensor = ctx.MultiInput<framework::Tensor>("SizeTensor");
  if (list_new_size_tensor.size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(list_new_size_tensor);
    out_h = new_size[0];
    out_w = new_size[1];
  }

  auto* output_grad_data = output_grad.data<T>();
  framework::DDim dim_grad;
  if (data_layout == DataLayout::kNCHW) {
    dim_grad = {n, c, in_h, in_w};
  } else {
    dim_grad = {n, in_h, in_w, c};
  }
  input_grad->mutable_data<T>(dim_grad, ctx.GetPlace());
  auto* input_grad_data = input_grad->mutable_data<T>(dim_grad, ctx.GetPlace());
  auto& device_ctx = ctx.template device_context<platform::CUDADeviceContext>();
  math::SetConstant<platform::CUDADeviceContext, T> zero;
  zero(device_ctx, input_grad, static_cast<T>(0.0));

  if (in_h == out_h && in_w == out_w) {
    framework::TensorCopy(output_grad, ctx.GetPlace(), input_grad);
    return;
  }

  float ratio_h = 0.f;
  float ratio_w = 0.f;
  if (out_h > 1) {
    ratio_h = (align_corners) ? static_cast<float>(in_h - 1) / (out_h - 1)
                              : static_cast<float>(in_h) / out_h;
  }
  if (out_w > 1) {
    ratio_w = (align_corners) ? static_cast<float>(in_w - 1) / (out_w - 1)
                              : static_cast<float>(in_w) / out_w;
  }

  int in_hw = in_h * in_w;
  int out_hw = out_h * out_w;
  int in_chw = c * in_hw;
  int out_chw = c * out_hw;

  int pixelNum = n * out_chw;

  platform::GpuLaunchConfig config =
      platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), pixelNum);

  if ("nearest" == interp_method) {
    KeNearestNeighborInterpBw<
        T><<<config.block_per_grid, config.thread_per_block, 0,
             ctx.cuda_device_context().stream()>>>(
        input_grad_data, in_h, in_w, n, in_chw, output_grad_data, out_h, out_w,
        n, out_chw, c, ratio_h, ratio_w, align_corners, data_layout);
  } else if ("bilinear" == interp_method) {
    KeBilinearInterpBw<T><<<config.block_per_grid, config.thread_per_block, 0,
                            ctx.cuda_device_context().stream()>>>(
        input_grad_data, in_h, in_w, n, in_chw, output_grad_data, out_h, out_w,
        n, out_chw, c, ratio_h, ratio_w, align_corners, align_mode,
        data_layout);
  } else if ("bicubic" == interp_method) {
    KeBicubicInterpBw<T><<<config.block_per_grid, 512, 0,
                           ctx.cuda_device_context().stream()>>>(
        input_grad_data, in_h, in_w, n, in_chw, output_grad_data, out_h, out_w,
        n, out_chw, c, ratio_h, ratio_w, align_corners, data_layout);
  }
}

template <typename T>
static void Interpolate3DCUDABwd(const framework::ExecutionContext& ctx,
                                 Tensor* input_grad,
                                 const Tensor& output_grad) {
  auto* input = ctx.Input<Tensor>("X");
  const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
  const DataLayout data_layout = framework::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  ExtractNCDWH(input->dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  auto interp_method = ctx.Attr<std::string>("interp_method");
  bool align_corners = ctx.Attr<bool>("align_corners");
  int align_mode = ctx.Attr<int>("align_mode");

  int out_d = ctx.Attr<int>("out_d");
  int out_h = ctx.Attr<int>("out_h");
  int out_w = ctx.Attr<int>("out_w");
  float scale;
  auto scale_tensor = ctx.Input<Tensor>("Scale");
  if (scale_tensor != nullptr) {
    auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
    scale = scale_data[0];
  } else {
    scale = ctx.Attr<float>("scale");
  }
  if (scale > 0) {
    out_d = static_cast<int>(in_d * scale);
    out_h = static_cast<int>(in_h * scale);
    out_w = static_cast<int>(in_w * scale);
  }

  auto out_size = ctx.Input<Tensor>("OutSize");
  if (out_size != nullptr) {
    Tensor sizes;
    framework::TensorCopySync(*out_size, platform::CPUPlace(), &sizes);
    auto size_data = sizes.data<int>();
    out_d = size_data[0];
    out_h = size_data[1];
    out_w = size_data[2];
  }
  auto list_new_size_tensor = ctx.MultiInput<framework::Tensor>("SizeTensor");
  if (list_new_size_tensor.size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(list_new_size_tensor);
    out_d = new_size[0];
    out_h = new_size[1];
    out_w = new_size[2];
  }

  auto* output_grad_data = output_grad.data<T>();
  framework::DDim dim_grad;
  if (data_layout == DataLayout::kNCHW) {
    dim_grad = {n, c, in_d, in_h, in_w};
  } else {
    dim_grad = {n, in_d, in_h, in_w, c};
  }
  auto* input_grad_data = input_grad->mutable_data<T>(dim_grad, ctx.GetPlace());
  auto& device_ctx = ctx.template device_context<platform::CUDADeviceContext>();
  math::SetConstant<platform::CUDADeviceContext, T> zero;
  zero(device_ctx, input_grad, static_cast<T>(0.0));

  if (in_d == out_d && in_h == out_h && in_w == out_w) {
    framework::TensorCopy(output_grad, ctx.GetPlace(), input_grad);
    return;
  }

  float ratio_d = 0.f;
  float ratio_h = 0.f;
  float ratio_w = 0.f;
  if (out_d > 1) {
    ratio_d = (align_corners) ? static_cast<float>(in_d - 1) / (out_d - 1)
                              : static_cast<float>(in_d) / out_d;
  }
  if (out_h > 1) {
    ratio_h = (align_corners) ? static_cast<float>(in_h - 1) / (out_h - 1)
                              : static_cast<float>(in_h) / out_h;
  }
  if (out_w > 1) {
    ratio_w = (align_corners) ? static_cast<float>(in_w - 1) / (out_w - 1)
                              : static_cast<float>(in_w) / out_w;
  }

  int in_dhw = in_d * in_h * in_w;
  int out_dhw = out_d * out_h * out_w;
  int in_cdhw = c * in_dhw;
  int out_cdhw = c * out_dhw;

  int pixelNum = n * out_cdhw;

  platform::GpuLaunchConfig config =
      platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), pixelNum);

  if ("trilinear" == interp_method) {
    KeTrilinearInterpBw<T><<<config.block_per_grid, config.thread_per_block, 0,
                             ctx.cuda_device_context().stream()>>>(
        input_grad_data, in_d, in_h, in_w, n, in_cdhw, output_grad_data, out_d,
        out_h, out_w, n, out_cdhw, c, ratio_d, ratio_h, ratio_w, align_corners,
        align_mode, data_layout);
  }
}

template <typename T>
class InterpolateOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::NotFound("This kernel only runs on GPU device."));
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");

    auto input_dims = input->dims();
    if (input_dims.size() == 3) {  // 1D interpolation
      Interpolate1DCUDAFwd<T>(ctx, *input, output);
    } else if (input_dims.size() == 4) {  // 2D interpolation
      Interpolate2DCUDAFwd<T>(ctx, *input, output);
    } else if (input_dims.size() == 5) {  // 3D interpolation
      Interpolate3DCUDAFwd<T>(ctx, *input, output);
    }
  }
};

template <typename T>
class InterpolateGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::NotFound("This kernel only runs on GPU device."));
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto output_grad_dims = output_grad->dims();
    if (output_grad_dims.size() == 3) {  // 1D interpolation
      Interpolate1DCUDABwd<T>(ctx, input_grad, *output_grad);
    } else if (output_grad_dims.size() == 4) {  // 2D interpolation
      Interpolate2DCUDABwd<T>(ctx, input_grad, *output_grad);
    } else if (output_grad_dims.size() == 5) {  // 3D interpolation
      Interpolate3DCUDABwd<T>(ctx, input_grad, *output_grad);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(bilinear_interp, ops::InterpolateOpCUDAKernel<float>,
                        ops::InterpolateOpCUDAKernel<double>,
                        ops::InterpolateOpCUDAKernel<int>);
REGISTER_OP_CUDA_KERNEL(bilinear_interp_grad,
                        ops::InterpolateGradOpCUDAKernel<float>,
                        ops::InterpolateGradOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(nearest_interp, ops::InterpolateOpCUDAKernel<float>,
                        ops::InterpolateOpCUDAKernel<double>,
                        ops::InterpolateOpCUDAKernel<int>);
REGISTER_OP_CUDA_KERNEL(nearest_interp_grad,
                        ops::InterpolateGradOpCUDAKernel<float>,
                        ops::InterpolateGradOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(trilinear_interp, ops::InterpolateOpCUDAKernel<float>,
                        ops::InterpolateOpCUDAKernel<double>,
                        ops::InterpolateOpCUDAKernel<int>);
REGISTER_OP_CUDA_KERNEL(trilinear_interp_grad,
                        ops::InterpolateGradOpCUDAKernel<float>,
                        ops::InterpolateGradOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(linear_interp, ops::InterpolateOpCUDAKernel<float>,
                        ops::InterpolateOpCUDAKernel<double>,
                        ops::InterpolateOpCUDAKernel<int>);
REGISTER_OP_CUDA_KERNEL(linear_interp_grad,
                        ops::InterpolateGradOpCUDAKernel<float>,
                        ops::InterpolateGradOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(bicubic_interp, ops::InterpolateOpCUDAKernel<float>,
                        ops::InterpolateOpCUDAKernel<double>,
                        ops::InterpolateOpCUDAKernel<int>);
REGISTER_OP_CUDA_KERNEL(bicubic_interp_grad,
                        ops::InterpolateGradOpCUDAKernel<float>,
                        ops::InterpolateGradOpCUDAKernel<double>);
