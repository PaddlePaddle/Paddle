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
#include "paddle/fluid/operators/interpolate_v2_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/fast_divmod.h"
#include "paddle/pten/kernels/funcs/math_cuda_utils.h"

namespace paddle {
namespace operators {

using framework::Tensor;
using platform::FastDivMod;
using DataLayout = framework::DataLayout;

static inline int GetLastPow2(int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max(1, n - (n >> 1));
}

inline platform::GpuLaunchConfig GetGpuLaunchConfig3D(
    const platform::CUDADeviceContext& context, int num_img, int height,
    int width) {
  const int kThreadsPerBlock = 256;
  int max_threads_per_block = context.GetMaxThreadsPerBlock();  // 1024
  int max_threads = std::min(kThreadsPerBlock, max_threads_per_block);

  int block_x = std::min(GetLastPow2(width), max_threads);
  int block_y = std::min(GetLastPow2(height), max_threads / block_x);
  int block_z = std::min(num_img, max_threads / block_x / block_y);

  auto max_grid_dim = context.GetCUDAMaxGridDimSize();
  int grid_x = std::min<int>(max_grid_dim[0], platform::DivUp(width, block_x));
  int grid_y = std::min<int>(max_grid_dim[1], platform::DivUp(height, block_y));
  int grid_z =
      std::min<int>(max_grid_dim[2], platform::DivUp(num_img, block_z * 4));

  const int capability = context.GetComputeCapability();
  platform::GpuLaunchConfig config;
  config.compute_capability = capability;
  config.thread_per_block = dim3(block_x, block_y, block_z);
  config.block_per_grid = dim3(grid_x, grid_y, grid_z);
  return config;
}

template <typename T>
__forceinline__ __device__ void PreCalculatorForLinearInterpInputIndex(
    int* in_img_idx, int* w_id, T* w1lambda, T* w2lambda, T src_w,
    const int in_img_w) {
  src_w = (src_w > 0) ? src_w : 0.f;
  *in_img_idx = static_cast<int>(src_w);
  *w_id = (*in_img_idx < in_img_w - 1) ? 1 : 0;
  *w1lambda = src_w - *in_img_idx;
  *w2lambda = 1.f - *w1lambda;
}

struct FastDivModForInterpolate {
 public:
  FastDivMod channels_div;
  FastDivMod output_w_div;
  FastDivMod output_wc_div;

  explicit HOSTDEVICE FastDivModForInterpolate(const int channels,
                                               const int output_w,
                                               const int outout_wc)
      : channels_div(FastDivMod(channels)),
        output_w_div(FastDivMod(output_w)),
        output_wc_div(FastDivMod(outout_wc)) {}
};

template <typename T>
__global__ void KeNearestNeighborInterpNCHWFw(
    const T* in, const size_t in_img_h, const size_t in_img_w, T* out,
    const size_t out_img_h, const size_t out_img_w, const size_t nc,
    const float ratio_h, const float ratio_w, const bool align_corners) {
  int out_img_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int out_img_idy = threadIdx.y + blockIdx.y * blockDim.y;
  int nc_id = threadIdx.z + blockIdx.z * blockDim.z;
  int nc_stride = blockDim.z * gridDim.z;

  // nearest_sampling by multiple read in_addr and write to out_addr
  int in_img_idx = (align_corners)
                       ? static_cast<int>(ratio_w * out_img_idx + 0.5)
                       : static_cast<int>(ratio_w * out_img_idx);
  int in_img_idy = (align_corners)
                       ? static_cast<int>(ratio_h * out_img_idy + 0.5)
                       : static_cast<int>(ratio_h * out_img_idy);

  int in_index = (nc_id * in_img_h + in_img_idy) * in_img_w + in_img_idx;
  int in_index_stride = nc_stride * in_img_h * in_img_w;

  int out_index = (nc_id * out_img_h + out_img_idy) * out_img_w + out_img_idx;
  int out_index_stride = nc_stride * out_img_h * out_img_w;

  // prevent from multiple threads writing
  if (out_img_idx < out_img_w && out_img_idy < out_img_h) {
    while (nc_id < nc) {
      out[out_index] = in[in_index];
      in_index += in_index_stride;
      out_index += out_index_stride;
      nc_id += nc_stride;
    }
  }
}

template <typename T>
__global__ void KeNearestNeighborInterpFw(
    const T* in, const size_t in_img_h, const size_t in_img_w,
    const size_t input_h, const size_t input_w, T* out, const size_t out_img_h,
    const size_t out_img_w, const size_t output_h, const size_t output_w,
    const size_t num_channels, const float ratio_h, const float ratio_w,
    const bool align_corners, FastDivModForInterpolate divmods) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int in_img_size = in_img_h * in_img_w;
  int out_img_size = out_img_h * out_img_w;

  for (; tid < nthreads; tid += stride) {
    auto out_id_divmod = divmods.output_w_div.Divmod(tid);
    int out_id_h = out_id_divmod.val[0];
    int out_id_w = out_id_divmod.val[1];

    int channel_id = divmods.channels_div.Divmod(tid).val[1];
    auto outimg_id_divmod = divmods.output_wc_div.Divmod(out_id_w);
    int out_img_idy = outimg_id_divmod.val[0];
    int out_img_idx =
        divmods.channels_div.Divmod(outimg_id_divmod.val[1]).val[0];

    int in_img_idy = (align_corners)
                         ? static_cast<int>(ratio_h * out_img_idy + 0.5)
                         : static_cast<int>(ratio_h * out_img_idy);
    int in_img_idx = (align_corners)
                         ? static_cast<int>(ratio_w * out_img_idx + 0.5)
                         : static_cast<int>(ratio_w * out_img_idx);

    out[tid] = in[out_id_h * input_w + in_img_idy * in_img_w * num_channels +
                  in_img_idx * num_channels + channel_id];
  }
}

template <typename T>
__global__ void KeNearestNeighbor3DInterpFw(
    const T* in, const size_t in_img_d, const size_t in_img_h,
    const size_t in_img_w, const size_t input_h, const size_t input_w, T* out,
    const size_t out_img_d, const size_t out_img_h, const size_t out_img_w,
    const size_t output_h, const size_t output_w, const size_t num_channels,
    const float ratio_d, const float ratio_h, const float ratio_w,
    const bool align_corners, const DataLayout data_layout) {
  int nthreads = output_h * output_w;  // ncdhw
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
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

    int in_img_idt = (align_corners)
                         ? static_cast<int>(ratio_d * out_img_idt + 0.5)
                         : static_cast<int>(ratio_d * out_img_idt);

    int in_img_idy = (align_corners)
                         ? static_cast<int>(ratio_h * out_img_idy + 0.5)
                         : static_cast<int>(ratio_h * out_img_idy);
    int in_img_idx = (align_corners)
                         ? static_cast<int>(ratio_w * out_img_idx + 0.5)
                         : static_cast<int>(ratio_w * out_img_idx);

    if (data_layout == DataLayout::kNCHW) {
      out[tid] = in[out_id_h * input_w + channel_id * in_img_size +
                    in_img_idt * in_img_h * in_img_w + in_img_idy * in_img_w +
                    in_img_idx];
    } else {
      out[tid] = in[out_id_h * input_w +
                    in_img_idt * in_img_h * in_img_w * num_channels +
                    in_img_idy * in_img_w * num_channels +
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
__global__ void KeNearestNeighbor3DInterpBw(
    T* in, const size_t in_img_d, const size_t in_img_h, const size_t in_img_w,
    const size_t input_h, const size_t input_w, const T* out,
    const size_t out_img_d, const size_t out_img_h, const size_t out_img_w,
    const size_t output_h, const size_t output_w, const size_t num_channels,
    const float ratio_d, const float ratio_h, const float ratio_w,
    const bool align_corners, const DataLayout data_layout) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
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

    int in_img_idt = (align_corners)
                         ? static_cast<int>(ratio_d * out_img_idt + 0.5)
                         : static_cast<int>(ratio_d * out_img_idt);
    int in_img_idy = (align_corners)
                         ? static_cast<int>(ratio_h * out_img_idy + 0.5)
                         : static_cast<int>(ratio_h * out_img_idy);
    int in_img_idx = (align_corners)
                         ? static_cast<int>(ratio_w * out_img_idx + 0.5)
                         : static_cast<int>(ratio_w * out_img_idx);

    T* in_pos;
    if (data_layout == DataLayout::kNCHW) {
      in_pos = &in[out_id_h * input_w + channel_id * in_img_size +
                   in_img_idt * in_img_h * in_img_w + in_img_idy * in_img_w +
                   in_img_idx];
    } else {
      in_pos = &in[out_id_h * input_w +
                   in_img_idt * in_img_h * in_img_w * num_channels +
                   in_img_idy * in_img_w * num_channels +
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
__global__ void KeBilinearInterpNCHWFw(const T* in, const size_t in_img_h,
                                       const size_t in_img_w, T* out,
                                       const size_t out_img_h,
                                       const size_t out_img_w, const size_t nc,
                                       const float ratio_h, const float ratio_w,
                                       const T align_type_value) {
  int out_img_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int out_img_idy = threadIdx.y + blockIdx.y * blockDim.y;
  int nc_id = threadIdx.z + blockIdx.z * blockDim.z;
  int nc_stride = blockDim.z * gridDim.z;

  int in_img_idx, in_img_idy, h_id, w_id;
  T h1lambda, w1lambda, h2lambda, w2lambda;
  T src_w = ratio_w * (out_img_idx + align_type_value) - align_type_value;
  T src_h = ratio_h * (out_img_idy + align_type_value) - align_type_value;

  PreCalculatorForLinearInterpInputIndex(&in_img_idx, &w_id, &w1lambda,
                                         &w2lambda, src_w, in_img_w);
  PreCalculatorForLinearInterpInputIndex(&in_img_idy, &h_id, &h1lambda,
                                         &h2lambda, src_h, in_img_h);

  int in_index = (nc_id * in_img_h + in_img_idy) * in_img_w + in_img_idx;
  int in_index_stride = nc_stride * in_img_h * in_img_w;

  int out_index = (nc_id * out_img_h + out_img_idy) * out_img_w + out_img_idx;
  int out_index_stride = nc_stride * out_img_h * out_img_w;

  // prevent from multiple threads writing
  if (out_img_idx < out_img_w && out_img_idy < out_img_h) {
    while (nc_id < nc) {
      const T* in_pos = &in[in_index];
      out[out_index] =
          h2lambda * (w2lambda * in_pos[0] + w1lambda * in_pos[w_id]) +
          h1lambda * (w2lambda * in_pos[h_id * in_img_w] +
                      w1lambda * in_pos[h_id * in_img_w + w_id]);

      in_index += in_index_stride;
      out_index += out_index_stride;
      nc_id += nc_stride;
    }
  }
}

template <typename T>
__global__ void KeBilinearInterpFw(
    const T* in, const size_t in_img_h, const size_t in_img_w,
    const size_t input_h, const size_t input_w, T* out, const size_t out_img_h,
    const size_t out_img_w, const size_t output_h, const size_t output_w,
    const size_t num_channels, const float ratio_h, const float ratio_w,
    const T align_type_value, FastDivModForInterpolate divmods) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (; tid < nthreads; tid += stride) {
    auto out_id_divmod = divmods.output_w_div.Divmod(tid);
    int out_id_h = out_id_divmod.val[0];
    int out_id_w = out_id_divmod.val[1];

    int channel_id = divmods.channels_div.Divmod(tid).val[1];
    auto outimg_id_divmod = divmods.output_wc_div.Divmod(out_id_w);
    int out_img_idy = outimg_id_divmod.val[0];
    int out_img_idx =
        divmods.channels_div.Divmod(outimg_id_divmod.val[1]).val[0];

    int in_img_idx, in_img_idy, h_id, w_id;
    T h1lambda, w1lambda, h2lambda, w2lambda;
    T src_w = ratio_w * (out_img_idx + align_type_value) - align_type_value;
    T src_h = ratio_h * (out_img_idy + align_type_value) - align_type_value;

    PreCalculatorForLinearInterpInputIndex(&in_img_idx, &w_id, &w1lambda,
                                           &w2lambda, src_w, in_img_w);
    PreCalculatorForLinearInterpInputIndex(&in_img_idy, &h_id, &h1lambda,
                                           &h2lambda, src_h, in_img_h);

    // bilinear interpolation
    const T* in_pos =
        &in[out_id_h * input_w + in_img_idy * in_img_w * num_channels +
            in_img_idx * num_channels + channel_id];
    out[tid] =
        h2lambda *
            (w2lambda * in_pos[0] + w1lambda * in_pos[w_id * num_channels]) +
        h1lambda *
            (w2lambda * in_pos[h_id * in_img_w * num_channels] +
             w1lambda *
                 in_pos[h_id * in_img_w * num_channels + w_id * num_channels]);
  }
}

/* Calculate the minimum of partial elements in a block */
template <typename T>
__inline__ __device__ T PartialBlockMin(T val, size_t threads_num_in_block,
                                        unsigned mask) {
  __shared__ T shared[WARP_SIZE];
  __shared__ T shared_last_val;
  __shared__ int shared_last_idx;
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;
  int threshold = (threads_num_in_block & (-WARP_SIZE));

  if (threadIdx.x < threshold) {
    shared_last_idx = (threshold >> 5) - 1;
    val = pten::funcs::warpReduceMin(val, mask);
    if (lane == 0) {
      shared[wid] = val;
    }
  } else {
    shared_last_val = std::numeric_limits<T>::max();
    platform::CudaAtomicMin(&shared_last_val, val);
    shared[wid] = shared_last_val;
    shared_last_idx = wid;
  }
  __syncthreads();

  if (threadIdx.x < threshold) {
    val = (lane <= shared_last_idx) ? shared[lane]
                                    : std::numeric_limits<T>::max();
    val = pten::funcs::warpReduceMin(val, mask);
    shared_last_val = val;
  }
  __syncthreads();
  if (threadIdx.x >= threshold) {
    val = shared_last_val;
  }
  return val;
}

template <typename T>
__global__ void KeBilinearInterpBwShareMemory(
    T* in, const int in_h, const int in_w, const T* __restrict__ out,
    const int out_h, const int out_w, const int n, const int num_channels,
    float ratio_h, float ratio_w, const T align_type_value, bool is_nchw) {
  __shared__ T s_data[2][1024];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int in_chw = in_h * in_w * num_channels;
  int out_chw = num_channels * out_h * out_w;
  int nthreads = n * out_chw;

  for (; tid < nthreads; tid += stride) {
    int out_id_h = tid / out_chw;
    int out_id_w = tid % out_chw;
    const int in_img_size = in_h * in_w;
    const int out_img_size = out_h * out_w;
    T value = out[out_id_h * out_chw + out_id_w];

    int channel_id = out_id_w / out_img_size;
    int out_img_idy = (out_id_w % out_img_size) / out_w;
    int out_img_idx = tid % out_w;

    int in_img_idx, in_img_idy, w_id, h_id;
    T w1lambda, h1lambda, w2lambda, h2lambda;
    T src_w = ratio_w * (out_img_idx + align_type_value) - align_type_value;
    T src_h = ratio_h * (out_img_idy + align_type_value) - align_type_value;

    PreCalculatorForLinearInterpInputIndex(&in_img_idx, &w_id, &w1lambda,
                                           &w2lambda, src_w, in_w);
    PreCalculatorForLinearInterpInputIndex(&in_img_idy, &h_id, &h1lambda,
                                           &h2lambda, src_h, in_h);

    // top_left_index is just input_index.
    int input_index = out_id_h * in_chw + channel_id * in_img_size +
                      in_img_idy * in_w + in_img_idx;
    int top_right_index = input_index + w_id;
    int bot_left_index = input_index + h_id * in_w;
    int bot_right_index = input_index + h_id * in_w + w_id;
    int in_top_min_index, in_bot_min_index;

    s_data[0][threadIdx.x] = 0.f;
    s_data[1][threadIdx.x] = 0.f;
    int remain = nthreads - (tid & (-blockDim.x));
    int in_top_max_index =
        pten::funcs::blockReduceMax(top_right_index, FINAL_MASK);
    int in_bot_max_index =
        pten::funcs::blockReduceMax(bot_right_index, FINAL_MASK);

    if (remain > blockDim.x) {
      in_top_min_index = pten::funcs::blockReduceMin(input_index, FINAL_MASK);
      in_bot_min_index =
          pten::funcs::blockReduceMin(bot_left_index, FINAL_MASK);
    } else {
      in_top_min_index = PartialBlockMin(input_index, remain, FINAL_MASK);
      in_bot_min_index = PartialBlockMin(bot_left_index, remain, FINAL_MASK);
    }
    int upper_limit_share_idx = (in_top_max_index - in_top_min_index) >
                                        (in_bot_max_index - in_bot_min_index)
                                    ? (in_top_max_index - in_top_min_index)
                                    : (in_bot_max_index - in_bot_min_index);
    if (h_id != 0) {
      platform::CudaAtomicAdd(&s_data[0][input_index - in_top_min_index],
                              h2lambda * w2lambda * value);
      platform::CudaAtomicAdd(&s_data[0][top_right_index - in_top_min_index],
                              h2lambda * w1lambda * value);
      platform::CudaAtomicAdd(&s_data[1][bot_left_index - in_bot_min_index],
                              h1lambda * w2lambda * value);
      platform::CudaAtomicAdd(&s_data[1][bot_right_index - in_bot_min_index],
                              h1lambda * w1lambda * value);
    } else {
      platform::CudaAtomicAdd(&s_data[0][top_right_index - in_top_min_index],
                              (h2lambda + h1lambda) * w1lambda * value);
      platform::CudaAtomicAdd(&s_data[1][bot_left_index - in_bot_min_index],
                              (h1lambda + h2lambda) * w2lambda * value);
    }
    __syncthreads();

    if (threadIdx.x <= upper_limit_share_idx) {
      platform::CudaAtomicAdd(&in[in_top_min_index + threadIdx.x],
                              s_data[0][threadIdx.x]);
      platform::CudaAtomicAdd(&in[in_bot_min_index + threadIdx.x],
                              s_data[1][threadIdx.x]);
    }
  }
}

template <typename T>
__global__ void KeBilinearInterpBw(T* in, const int in_h, const int in_w,
                                   const T* __restrict__ out, const int out_h,
                                   const int out_w, const int n,
                                   const int num_channels, float ratio_h,
                                   float ratio_w, const T align_type_value,
                                   bool is_nchw) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int in_chw = in_h * in_w * num_channels;
  int out_chw = num_channels * out_h * out_w;
  int nthreads = n * out_chw;

  if (is_nchw) {
    for (; tid < nthreads; tid += stride) {
      int out_id_h = tid / out_chw;
      int out_id_w = tid % out_chw;
      const int in_img_size = in_h * in_w;
      const int out_img_size = out_h * out_w;
      T value = out[out_id_h * out_chw + out_id_w];

      int channel_id = out_id_w / out_img_size;
      int out_img_idy = (out_id_w % out_img_size) / out_w;
      int out_img_idx = tid % out_w;
      int in_img_idx, in_img_idy, w_id, h_id;
      T w1lambda, h1lambda, w2lambda, h2lambda;

      T src_w = ratio_w * (out_img_idx + align_type_value) - align_type_value;
      T src_h = ratio_h * (out_img_idy + align_type_value) - align_type_value;

      PreCalculatorForLinearInterpInputIndex(&in_img_idx, &w_id, &w1lambda,
                                             &w2lambda, src_w, in_w);
      PreCalculatorForLinearInterpInputIndex(&in_img_idy, &h_id, &h1lambda,
                                             &h2lambda, src_h, in_h);

      T* in_pos = &in[out_id_h * in_chw + channel_id * in_img_size +
                      in_img_idy * in_w + in_img_idx];
      platform::CudaAtomicAdd(&in_pos[0], h2lambda * w2lambda * value);
      platform::CudaAtomicAdd(&in_pos[w_id], h2lambda * w1lambda * value);
      platform::CudaAtomicAdd(&in_pos[h_id * in_w],
                              h1lambda * w2lambda * value);
      platform::CudaAtomicAdd(&in_pos[h_id * in_w + w_id],
                              h1lambda * w1lambda * value);
    }
  } else {
    for (; tid < nthreads; tid += stride) {
      int out_id_h = tid / out_chw;
      int out_id_w = tid % out_chw;
      const int in_img_size = in_h * in_w;
      const int out_img_size = out_h * out_w;
      T value = out[out_id_h * out_chw + out_id_w];

      int out_img_idy = out_id_w / (out_w * num_channels);
      int out_img_idx = out_id_w % (out_w * num_channels) / num_channels;
      int channel_id = tid % num_channels;

      int in_img_idx, in_img_idy, w_id, h_id;
      T w1lambda, h1lambda, w2lambda, h2lambda;
      T src_w = ratio_w * (out_img_idx + align_type_value) - align_type_value;
      T src_h = ratio_h * (out_img_idy + align_type_value) - align_type_value;

      PreCalculatorForLinearInterpInputIndex(&in_img_idx, &w_id, &w1lambda,
                                             &w2lambda, src_w, in_w);
      PreCalculatorForLinearInterpInputIndex(&in_img_idy, &h_id, &h1lambda,
                                             &h2lambda, src_h, in_h);

      T* in_pos = &in[out_id_h * in_chw + in_img_idy * in_w * num_channels +
                      in_img_idx * num_channels + channel_id];
      platform::CudaAtomicAdd(&in_pos[0], h2lambda * w2lambda * value);
      platform::CudaAtomicAdd(&in_pos[w_id * num_channels],
                              h2lambda * w1lambda * value);
      platform::CudaAtomicAdd(&in_pos[h_id * in_w * num_channels],
                              h1lambda * w2lambda * value);
      platform::CudaAtomicAdd(
          &in_pos[h_id * in_w * num_channels + w_id * num_channels],
          h1lambda * w1lambda * value);
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
  float scale_w = -1;
  if (list_new_shape_tensor.size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(list_new_shape_tensor);
    out_w = new_size[0];
  } else {
    auto scale_tensor = ctx.Input<Tensor>("Scale");
    auto scale = ctx.Attr<std::vector<float>>("scale");
    if (scale_tensor != nullptr) {
      auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
      scale_w = scale_data[0];
      PADDLE_ENFORCE_EQ(
          scale_w > 0, true,
          platform::errors::InvalidArgument(
              "The scale_w in input 'Scale' Tensor of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
    } else {
      if (scale.size() > 0) {
        scale_w = scale[0];
        PADDLE_ENFORCE_EQ(
            scale_w > 0, true,
            platform::errors::InvalidArgument(
                "The scale_w in Attr(scale) of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_w));
      }
    }
    if (scale_w > 0.) {
      out_w = static_cast<int>(in_w * scale_w);
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
    float new_scale_w = 0.f;
    new_scale_w = (scale_w > 0) ? static_cast<float>(1. / scale_w)
                                : static_cast<float>(in_w) / out_w;
    ratio_w = (align_corners) ? static_cast<float>(in_w - 1.0) / (out_w - 1.0)
                              : static_cast<float>(new_scale_w);
  }

  int64_t in_cw = c * in_w;
  int64_t out_cw = c * out_w;
  auto pixelNum = n * out_cw;

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
  float scale_w = -1;
  float scale_h = -1;
  if (list_new_shape_tensor.size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(list_new_shape_tensor);
    out_h = new_size[0];
    out_w = new_size[1];
  } else {
    auto scale_tensor = ctx.Input<Tensor>("Scale");
    auto scale = ctx.Attr<std::vector<float>>("scale");
    if (scale_tensor != nullptr) {
      auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
      if (scale_data.size() > 1) {
        scale_h = scale_data[0];
        scale_w = scale_data[1];
      } else {
        scale_h = scale_data[0];
        scale_w = scale_data[0];
      }

      PADDLE_ENFORCE_EQ(
          scale_w > 0, true,
          platform::errors::InvalidArgument(
              "The scale_w in input 'Scale' Tensor of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
      PADDLE_ENFORCE_EQ(
          scale_h > 0, true,
          platform::errors::InvalidArgument(
              "The scale_h in input 'Scale' Tensor of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_h));
    } else {
      if (scale.size() > 1) {
        scale_w = scale[1];
        scale_h = scale[0];

        PADDLE_ENFORCE_EQ(
            scale_w > 0, true,
            platform::errors::InvalidArgument(
                "The scale_w in Attr(scale) of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_w));
        PADDLE_ENFORCE_EQ(
            scale_h > 0, true,
            platform::errors::InvalidArgument(
                "The scale_h in Attr(scale) of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_h));
      }
    }
    if (scale_w > 0. && scale_h > 0.) {
      out_h = static_cast<int>(in_h * scale_h);
      out_w = static_cast<int>(in_w * scale_w);
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
    float new_scale_h = 0.f;
    new_scale_h = (scale_h > 0) ? static_cast<float>(1. / scale_h)
                                : static_cast<float>(in_h) / out_h;
    ratio_h = (align_corners) ? static_cast<float>(in_h - 1) / (out_h - 1)
                              : static_cast<float>(new_scale_h);
  }
  if (out_w > 1) {
    float new_scale_w = 0.f;
    new_scale_w = (scale_w > 0) ? static_cast<float>(1. / scale_w)
                                : static_cast<float>(in_w) / out_w;
    ratio_w = (align_corners) ? static_cast<float>(in_w - 1) / (out_w - 1)
                              : static_cast<float>(new_scale_w);
  }

  int64_t in_hw = in_h * in_w;
  int64_t out_hw = out_h * out_w;
  int64_t in_chw = c * in_hw;
  int64_t out_chw = c * out_hw;

  auto pixelNum = n * out_chw;

  platform::GpuLaunchConfig config =
      platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), pixelNum);

  if ("nearest" == interp_method) {
    if (data_layout == DataLayout::kNCHW) {
      // get launch 3D config
      int nc = n * c;
      platform::GpuLaunchConfig config_3d =
          GetGpuLaunchConfig3D(ctx.cuda_device_context(), nc, out_h, out_w);
      KeNearestNeighborInterpNCHWFw<
          T><<<config_3d.block_per_grid, config_3d.thread_per_block, 0,
               ctx.cuda_device_context().stream()>>>(
          input_data, in_h, in_w, output_data, out_h, out_w, nc, ratio_h,
          ratio_w, align_corners);
    } else {
      int64_t cw = c * out_w;
      auto interp_divmods = FastDivModForInterpolate(c, out_chw, cw);
      KeNearestNeighborInterpFw<
          T><<<config.block_per_grid, config.thread_per_block, 0,
               ctx.cuda_device_context().stream()>>>(
          input_data, in_h, in_w, n, in_chw, output_data, out_h, out_w, n,
          out_chw, c, ratio_h, ratio_w, align_corners, interp_divmods);
    }
  } else if ("bilinear" == interp_method) {
    dim3 thread_num = config.thread_per_block;
#ifdef WITH_NV_JETSON
    if (config.compute_capability == 53 || config.compute_capability == 62) {
      thread_num = 512;
    }
#endif
    const T align_type_value = (align_mode == 0 && !align_corners) ? 0.5f : 0;
    if (data_layout == DataLayout::kNCHW) {
      // get launch 3D config
      int nc = n * c;
      platform::GpuLaunchConfig config_3d =
          GetGpuLaunchConfig3D(ctx.cuda_device_context(), nc, out_h, out_w);
      KeBilinearInterpNCHWFw<
          T><<<config_3d.block_per_grid, config_3d.thread_per_block, 0,
               ctx.cuda_device_context().stream()>>>(
          input_data, in_h, in_w, output_data, out_h, out_w, nc, ratio_h,
          ratio_w, align_type_value);
    } else {
      int64_t cw = c * out_w;
      auto interp_divmods = FastDivModForInterpolate(c, out_chw, cw);
      KeBilinearInterpFw<T><<<config.block_per_grid, thread_num, 0,
                              ctx.cuda_device_context().stream()>>>(
          input_data, in_h, in_w, n, in_chw, output_data, out_h, out_w, n,
          out_chw, c, ratio_h, ratio_w, align_type_value, interp_divmods);
    }
  } else if ("bicubic" == interp_method) {
#ifdef __HIPCC__
    constexpr int thread_per_block = 256;
#else
    constexpr int thread_per_block = 512;
#endif
    KeBicubicInterpFw<T><<<config.block_per_grid, thread_per_block, 0,
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
  float scale_w = -1;
  float scale_d = -1;
  float scale_h = -1;
  if (list_new_shape_tensor.size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(list_new_shape_tensor);
    out_d = new_size[0];
    out_h = new_size[1];
    out_w = new_size[2];
  } else {
    auto scale_tensor = ctx.Input<Tensor>("Scale");
    auto scale = ctx.Attr<std::vector<float>>("scale");
    if (scale_tensor != nullptr) {
      auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
      if (scale_data.size() > 1) {
        scale_d = scale_data[0];
        scale_h = scale_data[1];
        scale_w = scale_data[2];
      } else {
        scale_d = scale_data[0];
        scale_h = scale_data[0];
        scale_w = scale_data[0];
      }

      PADDLE_ENFORCE_EQ(
          scale_w > 0, true,
          platform::errors::InvalidArgument(
              "The scale_w in input 'Scale' Tensor of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
      PADDLE_ENFORCE_EQ(
          scale_h > 0, true,
          platform::errors::InvalidArgument(
              "The scale_h in input 'Scale' Tensor of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_h));
      PADDLE_ENFORCE_EQ(
          scale_d > 0, true,
          platform::errors::InvalidArgument(
              "The scale_d in input 'Scale' Tensor of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_d));
    } else {
      if (scale.size() > 1) {
        scale_d = scale[0];
        scale_h = scale[1];
        scale_w = scale[2];

        PADDLE_ENFORCE_EQ(
            scale_w > 0, true,
            platform::errors::InvalidArgument(
                "The scale_w in Attr(scale) of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_w));
        PADDLE_ENFORCE_EQ(
            scale_h > 0, true,
            platform::errors::InvalidArgument(
                "The scale_h in Attr(scale) of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_h));
        PADDLE_ENFORCE_EQ(
            scale_d > 0, true,
            platform::errors::InvalidArgument(
                "The scale_d in Attr(scale) of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_d));
      }
    }
    if (scale_d > 0. && scale_h > 0. && scale_w > 0.) {
      out_d = static_cast<int>(in_d * scale_d);
      out_h = static_cast<int>(in_h * scale_h);
      out_w = static_cast<int>(in_w * scale_w);
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
    float new_scale_d = 0.f;
    new_scale_d = (scale_d > 0) ? static_cast<float>(1. / scale_d)
                                : static_cast<float>(in_d) / out_d;
    ratio_d = (align_corners) ? static_cast<float>(in_d - 1) / (out_d - 1)
                              : static_cast<float>(new_scale_d);
  }
  if (out_h > 1) {
    float new_scale_h = 0.f;
    new_scale_h = (scale_h > 0) ? static_cast<float>(1. / scale_h)
                                : static_cast<float>(in_h) / out_h;
    ratio_h = (align_corners) ? static_cast<float>(in_h - 1) / (out_h - 1)
                              : static_cast<float>(new_scale_h);
  }
  if (out_w > 1) {
    float new_scale_w = 0.f;
    new_scale_w = (scale_w > 0) ? static_cast<float>(1. / scale_w)
                                : static_cast<float>(in_w) / out_w;
    ratio_w = (align_corners) ? static_cast<float>(in_w - 1) / (out_w - 1)
                              : static_cast<float>(new_scale_w);
  }

  int64_t in_dhw = in_d * in_h * in_w;
  int64_t out_dhw = out_d * out_h * out_w;
  int64_t in_cdhw = c * in_dhw;
  int64_t out_cdhw = c * out_dhw;

  auto pixelNum = n * out_cdhw;

  platform::GpuLaunchConfig config =
      platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), pixelNum);

  if ("trilinear" == interp_method) {
    KeTrilinearInterpFw<T><<<config.block_per_grid, config.thread_per_block, 0,
                             ctx.cuda_device_context().stream()>>>(
        input_data, in_d, in_h, in_w, n, in_cdhw, output_data, out_d, out_h,
        out_w, n, out_cdhw, c, ratio_d, ratio_h, ratio_w, align_corners,
        align_mode, data_layout);
  } else if ("nearest" == interp_method) {
    KeNearestNeighbor3DInterpFw<
        T><<<config.block_per_grid, config.thread_per_block, 0,
             ctx.cuda_device_context().stream()>>>(
        input_data, in_d, in_h, in_w, n, in_cdhw, output_data, out_d, out_h,
        out_w, n, out_cdhw, c, ratio_d, ratio_h, ratio_w, align_corners,
        data_layout);
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
  float scale_w = -1;
  auto scale_tensor = ctx.Input<Tensor>("Scale");
  auto scale = ctx.Attr<std::vector<float>>("scale");
  if (scale_tensor != nullptr) {
    auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
    scale_w = scale_data[0];
    PADDLE_ENFORCE_EQ(
        scale_w > 0, true,
        platform::errors::InvalidArgument(
            "The scale_w in input 'Scale' Tensor of Operator(interpolate) "
            "should be greater than 0, but received value is %d.",
            scale_w));
  } else {
    if (scale.size() > 0) {
      scale_w = scale[0];

      PADDLE_ENFORCE_EQ(
          scale_w > 0, true,
          platform::errors::InvalidArgument(
              "The scale_w in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
    }
  }
  if (scale_w > 0.) {
    out_w = static_cast<int>(in_w * scale_w);
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
  pten::funcs::SetConstant<platform::CUDADeviceContext, T> zero;
  zero(device_ctx, input_grad, static_cast<T>(0.0));

  if (in_w == out_w) {
    framework::TensorCopy(output_grad, ctx.GetPlace(), input_grad);
    return;
  }

  float ratio_w = 0.f;
  if (out_w > 1) {
    float new_scale_w = 0.f;
    new_scale_w = (scale_w > 0) ? static_cast<float>(1. / scale_w)
                                : static_cast<float>(in_w) / out_w;
    ratio_w = (align_corners) ? static_cast<float>(in_w - 1) / (out_w - 1)
                              : static_cast<float>(new_scale_w);
  }
  int64_t in_cw = c * in_w;
  int64_t out_cw = c * out_w;
  auto pixelNum = n * out_cw;

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
  float scale_h = -1;
  float scale_w = -1;
  auto scale_tensor = ctx.Input<Tensor>("Scale");
  auto scale = ctx.Attr<std::vector<float>>("scale");
  if (scale_tensor != nullptr) {
    auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
    if (scale_data.size() > 1) {
      scale_h = scale_data[0];
      scale_w = scale_data[1];
    } else {
      scale_h = scale_data[0];
      scale_w = scale_data[0];
    }

    PADDLE_ENFORCE_EQ(
        scale_w > 0, true,
        platform::errors::InvalidArgument(
            "The scale_w in input 'Scale' Tensor of Operator(interpolate) "
            "should be greater than 0, but received value is %d.",
            scale_w));
    PADDLE_ENFORCE_EQ(
        scale_h > 0, true,
        platform::errors::InvalidArgument(
            "The scale_h in input 'Scale' Tensor of Operator(interpolate) "
            "should be greater than 0, but received value is %d.",
            scale_h));
  } else {
    if (scale.size() > 1) {
      scale_w = scale[1];
      scale_h = scale[0];

      PADDLE_ENFORCE_EQ(
          scale_w > 0, true,
          platform::errors::InvalidArgument(
              "The scale_w in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
      PADDLE_ENFORCE_EQ(
          scale_h > 0, true,
          platform::errors::InvalidArgument(
              "The scale_h in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_h));
    }
  }
  if (scale_w > 0. && scale_h > 0.) {
    out_h = static_cast<int>(in_h * scale_h);
    out_w = static_cast<int>(in_w * scale_w);
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
  pten::funcs::SetConstant<platform::CUDADeviceContext, T> zero;
  zero(device_ctx, input_grad, static_cast<T>(0.0));

  if (in_h == out_h && in_w == out_w) {
    framework::TensorCopy(output_grad, ctx.GetPlace(), input_grad);
    return;
  }

  float ratio_h = 0.f;
  float ratio_w = 0.f;
  if (out_h > 1) {
    float new_scale_h = 0.f;
    new_scale_h = (scale_h > 0) ? static_cast<float>(1. / scale_h)
                                : static_cast<float>(in_h) / out_h;
    ratio_h = (align_corners) ? static_cast<float>(in_h - 1) / (out_h - 1)
                              : static_cast<float>(new_scale_h);
  }
  if (out_w > 1) {
    float new_scale_w = 0.f;
    new_scale_w = (scale_w > 0) ? static_cast<float>(1. / scale_w)
                                : static_cast<float>(in_w) / out_w;
    ratio_w = (align_corners) ? static_cast<float>(in_w - 1) / (out_w - 1)
                              : static_cast<float>(new_scale_w);
  }

  int64_t in_hw = in_h * in_w;
  int64_t out_hw = out_h * out_w;
  int64_t in_chw = c * in_hw;
  int64_t out_chw = c * out_hw;
  auto pixelNum = n * out_chw;

  platform::GpuLaunchConfig config =
      platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), pixelNum);

  if ("nearest" == interp_method) {
    KeNearestNeighborInterpBw<
        T><<<config.block_per_grid, config.thread_per_block, 0,
             ctx.cuda_device_context().stream()>>>(
        input_grad_data, in_h, in_w, n, in_chw, output_grad_data, out_h, out_w,
        n, out_chw, c, ratio_h, ratio_w, align_corners, data_layout);
  } else if ("bilinear" == interp_method) {
    const T align_type_value = (align_mode == 0 && !align_corners) ? 0.5f : 0;
    bool is_nchw = (data_layout == DataLayout::kNCHW) ? true : false;
    bool optimize_flag = false;
#ifndef __HIPCC__
    optimize_flag = (in_h < (out_h >> 6) && in_w < (out_w >> 6))
                        ? true
                        : ((in_h == 1 && in_w == 1) ? true : false);
#endif

    if (optimize_flag & is_nchw) {
      KeBilinearInterpBwShareMemory<
          T><<<config.block_per_grid, config.thread_per_block, 0,
               ctx.cuda_device_context().stream()>>>(
          input_grad_data, in_h, in_w, output_grad_data, out_h, out_w, n, c,
          ratio_h, ratio_w, align_type_value, is_nchw);
    } else {
      KeBilinearInterpBw<T><<<config.block_per_grid, config.thread_per_block, 0,
                              ctx.cuda_device_context().stream()>>>(
          input_grad_data, in_h, in_w, output_grad_data, out_h, out_w, n, c,
          ratio_h, ratio_w, align_type_value, is_nchw);
    }
  } else if ("bicubic" == interp_method) {
#ifdef __HIPCC__
    constexpr int thread_per_block = 256;
#else
    constexpr int thread_per_block = 512;
#endif
    KeBicubicInterpBw<T><<<config.block_per_grid, thread_per_block, 0,
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
  float scale_d = -1;
  float scale_h = -1;
  float scale_w = -1;
  auto scale_tensor = ctx.Input<Tensor>("Scale");
  auto scale = ctx.Attr<std::vector<float>>("scale");
  if (scale_tensor != nullptr) {
    auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
    if (scale_data.size() > 1) {
      scale_d = scale_data[0];
      scale_h = scale_data[1];
      scale_w = scale_data[2];
    } else {
      scale_d = scale_data[0];
      scale_h = scale_data[0];
      scale_w = scale_data[0];
    }
    PADDLE_ENFORCE_EQ(
        scale_w > 0, true,
        platform::errors::InvalidArgument(
            "The scale_w in input 'Scale' Tensor of Operator(interpolate) "
            "should be greater than 0, but received value is %d.",
            scale_w));
    PADDLE_ENFORCE_EQ(
        scale_h > 0, true,
        platform::errors::InvalidArgument(
            "The scale_h in input 'Scale' Tensor of Operator(interpolate) "
            "should be greater than 0, but received value is %d.",
            scale_h));
    PADDLE_ENFORCE_EQ(
        scale_d > 0, true,
        platform::errors::InvalidArgument(
            "The scale_d in input 'Scale' Tensor of Operator(interpolate) "
            "should be greater than 0, but received value is %d.",
            scale_d));
  } else {
    if (scale.size() > 1) {
      scale_d = scale[0];
      scale_h = scale[1];
      scale_w = scale[2];

      PADDLE_ENFORCE_EQ(
          scale_w > 0, true,
          platform::errors::InvalidArgument(
              "The scale_w in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
      PADDLE_ENFORCE_EQ(
          scale_h > 0, true,
          platform::errors::InvalidArgument(
              "The scale_h in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_h));
      PADDLE_ENFORCE_EQ(
          scale_d > 0, true,
          platform::errors::InvalidArgument(
              "The scale_d in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_d));
    }
  }
  if (scale_d > 0. && scale_h > 0. && scale_w > 0.) {
    out_d = static_cast<int>(in_d * scale_d);
    out_h = static_cast<int>(in_h * scale_h);
    out_w = static_cast<int>(in_w * scale_w);
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
  pten::funcs::SetConstant<platform::CUDADeviceContext, T> zero;
  zero(device_ctx, input_grad, static_cast<T>(0.0));

  if (in_d == out_d && in_h == out_h && in_w == out_w) {
    framework::TensorCopy(output_grad, ctx.GetPlace(), input_grad);
    return;
  }

  float ratio_d = 0.f;
  float ratio_h = 0.f;
  float ratio_w = 0.f;
  if (out_d > 1) {
    float new_scale_d = 0.f;
    new_scale_d = (scale_d > 0) ? static_cast<float>(1. / scale_d)
                                : static_cast<float>(in_d) / out_d;
    ratio_d = (align_corners) ? static_cast<float>(in_d - 1) / (out_d - 1)
                              : static_cast<float>(new_scale_d);
  }
  if (out_h > 1) {
    float new_scale_h = 0.f;
    new_scale_h = (scale_h > 0) ? static_cast<float>(1. / scale_h)
                                : static_cast<float>(in_h) / out_h;
    ratio_h = (align_corners) ? static_cast<float>(in_h - 1) / (out_h - 1)
                              : static_cast<float>(new_scale_h);
  }
  if (out_w > 1) {
    float new_scale_w = 0.f;
    new_scale_w = (scale_w > 0) ? static_cast<float>(1. / scale_w)
                                : static_cast<float>(in_w) / out_w;
    ratio_w = (align_corners) ? static_cast<float>(in_w - 1) / (out_w - 1)
                              : static_cast<float>(new_scale_w);
  }

  int64_t in_dhw = in_d * in_h * in_w;
  int64_t out_dhw = out_d * out_h * out_w;
  int64_t in_cdhw = c * in_dhw;
  int64_t out_cdhw = c * out_dhw;

  auto pixelNum = n * out_cdhw;

  platform::GpuLaunchConfig config =
      platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), pixelNum);

  if ("trilinear" == interp_method) {
    KeTrilinearInterpBw<T><<<config.block_per_grid, config.thread_per_block, 0,
                             ctx.cuda_device_context().stream()>>>(
        input_grad_data, in_d, in_h, in_w, n, in_cdhw, output_grad_data, out_d,
        out_h, out_w, n, out_cdhw, c, ratio_d, ratio_h, ratio_w, align_corners,
        align_mode, data_layout);
  } else if ("nearest" == interp_method) {
    KeNearestNeighbor3DInterpBw<
        T><<<config.block_per_grid, config.thread_per_block, 0,
             ctx.cuda_device_context().stream()>>>(
        input_grad_data, in_d, in_h, in_w, n, in_cdhw, output_grad_data, out_d,
        out_h, out_w, n, out_cdhw, c, ratio_d, ratio_h, ratio_w, align_corners,
        data_layout);
  }
}

template <typename T>
class InterpolateOpV2CUDAKernel : public framework::OpKernel<T> {
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
class InterpolateV2GradOpCUDAKernel : public framework::OpKernel<T> {
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
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(bilinear_interp_v2,
                        ops::InterpolateOpV2CUDAKernel<float>,
                        ops::InterpolateOpV2CUDAKernel<double>,
                        ops::InterpolateOpV2CUDAKernel<int>);
REGISTER_OP_CUDA_KERNEL(bilinear_interp_v2_grad,
                        ops::InterpolateV2GradOpCUDAKernel<float>,
                        ops::InterpolateV2GradOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(nearest_interp_v2,
                        ops::InterpolateOpV2CUDAKernel<float>,
                        ops::InterpolateOpV2CUDAKernel<double>,
                        ops::InterpolateOpV2CUDAKernel<int64_t>,
                        ops::InterpolateOpV2CUDAKernel<int>);
REGISTER_OP_CUDA_KERNEL(nearest_interp_v2_grad,
                        ops::InterpolateV2GradOpCUDAKernel<float>,
                        ops::InterpolateV2GradOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(trilinear_interp_v2,
                        ops::InterpolateOpV2CUDAKernel<float>,
                        ops::InterpolateOpV2CUDAKernel<double>,
                        ops::InterpolateOpV2CUDAKernel<int>);
REGISTER_OP_CUDA_KERNEL(trilinear_interp_v2_grad,
                        ops::InterpolateV2GradOpCUDAKernel<float>,
                        ops::InterpolateV2GradOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(linear_interp_v2, ops::InterpolateOpV2CUDAKernel<float>,
                        ops::InterpolateOpV2CUDAKernel<double>,
                        ops::InterpolateOpV2CUDAKernel<int>);
REGISTER_OP_CUDA_KERNEL(linear_interp_v2_grad,
                        ops::InterpolateV2GradOpCUDAKernel<float>,
                        ops::InterpolateV2GradOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(bicubic_interp_v2,
                        ops::InterpolateOpV2CUDAKernel<float>,
                        ops::InterpolateOpV2CUDAKernel<double>,
                        ops::InterpolateOpV2CUDAKernel<int>);
REGISTER_OP_CUDA_KERNEL(bicubic_interp_v2_grad,
                        ops::InterpolateV2GradOpCUDAKernel<float>,
                        ops::InterpolateV2GradOpCUDAKernel<double>);
