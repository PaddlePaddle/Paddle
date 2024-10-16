// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/interpolate_grad_kernel.h"

#include "paddle/common/layout.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/interpolate_function.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/primitive/datamover_primitives.h"

namespace phi {

template <typename T>
__forceinline__ __device__ void PreCalculatorForLinearInterpInputIndex(
    int* in_img_idx,
    int* x_id,
    T* lambda1,
    T* lambda2,
    T src_x,
    const int in_img_x) {
  src_x = max(src_x, static_cast<T>(0));
  T src_x_floor = floorf(src_x);
  T frac_part = src_x - src_x_floor;
  *lambda1 = frac_part;
  *lambda2 = static_cast<T>(1) - frac_part;
  *in_img_idx = static_cast<int>(src_x_floor);
  *x_id = (*in_img_idx < in_img_x - 1);
}

template <typename T>
__global__ void KeLinearInterpBw(T* in,
                                 const size_t in_img_w,
                                 const size_t input_w,
                                 const T* out,
                                 const size_t out_img_w,
                                 const size_t output_h,
                                 const size_t output_w,
                                 const size_t num_channels,
                                 const float ratio_w,
                                 const bool align_corners,
                                 const int align_mode,
                                 const DataLayout data_layout) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  bool align_flag = (align_mode == 0 && !align_corners);
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
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

    MT src_w = ratio_w * (out_img_idx + 0.5) - 0.5;
    src_w = (src_w > 0) ? src_w : 0;
    MT w1lambda =
        align_flag ? src_w - in_img_idx : ratio_w * out_img_idx - in_img_idx;
    MT w2lambda = 1.0 - w1lambda;

    T* in_pos;
    if (data_layout == DataLayout::kNCHW) {
      in_pos = &in[out_id_h * input_w + channel_id * in_img_size + in_img_idx];
    } else {
      in_pos = &in[out_id_h * input_w + in_img_idx * num_channels + channel_id];
    }
    const T* out_pos = &out[out_id_w];

    if (data_layout == DataLayout::kNCHW) {
      phi::CudaAtomicAdd(
          &in_pos[0], static_cast<T>(w2lambda * static_cast<MT>(out_pos[0])));
      phi::CudaAtomicAdd(
          &in_pos[w_id],
          static_cast<T>(w1lambda * static_cast<MT>(out_pos[0])));
    } else {
      phi::CudaAtomicAdd(
          &in_pos[0], static_cast<T>(w2lambda * static_cast<MT>(out_pos[0])));
      phi::CudaAtomicAdd(
          &in_pos[w_id * num_channels],
          static_cast<T>(w1lambda * static_cast<MT>(out_pos[0])));
    }
  }
}

template <typename T>
__global__ void KeNearestNeighborInterpNCHWBw(T* in,
                                              const size_t in_img_h,
                                              const size_t in_img_w,
                                              const T* out,
                                              const size_t out_img_h,
                                              const size_t out_img_w,
                                              const size_t nc,
                                              const float ratio_h,
                                              const float ratio_w,
                                              const bool align_corners) {
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
      T* in_pos = &in[in_index];
      const T out_pos = out[out_index];
      phi::CudaAtomicAdd(in_pos, out_pos);
      in_index += in_index_stride;
      out_index += out_index_stride;
      nc_id += nc_stride;
    }
  }
}

template <typename T>
__global__ void KeNearestNeighborInterpBw(
    T* in,
    const size_t in_img_h,
    const size_t in_img_w,
    const size_t input_h,
    const size_t input_w,
    const T* out,
    const size_t out_img_h,
    const size_t out_img_w,
    const size_t output_h,
    const size_t output_w,
    const size_t num_channels,
    const float ratio_h,
    const float ratio_w,
    const bool align_corners,
    funcs::FastDivModForInterpolate divmods) {
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

    T* in_pos = &in[out_id_h * input_w + in_img_idy * in_img_w * num_channels +
                    in_img_idx * num_channels + channel_id];

    const T out_pos = out[tid];
    phi::CudaAtomicAdd(in_pos, out_pos);
  }
}

/* Calculate the minimum of partial elements in a block */
template <typename T>
__inline__ __device__ T PartialBlockMin(T val,
                                        size_t threads_num_in_block,
                                        unsigned mask) {
  __shared__ T shared[WARP_SIZE];
  __shared__ T shared_last_val;
  __shared__ int shared_last_idx;
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;
  int threshold = (threads_num_in_block & (-WARP_SIZE));

  if (threadIdx.x < threshold) {
    shared_last_idx = (threshold >> 5) - 1;
    val = phi::funcs::WarpReduceMin(val, mask);
    if (lane == 0) {
      shared[wid] = val;
    }
  } else {
    shared_last_val = std::numeric_limits<T>::max();
    phi::CudaAtomicMin(&shared_last_val, val);
    shared[wid] = shared_last_val;
    shared_last_idx = wid;
  }
  __syncthreads();

  if (threadIdx.x < threshold) {
    val = (lane <= shared_last_idx) ? shared[lane]
                                    : std::numeric_limits<T>::max();
    val = phi::funcs::WarpReduceMin(val, mask);
    shared_last_val = val;
  }
  __syncthreads();
  if (threadIdx.x >= threshold) {
    val = shared_last_val;
  }
  return val;
}

template <typename T>
__global__ void KeBilinearInterpBwShareMemory(T* in,
                                              const int in_h,
                                              const int in_w,
                                              const T* __restrict__ out,
                                              const int out_h,
                                              const int out_w,
                                              const int n,
                                              const int num_channels,
                                              float ratio_h,
                                              float ratio_w,
                                              const float align_type_value,
                                              bool is_nchw) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  __shared__ MT s_data[2][1024];
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
    MT value = static_cast<MT>(out[out_id_h * out_chw + out_id_w]);

    int channel_id = out_id_w / out_img_size;
    int out_img_idy = (out_id_w % out_img_size) / out_w;
    int out_img_idx = tid % out_w;

    int in_img_idx, in_img_idy, w_id, h_id;
    MT w1lambda, h1lambda, w2lambda, h2lambda;
    MT src_w = static_cast<MT>(ratio_w * (out_img_idx + align_type_value) -
                               align_type_value);
    MT src_h = static_cast<MT>(ratio_h * (out_img_idy + align_type_value) -
                               align_type_value);

    PreCalculatorForLinearInterpInputIndex(
        &in_img_idx, &w_id, &w1lambda, &w2lambda, src_w, in_w);
    PreCalculatorForLinearInterpInputIndex(
        &in_img_idy, &h_id, &h1lambda, &h2lambda, src_h, in_h);

    // top_left_index is just input_index.
    int input_index = out_id_h * in_chw + channel_id * in_img_size +
                      in_img_idy * in_w + in_img_idx;
    int top_right_index = input_index + w_id;
    int bot_left_index = input_index + h_id * in_w;
    int bot_right_index = input_index + h_id * in_w + w_id;
    int in_top_min_index, in_bot_min_index;

    s_data[0][threadIdx.x] = static_cast<MT>(0);
    s_data[1][threadIdx.x] = static_cast<MT>(0);
    int remain = nthreads - (tid & (-blockDim.x));
    int in_top_max_index =
        phi::funcs::BlockReduceMax(top_right_index, FINAL_MASK);
    int in_bot_max_index =
        phi::funcs::BlockReduceMax(bot_right_index, FINAL_MASK);

    if (remain > blockDim.x) {
      in_top_min_index = phi::funcs::BlockReduceMin(input_index, FINAL_MASK);
      in_bot_min_index = phi::funcs::BlockReduceMin(bot_left_index, FINAL_MASK);
    } else {
      in_top_min_index = PartialBlockMin(input_index, remain, FINAL_MASK);
      in_bot_min_index = PartialBlockMin(bot_left_index, remain, FINAL_MASK);
    }
    int upper_limit_share_idx = (in_top_max_index - in_top_min_index) >
                                        (in_bot_max_index - in_bot_min_index)
                                    ? (in_top_max_index - in_top_min_index)
                                    : (in_bot_max_index - in_bot_min_index);
    if (h_id != 0) {
      phi::CudaAtomicAdd(&s_data[0][input_index - in_top_min_index],
                         h2lambda * w2lambda * value);
      phi::CudaAtomicAdd(&s_data[0][top_right_index - in_top_min_index],
                         h2lambda * w1lambda * value);
      phi::CudaAtomicAdd(&s_data[1][bot_left_index - in_bot_min_index],
                         h1lambda * w2lambda * value);
      phi::CudaAtomicAdd(&s_data[1][bot_right_index - in_bot_min_index],
                         h1lambda * w1lambda * value);
    } else {
      phi::CudaAtomicAdd(&s_data[0][top_right_index - in_top_min_index],
                         (h2lambda + h1lambda) * w1lambda * value);
      phi::CudaAtomicAdd(&s_data[1][bot_left_index - in_bot_min_index],
                         (h1lambda + h2lambda) * w2lambda * value);
    }
    __syncthreads();

    if (threadIdx.x <= upper_limit_share_idx) {
      phi::CudaAtomicAdd(&in[in_top_min_index + threadIdx.x],
                         static_cast<T>(s_data[0][threadIdx.x]));
      phi::CudaAtomicAdd(&in[in_bot_min_index + threadIdx.x],
                         static_cast<T>(s_data[1][threadIdx.x]));
    }
  }
}

__device__ __forceinline__ int GetInputIndex(const size_t nc,
                                             const int height,
                                             const int width,
                                             const int h,
                                             const int w) {
  return (nc * height + h) * width + w;
}

template <typename T>
__global__ void KeBilinearInterpNCHWBw(T* in,
                                       const int in_h,
                                       const int in_w,
                                       const int out_h,
                                       const int out_w,
                                       const int n,
                                       const int num_channels,
                                       float ratio_h,
                                       float ratio_w,
                                       const T* __restrict__ out,
                                       const float align_type_value) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int num_out = n * num_channels * out_h * out_w;
  const int num_in = n * num_channels * in_h * in_w;
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

  // Restricted parallelism if ratio_w is over threshold
  // to avoid atomic contention overhead.
  // This threshold 0.5f is come up with extensive quantitative analysis,
  // corresponding to 2x or larger scale factor in W axis.
  if (ratio_w < 0.5f) [[likely]] {  // NOLINT
    if (index < num_in) {
      int index_tmp = index;
      const int w1 = index_tmp % in_w;
      index_tmp /= in_w;
      const int h1 = index_tmp % in_h;
      const int nc = index_tmp / in_h;

      MT d2val_sum = 0.0f;

      // Precompute constants
      const MT inv_ratio_h = 1.0f / ratio_h;
      const MT inv_ratio_w = 1.0f / ratio_w;

      // Compute the range of output pixels (h2_min, h2_max) that could affect
      // input pixel h1
      const MT h2r_min =
          (h1 - 1 + align_type_value) * inv_ratio_h - align_type_value;
      const int h2_min = max(static_cast<int>(ceilf(h2r_min)), 0);

      const MT h2r_max =
          (h1 + 1 + align_type_value) * inv_ratio_h - align_type_value;
      const int h2_max = min(static_cast<int>(floorf(h2r_max)), out_h - 1);

      // Compute the range of output pixels (w2_min, w2_max) that could affect
      // input pixel w1
      const MT w2r_min =
          (w1 - 1 + align_type_value) * inv_ratio_w - align_type_value;
      const int w2_min = max(static_cast<int>(ceilf(w2r_min)), 0);

      const MT w2r_max =
          (w1 + 1 + align_type_value) * inv_ratio_w - align_type_value;
      const int w2_max = min(static_cast<int>(floorf(w2r_max)), out_w - 1);

      for (int h2 = h2_min; h2 <= h2_max; ++h2) {
        const MT src_y = ratio_h * (h2 + align_type_value) - align_type_value;
        int h1_, y_id;
        MT h1lambda, h0lambda;
        PreCalculatorForLinearInterpInputIndex(
            &h1_, &y_id, &h1lambda, &h0lambda, src_y, in_h);

        if (h1 != h1_ && h1 != h1_ + y_id) [[unlikely]] {
          continue;
        }

        for (int w2 = w2_min; w2 <= w2_max; ++w2) {
          int w1_, x_id;
          const MT src_x = ratio_w * (w2 + align_type_value) - align_type_value;
          MT w1lambda, w0lambda;
          PreCalculatorForLinearInterpInputIndex(
              &w1_, &x_id, &w1lambda, &w0lambda, src_x, in_w);
          if (w1 != w1_ && w1 != w1_ + x_id) [[unlikely]] {
            continue;
          }

          const MT grad_output = out[nc * out_h * out_w + h2 * out_w + w2];

          float hlambda = (h1 == h1_) ? h0lambda : 0.0f;
          hlambda += (h1 == h1_ + y_id) ? h1lambda : 0.0f;

          float wlambda = (w1 == w1_) ? w0lambda : 0.0f;
          wlambda += (w1 == w1_ + x_id) ? w1lambda : 0.0f;

          d2val_sum += hlambda * wlambda * grad_output;
        }
      }
      in[index] = static_cast<T>(d2val_sum);
    }
  } else [[unlikely]] {  // NOLINT
    for (; index < num_out; index += stride) {
      int index_tmp = index;
      int w2 = index_tmp % out_w;
      index_tmp /= out_w;
      int h2 = index_tmp % out_h;
      int nc = index_tmp / out_h;

      int h1, y_id;
      MT h1lambda, h0lambda;
      MT src_y =
          static_cast<MT>(ratio_h * (h2 + align_type_value) - align_type_value);

      PreCalculatorForLinearInterpInputIndex(
          &h1, &y_id, &h1lambda, &h0lambda, src_y, in_h);
      int w1, x_id;
      MT w1lambda, w0lambda;
      MT src_x =
          static_cast<MT>(ratio_w * (w2 + align_type_value) - align_type_value);
      PreCalculatorForLinearInterpInputIndex(
          &w1, &x_id, &w1lambda, &w0lambda, src_x, in_w);

      MT d2val = static_cast<MT>(out[index]);

      phi::CudaAtomicAdd(in + GetInputIndex(nc, in_h, in_w, h1, w1),
                         static_cast<T>(h0lambda * w0lambda * d2val));
      phi::CudaAtomicAdd(in + GetInputIndex(nc, in_h, in_w, h1, w1 + x_id),
                         static_cast<T>(h0lambda * w1lambda * d2val));
      phi::CudaAtomicAdd(in + GetInputIndex(nc, in_h, in_w, h1 + y_id, w1),
                         static_cast<T>(h1lambda * w0lambda * d2val));
      phi::CudaAtomicAdd(
          in + GetInputIndex(nc, in_h, in_w, h1 + y_id, w1 + x_id),
          static_cast<T>(h1lambda * w1lambda * d2val));
    }
  }
}

template <typename T>
__global__ void KeBilinearInterpBw(T* in,
                                   const int in_h,
                                   const int in_w,
                                   const T* __restrict__ out,
                                   const int out_h,
                                   const int out_w,
                                   const int n,
                                   const int out_chw,
                                   const int num_channels,
                                   float ratio_h,
                                   float ratio_w,
                                   const float align_type_value,
                                   funcs::FastDivModForInterpolate divmods) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int in_chw = in_h * in_w * num_channels;
  int nthreads = n * out_chw;
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

  for (; tid < nthreads; tid += stride) {
    auto out_id_divmod = divmods.output_w_div.Divmod(tid);
    int out_id_h = out_id_divmod.val[0];
    int out_id_w = out_id_divmod.val[1];

    int channel_id = divmods.channels_div.Divmod(tid).val[1];
    auto outimg_id_divmod = divmods.output_wc_div.Divmod(out_id_w);
    int out_img_idy = outimg_id_divmod.val[0];
    int out_img_idx =
        divmods.channels_div.Divmod(outimg_id_divmod.val[1]).val[0];

    int in_img_idx, in_img_idy, w_id, h_id;
    MT w1lambda, h1lambda, w2lambda, h2lambda;
    MT src_w = static_cast<MT>(ratio_w * (out_img_idx + align_type_value) -
                               align_type_value);
    MT src_h = static_cast<MT>(ratio_h * (out_img_idy + align_type_value) -
                               align_type_value);

    PreCalculatorForLinearInterpInputIndex(
        &in_img_idx, &w_id, &w1lambda, &w2lambda, src_w, in_w);
    PreCalculatorForLinearInterpInputIndex(
        &in_img_idy, &h_id, &h1lambda, &h2lambda, src_h, in_h);

    MT value = static_cast<MT>(out[tid]);
    T* in_pos = &in[out_id_h * in_chw + in_img_idy * in_w * num_channels +
                    in_img_idx * num_channels + channel_id];
    phi::CudaAtomicAdd(&in_pos[0], static_cast<T>(h2lambda * w2lambda * value));
    phi::CudaAtomicAdd(&in_pos[w_id * num_channels],
                       static_cast<T>(h2lambda * w1lambda * value));
    phi::CudaAtomicAdd(&in_pos[h_id * in_w * num_channels],
                       static_cast<T>(h1lambda * w2lambda * value));
    phi::CudaAtomicAdd(
        &in_pos[h_id * in_w * num_channels + w_id * num_channels],
        static_cast<T>(h1lambda * w1lambda * value));
  }
}

template <typename T>
__global__ void KeBicubicInterpBw(T* in,
                                  const size_t in_img_h,
                                  const size_t in_img_w,
                                  const size_t input_h,
                                  const size_t input_w,
                                  const T* out,
                                  const size_t out_img_h,
                                  const size_t out_img_w,
                                  const size_t output_h,
                                  const size_t output_w,
                                  const size_t num_channels,
                                  const float ratio_h,
                                  const float ratio_w,
                                  const bool align_corners,
                                  const DataLayout data_layout) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

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

    MT in_img_idy = align_corners ? ratio_h * out_img_idy
                                  : ratio_h * (out_img_idy + 0.5) - 0.5;
    int input_y = floorf(static_cast<float>(in_img_idy));

    const MT y_t = in_img_idy - input_y;
    MT in_img_idx = align_corners ? ratio_w * out_img_idx
                                  : ratio_w * (out_img_idx + 0.5) - 0.5;
    int input_x = floorf(static_cast<float>(in_img_idx));
    const MT x_t = in_img_idx - input_x;

    MT x_coeffs[4];
    MT y_coeffs[4];

    funcs::get_cubic_upsample_coefficients<MT>(x_coeffs, x_t);
    funcs::get_cubic_upsample_coefficients<MT>(y_coeffs, y_t);

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
        phi::CudaAtomicAdd(&in_pos[0],
                           static_cast<T>(static_cast<MT>(out_pos[0]) *
                                          y_coeffs[j] * x_coeffs[i]));
      }
    }
  }
}

template <typename T>
__global__ void KeTrilinearInterpBw(T* in,
                                    const size_t in_img_d,
                                    const size_t in_img_h,
                                    const size_t in_img_w,
                                    const size_t input_h,
                                    const size_t input_w,
                                    const T* out,
                                    const size_t out_img_d,
                                    const size_t out_img_h,
                                    const size_t out_img_w,
                                    const size_t output_h,
                                    const size_t output_w,
                                    const size_t num_channels,
                                    const float ratio_d,
                                    const float ratio_h,
                                    const float ratio_w,
                                    const bool align_corners,
                                    const int align_mode,
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
    T src_d = static_cast<T>(ratio_d * (out_img_idt + 0.5) - 0.5);
    src_d = (src_d > static_cast<T>(0)) ? src_d : static_cast<T>(0);
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    T d1lambda = align_flag
                     ? static_cast<T>(static_cast<MT>(src_d) - in_img_idt)
                     : static_cast<T>(ratio_d * out_img_idt - in_img_idt);
    T d2lambda = static_cast<T>(1.0) - d1lambda;

    int in_img_idy = align_flag
                         ? static_cast<int>(ratio_h * (out_img_idy + 0.5) - 0.5)
                         : static_cast<int>(ratio_h * out_img_idy);
    in_img_idy = (in_img_idy > 0) ? in_img_idy : 0;
    int h_id = (in_img_idy < in_img_h - 1) ? 1 : 0;
    T src_h = static_cast<T>(ratio_h * (out_img_idy + 0.5) - 0.5);
    src_h = (src_h > static_cast<T>(0)) ? src_h : static_cast<T>(0);
    T h1lambda = align_flag
                     ? static_cast<T>(static_cast<MT>(src_h) - in_img_idy)
                     : static_cast<T>(ratio_h * out_img_idy - in_img_idy);
    T h2lambda = static_cast<T>(1.0) - h1lambda;

    int in_img_idx = align_flag
                         ? static_cast<int>(ratio_w * (out_img_idx + 0.5) - 0.5)
                         : static_cast<int>(ratio_w * out_img_idx);
    in_img_idx = (in_img_idx > 0) ? in_img_idx : 0;
    int w_id = (in_img_idx < in_img_w - 1) ? 1 : 0;
    T src_w = static_cast<T>(ratio_w * (out_img_idx + 0.5) - 0.5);
    src_w = (src_w > static_cast<T>(0)) ? src_w : static_cast<T>(0);
    T w1lambda = align_flag
                     ? static_cast<T>(static_cast<MT>(src_w) - in_img_idx)
                     : static_cast<T>(ratio_w * out_img_idx - in_img_idx);
    T w2lambda = static_cast<T>(1.0) - w1lambda;

    if (data_layout == DataLayout::kNCHW) {
      int in_pos1_idx = out_id_h * input_w + channel_id * in_img_size +
                        (in_img_idt * in_img_h + in_img_idy) * in_img_w +
                        in_img_idx;
      T* in_pos1 = &in[in_pos1_idx];
      int in_pos2_idx = in_pos1_idx + d_id * in_img_h * in_img_w;
      T* in_pos2 = &in[in_pos2_idx];

      const T* out_pos = &out[out_id_h * output_w + out_id_w];

      // trilinear interpolation grad
      phi::CudaAtomicAdd(&in_pos1[0],
                         d2lambda * h2lambda * w2lambda * out_pos[0]);
      phi::CudaAtomicAdd(&in_pos1[w_id],
                         d2lambda * h2lambda * w1lambda * out_pos[0]);
      phi::CudaAtomicAdd(&in_pos1[h_id * in_img_w],
                         d2lambda * h1lambda * w2lambda * out_pos[0]);
      phi::CudaAtomicAdd(&in_pos1[h_id * in_img_w + w_id],
                         d2lambda * h1lambda * w1lambda * out_pos[0]);
      phi::CudaAtomicAdd(&in_pos2[0],
                         d1lambda * h2lambda * w2lambda * out_pos[0]);
      phi::CudaAtomicAdd(&in_pos2[w_id],
                         d1lambda * h2lambda * w1lambda * out_pos[0]);
      phi::CudaAtomicAdd(&in_pos2[h_id * in_img_w],
                         d1lambda * h1lambda * w2lambda * out_pos[0]);
      phi::CudaAtomicAdd(&in_pos2[h_id * in_img_w + w_id],
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
      phi::CudaAtomicAdd(&in_pos1[0],
                         d2lambda * h2lambda * w2lambda * out_pos[0]);
      phi::CudaAtomicAdd(&in_pos1[w_id * num_channels],
                         d2lambda * h2lambda * w1lambda * out_pos[0]);
      phi::CudaAtomicAdd(&in_pos1[h_id * in_img_w * num_channels],
                         d2lambda * h1lambda * w2lambda * out_pos[0]);
      phi::CudaAtomicAdd(
          &in_pos1[h_id * in_img_w * num_channels + w_id * num_channels],
          d2lambda * h1lambda * w1lambda * out_pos[0]);
      phi::CudaAtomicAdd(&in_pos2[0],
                         d1lambda * h2lambda * w2lambda * out_pos[0]);
      phi::CudaAtomicAdd(&in_pos2[w_id * num_channels],
                         d1lambda * h2lambda * w1lambda * out_pos[0]);
      phi::CudaAtomicAdd(&in_pos2[h_id * in_img_w * num_channels],
                         d1lambda * h1lambda * w2lambda * out_pos[0]);
      phi::CudaAtomicAdd(
          &in_pos2[h_id * in_img_w * num_channels + w_id * num_channels],
          d1lambda * h1lambda * w1lambda * out_pos[0]);
    }
  }
}

template <typename T>
__global__ void KeNearestNeighbor3DInterpBw(T* in,
                                            const size_t in_img_d,
                                            const size_t in_img_h,
                                            const size_t in_img_w,
                                            const size_t input_h,
                                            const size_t input_w,
                                            const T* out,
                                            const size_t out_img_d,
                                            const size_t out_img_h,
                                            const size_t out_img_w,
                                            const size_t output_h,
                                            const size_t output_w,
                                            const size_t num_channels,
                                            const float ratio_d,
                                            const float ratio_h,
                                            const float ratio_w,
                                            const bool align_corners,
                                            const DataLayout data_layout) {
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
    phi::CudaAtomicAdd(in_pos, out_pos);
  }
}

template <typename T, typename Context>
static void Interpolate1DCUDABwd(
    const Context& dev_ctx,
    const DenseTensor& input,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const DenseTensor& output_grad,
    const std::string& data_layout_str,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* input_grad) {
  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  funcs::ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  float scale_w = -1;
  if (scale_tensor) {
    auto scale_data =
        funcs::get_new_data_from_tensor<float>(scale_tensor.get_ptr());
    scale_w = scale_data[0];
    PADDLE_ENFORCE_EQ(
        scale_w > 0,
        true,
        errors::InvalidArgument(
            "The scale_w in input 'Scale' Tensor of Operator(interpolate) "
            "should be greater than 0, but received value is %d.",
            scale_w));
  } else {
    if (scale.size() > 0) {
      scale_w = scale[0];

      PADDLE_ENFORCE_EQ(
          scale_w > 0,
          true,
          errors::InvalidArgument(
              "The scale_w in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
    }
  }
  if (scale_w > 0.) {
    out_w = static_cast<int>(in_w * scale_w);
  }

  if (out_size) {
    DenseTensor sizes;
    phi::Copy(dev_ctx, *out_size, phi::CPUPlace(), true, &sizes);

    auto size_data = sizes.data<int>();
    out_w = size_data[0];
  }
  if (size_tensor && size_tensor->size() > 0) {
    // have size tensor
    auto new_size = funcs::get_new_shape(size_tensor.get());
    out_w = new_size[0];
  }

  auto* output_grad_data = output_grad.data<T>();
  phi::DDim dim_grad;
  if (data_layout == DataLayout::kNCHW) {
    dim_grad = {n, c, in_w};
  } else {
    dim_grad = {n, in_w, c};
  }
  input_grad->Resize(dim_grad);
  auto* input_grad_data = dev_ctx.template Alloc<T>(input_grad);

  phi::funcs::SetConstant<Context, T> zero;
  zero(dev_ctx, input_grad, static_cast<T>(0.0));

  if (in_w == out_w) {
    phi::Copy(dev_ctx, output_grad, dev_ctx.GetPlace(), false, input_grad);
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

  backends::gpu::GpuLaunchConfig config =
      backends::gpu::GetGpuLaunchConfig1D(dev_ctx, pixelNum);

  if ("linear" == interp_method) {
    KeLinearInterpBw<T><<<config.block_per_grid,
                          config.thread_per_block,
                          0,
                          dev_ctx.stream()>>>(input_grad_data,
                                              in_w,
                                              in_cw,
                                              output_grad_data,
                                              out_w,
                                              n,
                                              out_cw,
                                              c,
                                              ratio_w,
                                              align_corners,
                                              align_mode,
                                              data_layout);
  }
}

template <typename T, typename Context>
static void Interpolate2DCUDABwd(
    const Context& dev_ctx,
    const DenseTensor& input,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const DenseTensor& output_grad,
    const std::string& data_layout_str,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* input_grad) {
  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  funcs::ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  float scale_h = -1;
  float scale_w = -1;
  if (scale_tensor) {
    auto scale_data =
        funcs::get_new_data_from_tensor<float>(scale_tensor.get_ptr());
    if (scale_data.size() > 1) {
      scale_h = scale_data[0];
      scale_w = scale_data[1];
    } else {
      scale_h = scale_data[0];
      scale_w = scale_data[0];
    }

    PADDLE_ENFORCE_EQ(
        scale_w > 0,
        true,
        errors::InvalidArgument(
            "The scale_w in input 'Scale' Tensor of Operator(interpolate) "
            "should be greater than 0, but received value is %d.",
            scale_w));
    PADDLE_ENFORCE_EQ(
        scale_h > 0,
        true,
        errors::InvalidArgument(
            "The scale_h in input 'Scale' Tensor of Operator(interpolate) "
            "should be greater than 0, but received value is %d.",
            scale_h));
  } else {
    if (scale.size() > 1) {
      scale_w = scale[1];
      scale_h = scale[0];

      PADDLE_ENFORCE_EQ(
          scale_w > 0,
          true,
          errors::InvalidArgument(
              "The scale_w in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
      PADDLE_ENFORCE_EQ(
          scale_h > 0,
          true,
          errors::InvalidArgument(
              "The scale_h in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_h));
    }
  }
  if (scale_w > 0. && scale_h > 0.) {
    out_h = static_cast<int>(in_h * scale_h);
    out_w = static_cast<int>(in_w * scale_w);
  }

  if (out_size) {
    DenseTensor sizes;
    phi::Copy(dev_ctx, *out_size, phi::CPUPlace(), true, &sizes);
    auto size_data = sizes.data<int>();
    out_h = size_data[0];
    out_w = size_data[1];
  }
  if (size_tensor && size_tensor->size() > 0) {
    // have size tensor
    auto new_size = funcs::get_new_shape(size_tensor.get());
    out_h = new_size[0];
    out_w = new_size[1];
  }

  auto* output_grad_data = output_grad.data<T>();
  phi::DDim dim_grad;
  if (data_layout == DataLayout::kNCHW) {
    dim_grad = {n, c, in_h, in_w};
  } else {
    dim_grad = {n, in_h, in_w, c};
  }
  input_grad->Resize(dim_grad);
  auto* input_grad_data = dev_ctx.template Alloc<T>(input_grad);
  phi::funcs::SetConstant<Context, T> zero;
  zero(dev_ctx, input_grad, static_cast<T>(0.0));

  if (in_h == out_h && in_w == out_w) {
    phi::Copy(dev_ctx, output_grad, dev_ctx.GetPlace(), false, input_grad);
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

  backends::gpu::GpuLaunchConfig config =
      backends::gpu::GetGpuLaunchConfig1D(dev_ctx, pixelNum);

  if ("nearest" == interp_method) {
    if (data_layout == DataLayout::kNCHW) {
      // get launch 3D config
      int nc = n * c;
      backends::gpu::GpuLaunchConfig config_3d =
          backends::gpu::GetGpuLaunchConfig3D(dev_ctx, nc, out_h, out_w);
      KeNearestNeighborInterpNCHWBw<T><<<config_3d.block_per_grid,
                                         config_3d.thread_per_block,
                                         0,
                                         dev_ctx.stream()>>>(input_grad_data,
                                                             in_h,
                                                             in_w,
                                                             output_grad_data,
                                                             out_h,
                                                             out_w,
                                                             nc,
                                                             ratio_h,
                                                             ratio_w,
                                                             align_corners);
    } else {
      int64_t cw = c * out_w;
      auto interp_divmods = funcs::FastDivModForInterpolate(c, out_chw, cw);
      KeNearestNeighborInterpBw<T><<<config.block_per_grid,
                                     config.thread_per_block,
                                     0,
                                     dev_ctx.stream()>>>(input_grad_data,
                                                         in_h,
                                                         in_w,
                                                         n,
                                                         in_chw,
                                                         output_grad_data,
                                                         out_h,
                                                         out_w,
                                                         n,
                                                         out_chw,
                                                         c,
                                                         ratio_h,
                                                         ratio_w,
                                                         align_corners,
                                                         interp_divmods);
    }
  } else if ("bilinear" == interp_method) {
    const float align_type_value =
        (align_mode == 0 && !align_corners) ? 0.5f : 0.f;
    bool is_nchw = (data_layout == DataLayout::kNCHW) ? true : false;
    bool optimize_flag = false;
#ifndef __HIPCC__
    optimize_flag = (in_h < (out_h >> 6) && in_w < (out_w >> 6))
                        ? true
                        : ((in_h == 1 && in_w == 1) ? true : false);
#endif

    if (optimize_flag & is_nchw) {
      KeBilinearInterpBwShareMemory<T><<<config.block_per_grid,
                                         config.thread_per_block,
                                         0,
                                         dev_ctx.stream()>>>(input_grad_data,
                                                             in_h,
                                                             in_w,
                                                             output_grad_data,
                                                             out_h,
                                                             out_w,
                                                             n,
                                                             c,
                                                             ratio_h,
                                                             ratio_w,
                                                             align_type_value,
                                                             is_nchw);
    } else if (!optimize_flag & is_nchw) {
      const int num_kernels = n * c * out_h * out_w;
      const int num_threads = std::min(dev_ctx.GetMaxThreadsPerBlock(), 1024);
      KeBilinearInterpNCHWBw<T>
          <<<backends::gpu::DivUp(num_kernels, num_threads),
             num_threads,
             0,
             dev_ctx.stream()>>>(input_grad_data,
                                 in_h,
                                 in_w,
                                 out_h,
                                 out_w,
                                 n,
                                 c,
                                 ratio_h,
                                 ratio_w,
                                 output_grad_data,
                                 align_type_value);
    } else {
      int64_t cw = c * out_w;
      auto interp_divmods = funcs::FastDivModForInterpolate(c, out_chw, cw);
      KeBilinearInterpBw<T><<<config.block_per_grid,
                              config.thread_per_block,
                              0,
                              dev_ctx.stream()>>>(input_grad_data,
                                                  in_h,
                                                  in_w,
                                                  output_grad_data,
                                                  out_h,
                                                  out_w,
                                                  n,
                                                  out_chw,
                                                  c,
                                                  ratio_h,
                                                  ratio_w,
                                                  align_type_value,
                                                  interp_divmods);
    }
  } else if ("bicubic" == interp_method) {
    constexpr int thread_per_block = 512;
    KeBicubicInterpBw<T>
        <<<config.block_per_grid, thread_per_block, 0, dev_ctx.stream()>>>(
            input_grad_data,
            in_h,
            in_w,
            n,
            in_chw,
            output_grad_data,
            out_h,
            out_w,
            n,
            out_chw,
            c,
            ratio_h,
            ratio_w,
            align_corners,
            data_layout);
  }
}

template <typename T, typename Context>
static void Interpolate3DCUDABwd(
    const Context& dev_ctx,
    const DenseTensor& input,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const DenseTensor& output_grad,
    const std::string& data_layout_str,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* input_grad) {
  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  funcs::ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  float scale_d = -1;
  float scale_h = -1;
  float scale_w = -1;
  if (scale_tensor) {
    auto scale_data =
        funcs::get_new_data_from_tensor<float>(scale_tensor.get_ptr());
    if (scale_data.size() > 2) {
      scale_d = scale_data[0];
      scale_h = scale_data[1];
      scale_w = scale_data[2];
    } else {
      scale_d = scale_data[0];
      scale_h = scale_data[0];
      scale_w = scale_data[0];
    }
    PADDLE_ENFORCE_EQ(
        scale_w > 0,
        true,
        errors::InvalidArgument(
            "The scale_w in input 'Scale' Tensor of Operator(interpolate) "
            "should be greater than 0, but received value is %d.",
            scale_w));
    PADDLE_ENFORCE_EQ(
        scale_h > 0,
        true,
        errors::InvalidArgument(
            "The scale_h in input 'Scale' Tensor of Operator(interpolate) "
            "should be greater than 0, but received value is %d.",
            scale_h));
    PADDLE_ENFORCE_EQ(
        scale_d > 0,
        true,
        errors::InvalidArgument(
            "The scale_d in input 'Scale' Tensor of Operator(interpolate) "
            "should be greater than 0, but received value is %d.",
            scale_d));
  } else {
    if (scale.size() > 2) {
      scale_d = scale[0];
      scale_h = scale[1];
      scale_w = scale[2];

      PADDLE_ENFORCE_EQ(
          scale_w > 0,
          true,
          errors::InvalidArgument(
              "The scale_w in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
      PADDLE_ENFORCE_EQ(
          scale_h > 0,
          true,
          errors::InvalidArgument(
              "The scale_h in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_h));
      PADDLE_ENFORCE_EQ(
          scale_d > 0,
          true,
          errors::InvalidArgument(
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

  if (out_size) {
    DenseTensor sizes;
    phi::Copy(dev_ctx, *out_size, phi::CPUPlace(), true, &sizes);
    auto size_data = sizes.data<int>();
    out_d = size_data[0];
    out_h = size_data[1];
    out_w = size_data[2];
  }
  if (size_tensor && size_tensor->size() > 0) {
    // have size tensor
    auto new_size = funcs::get_new_shape(size_tensor.get());
    out_d = new_size[0];
    out_h = new_size[1];
    out_w = new_size[2];
  }

  auto* output_grad_data = output_grad.data<T>();
  phi::DDim dim_grad;
  if (data_layout == DataLayout::kNCHW) {
    dim_grad = {n, c, in_d, in_h, in_w};
  } else {
    dim_grad = {n, in_d, in_h, in_w, c};
  }
  input_grad->Resize(dim_grad);
  auto* input_grad_data = dev_ctx.template Alloc<T>(input_grad);
  phi::funcs::SetConstant<Context, T> zero;
  zero(dev_ctx, input_grad, static_cast<T>(0.0));

  if (in_d == out_d && in_h == out_h && in_w == out_w) {
    phi::Copy(dev_ctx, output_grad, dev_ctx.GetPlace(), false, input_grad);
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

  backends::gpu::GpuLaunchConfig config =
      backends::gpu::GetGpuLaunchConfig1D(dev_ctx, pixelNum);

  if ("trilinear" == interp_method) {
    KeTrilinearInterpBw<T><<<config.block_per_grid,
                             config.thread_per_block,
                             0,
                             dev_ctx.stream()>>>(input_grad_data,
                                                 in_d,
                                                 in_h,
                                                 in_w,
                                                 n,
                                                 in_cdhw,
                                                 output_grad_data,
                                                 out_d,
                                                 out_h,
                                                 out_w,
                                                 n,
                                                 out_cdhw,
                                                 c,
                                                 ratio_d,
                                                 ratio_h,
                                                 ratio_w,
                                                 align_corners,
                                                 align_mode,
                                                 data_layout);
  } else if ("nearest" == interp_method) {
    KeNearestNeighbor3DInterpBw<T><<<config.block_per_grid,
                                     config.thread_per_block,
                                     0,
                                     dev_ctx.stream()>>>(input_grad_data,
                                                         in_d,
                                                         in_h,
                                                         in_w,
                                                         n,
                                                         in_cdhw,
                                                         output_grad_data,
                                                         out_d,
                                                         out_h,
                                                         out_w,
                                                         n,
                                                         out_cdhw,
                                                         c,
                                                         ratio_d,
                                                         ratio_h,
                                                         ratio_w,
                                                         align_corners,
                                                         data_layout);
  }
}

template <typename T, typename Context>
void InterpolateGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const DenseTensor& output_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* x_grad) {
  auto output_grad_dims = output_grad.dims();
  if (output_grad_dims.size() == 3) {  // 1D interpolation grad
    Interpolate1DCUDABwd<T, Context>(dev_ctx,
                                     x,
                                     out_size,
                                     size_tensor,
                                     scale_tensor,
                                     output_grad,
                                     data_layout,
                                     out_w,
                                     scale,
                                     interp_method,
                                     align_corners,
                                     align_mode,
                                     x_grad);
  } else if (output_grad_dims.size() == 4) {  // 2D interpolation grad
    Interpolate2DCUDABwd<T, Context>(dev_ctx,
                                     x,
                                     out_size,
                                     size_tensor,
                                     scale_tensor,
                                     output_grad,
                                     data_layout,
                                     out_h,
                                     out_w,
                                     scale,
                                     interp_method,
                                     align_corners,
                                     align_mode,
                                     x_grad);

  } else if (output_grad_dims.size() == 5) {  // 3D interpolation grad
    Interpolate3DCUDABwd<T, Context>(dev_ctx,
                                     x,
                                     out_size,
                                     size_tensor,
                                     scale_tensor,
                                     output_grad,
                                     data_layout,
                                     out_d,
                                     out_h,
                                     out_w,
                                     scale,
                                     interp_method,
                                     align_corners,
                                     align_mode,
                                     x_grad);
  }
}

template <typename T, typename Context>
void BilinearInterpGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* x_grad) {
  InterpolateGradKernel<T, Context>(dev_ctx,
                                    x,
                                    out_size,
                                    size_tensor,
                                    scale_tensor,
                                    out_grad,
                                    data_layout,
                                    out_d,
                                    out_h,
                                    out_w,
                                    scale,
                                    interp_method,
                                    align_corners,
                                    align_mode,
                                    x_grad);
}

template <typename T, typename Context>
void LegacyBilinearInterpGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    float scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* x_grad) {
  const auto& dim_x = x.dims();
  std::vector<float> scale_vec;
  if (scale > 0) {
    for (int i = 0; i < dim_x.size() - 2; i++) {
      scale_vec.push_back(scale);
    }
  }
  InterpolateGradKernel<T, Context>(dev_ctx,
                                    x,
                                    out_size,
                                    size_tensor,
                                    scale_tensor,
                                    out_grad,
                                    data_layout,
                                    out_d,
                                    out_h,
                                    out_w,
                                    scale_vec,
                                    interp_method,
                                    align_corners,
                                    align_mode,
                                    x_grad);
}

template <typename T, typename Context>
void NearestInterpGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* x_grad) {
  InterpolateGradKernel<T, Context>(dev_ctx,
                                    x,
                                    out_size,
                                    size_tensor,
                                    scale_tensor,
                                    out_grad,
                                    data_layout,
                                    out_d,
                                    out_h,
                                    out_w,
                                    scale,
                                    interp_method,
                                    align_corners,
                                    align_mode,
                                    x_grad);
}

template <typename T, typename Context>
void LegacyNearestInterpGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    float scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* x_grad) {
  const auto& dim_x = x.dims();
  std::vector<float> scale_vec;
  if (scale > 0) {
    for (int i = 0; i < dim_x.size() - 2; i++) {
      scale_vec.push_back(scale);
    }
  }
  InterpolateGradKernel<T, Context>(dev_ctx,
                                    x,
                                    out_size,
                                    size_tensor,
                                    scale_tensor,
                                    out_grad,
                                    data_layout,
                                    out_d,
                                    out_h,
                                    out_w,
                                    scale_vec,
                                    interp_method,
                                    align_corners,
                                    align_mode,
                                    x_grad);
}

template <typename T, typename Context>
void TrilinearInterpGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* x_grad) {
  InterpolateGradKernel<T, Context>(dev_ctx,
                                    x,
                                    out_size,
                                    size_tensor,
                                    scale_tensor,
                                    out_grad,
                                    data_layout,
                                    out_d,
                                    out_h,
                                    out_w,
                                    scale,
                                    interp_method,
                                    align_corners,
                                    align_mode,
                                    x_grad);
}

template <typename T, typename Context>
void LinearInterpGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* x_grad) {
  InterpolateGradKernel<T, Context>(dev_ctx,
                                    x,
                                    out_size,
                                    size_tensor,
                                    scale_tensor,
                                    out_grad,
                                    data_layout,
                                    out_d,
                                    out_h,
                                    out_w,
                                    scale,
                                    interp_method,
                                    align_corners,
                                    align_mode,
                                    x_grad);
}

template <typename T, typename Context>
void BicubicInterpGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* x_grad) {
  InterpolateGradKernel<T, Context>(dev_ctx,
                                    x,
                                    out_size,
                                    size_tensor,
                                    scale_tensor,
                                    out_grad,
                                    data_layout,
                                    out_d,
                                    out_h,
                                    out_w,
                                    scale,
                                    interp_method,
                                    align_corners,
                                    align_mode,
                                    x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(bilinear_interp_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::BilinearInterpGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(1).SetBackend(phi::Backend::CPU);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(legacy_bilinear_interp_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::LegacyBilinearInterpGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(1).SetBackend(phi::Backend::CPU);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(nearest_interp_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::NearestInterpGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(1).SetBackend(phi::Backend::CPU);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(legacy_nearest_interp_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::LegacyNearestInterpGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(1).SetBackend(phi::Backend::CPU);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(trilinear_interp_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::TrilinearInterpGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(1).SetBackend(phi::Backend::CPU);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(linear_interp_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::LinearInterpGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(1).SetBackend(phi::Backend::CPU);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(bicubic_interp_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::BicubicInterpGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(1).SetBackend(phi::Backend::CPU);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
