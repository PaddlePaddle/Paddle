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

#include "paddle/phi/kernels/interpolate_kernel.h"

#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/interpolate_function.h"
#include "paddle/phi/kernels/primitive/datamover_primitives.h"

namespace phi {
using phi::kps::details::FastDivMod;

template <typename T>
__forceinline__ __device__ void PreCalculatorForLinearInterpInputIndex(
    int* in_img_idx,
    int* x_id,
    T* lambda1,
    T* lambda2,
    T src_x,
    const int in_img_x) {
  src_x = (src_x > static_cast<T>(0)) ? src_x : static_cast<T>(0);
  *in_img_idx = static_cast<int>(src_x);
  *x_id = (*in_img_idx < in_img_x - 1) ? 1 : 0;
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  *lambda1 = static_cast<T>(static_cast<MT>(src_x) - *in_img_idx);
  *lambda2 = static_cast<T>(1.0) - *lambda1;
}

template <typename T>
__global__ void KeLinearInterpFw(const T* in,
                                 const size_t in_img_w,
                                 const size_t input_w,
                                 T* out,
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
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    T src_w = static_cast<T>(ratio_w * (out_img_idx + 0.5) - 0.5);
    src_w = (src_w > static_cast<T>(0)) ? src_w : static_cast<T>(0);
    T w1lambda = align_flag
                     ? static_cast<T>(static_cast<MT>(src_w) - in_img_idx)
                     : static_cast<T>(ratio_w * out_img_idx - in_img_idx);
    T w2lambda = static_cast<T>(1.0) - w1lambda;

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
__global__ void KeNearestNeighborInterpNCHWFw(const T* in,
                                              const size_t in_img_h,
                                              const size_t in_img_w,
                                              T* out,
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
      out[out_index] = in[in_index];
      in_index += in_index_stride;
      out_index += out_index_stride;
      nc_id += nc_stride;
    }
  }
}

template <typename T>
__global__ void KeNearestNeighborInterpFw(
    const T* in,
    const size_t in_img_h,
    const size_t in_img_w,
    const size_t input_h,
    const size_t input_w,
    T* out,
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

    out[tid] = in[out_id_h * input_w + in_img_idy * in_img_w * num_channels +
                  in_img_idx * num_channels + channel_id];
  }
}

template <typename T>
__global__ void KeBilinearInterpFw(const T* in,
                                   const size_t in_img_h,
                                   const size_t in_img_w,
                                   const size_t input_h,
                                   const size_t input_w,
                                   T* out,
                                   const size_t out_img_h,
                                   const size_t out_img_w,
                                   const size_t output_h,
                                   const size_t output_w,
                                   const size_t num_channels,
                                   const float ratio_h,
                                   const float ratio_w,
                                   const float align_type_value,
                                   funcs::FastDivModForInterpolate divmods) {
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
    T src_w = static_cast<T>(ratio_w * (out_img_idx + align_type_value) -
                             align_type_value);
    T src_h = static_cast<T>(ratio_h * (out_img_idy + align_type_value) -
                             align_type_value);

    PreCalculatorForLinearInterpInputIndex(
        &in_img_idx, &w_id, &w1lambda, &w2lambda, src_w, in_img_w);
    PreCalculatorForLinearInterpInputIndex(
        &in_img_idy, &h_id, &h1lambda, &h2lambda, src_h, in_img_h);

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

template <typename T>
__global__ void KeBilinearInterpNCHWFw(const T* in,
                                       const size_t in_img_h,
                                       const size_t in_img_w,
                                       T* out,
                                       const size_t out_img_h,
                                       const size_t out_img_w,
                                       const size_t nc,
                                       const float ratio_h,
                                       const float ratio_w,
                                       const float align_type_value) {
  int out_img_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int out_img_idy = threadIdx.y + blockIdx.y * blockDim.y;
  int nc_id = threadIdx.z + blockIdx.z * blockDim.z;
  int nc_stride = blockDim.z * gridDim.z;

  int in_img_idx, in_img_idy, h_id, w_id;
  T h1lambda, w1lambda, h2lambda, w2lambda;
  T src_w = static_cast<T>(ratio_w * (out_img_idx + align_type_value) -
                           align_type_value);
  T src_h = static_cast<T>(ratio_h * (out_img_idy + align_type_value) -
                           align_type_value);

  PreCalculatorForLinearInterpInputIndex(
      &in_img_idx, &w_id, &w1lambda, &w2lambda, src_w, in_img_w);
  PreCalculatorForLinearInterpInputIndex(
      &in_img_idy, &h_id, &h1lambda, &h2lambda, src_h, in_img_h);

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
__device__ __forceinline__ static T Kecubic_interp(
    const T x0, const T x1, const T x2, const T x3, T t) {
  T coeffs[4];
  T a = static_cast<T>(-0.75);
  T x_1 = t;
  T x_2 = static_cast<T>(1.0) - t;
  coeffs[0] = funcs::CubicConvolution2<T>(x_1 + static_cast<T>(1.0), a);
  coeffs[1] = funcs::CubicConvolution1<T>(x_1, a);
  coeffs[2] = funcs::CubicConvolution1<T>(x_2, a);
  coeffs[3] = funcs::CubicConvolution2<T>(x_2 + static_cast<T>(1.0), a);
  return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

template <typename T>
__global__ void KeBicubicInterpFw(const T* in,
                                  const size_t in_img_h,
                                  const size_t in_img_w,
                                  const size_t input_h,
                                  const size_t input_w,
                                  T* out,
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
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    const T y_t = static_cast<T>(static_cast<MT>(in_img_idy) - input_y);

    T in_img_idx = align_corners
                       ? static_cast<T>(ratio_w * out_img_idx)
                       : static_cast<T>(ratio_w * (out_img_idx + 0.5) - 0.5);
    int input_x = floorf(in_img_idx);
    const T x_t = static_cast<T>(static_cast<MT>(in_img_idx) - input_x);

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

        coefficients[k] = Kecubic_interp<T>(
            in_pos_0[0], in_pos_1[0], in_pos_2[0], in_pos_3[0], x_t);
      }

      out[out_id_h * output_w + out_id_w] = Kecubic_interp<T>(coefficients[0],
                                                              coefficients[1],
                                                              coefficients[2],
                                                              coefficients[3],
                                                              y_t);

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

        coefficients[k] = Kecubic_interp<T>(
            in_pos_0[0], in_pos_1[0], in_pos_2[0], in_pos_3[0], x_t);
      }

      out[out_id_h * output_w + out_id_w] = Kecubic_interp<T>(coefficients[0],
                                                              coefficients[1],
                                                              coefficients[2],
                                                              coefficients[3],
                                                              y_t);
    }
  }
}

template <typename T>
__global__ void KeTrilinearInterpFw(const T* in,
                                    const size_t in_img_d,
                                    const size_t in_img_h,
                                    const size_t in_img_w,
                                    const size_t input_h,
                                    const size_t input_w,
                                    T* out,
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
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    T src_d = static_cast<T>(ratio_d * (out_img_idt + 0.5) - 0.5);
    src_d = (src_d > static_cast<T>(0)) ? src_d : static_cast<T>(0);
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
__global__ void KeNearestNeighbor3DInterpFw(const T* in,
                                            const size_t in_img_d,
                                            const size_t in_img_h,
                                            const size_t in_img_w,
                                            const size_t input_h,
                                            const size_t input_w,
                                            T* out,
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

template <typename T, typename Context>
static void Interpolate1DCUDAFwd(
    const Context& dev_ctx,
    const DenseTensor& input,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout_str,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  auto* input_data = input.data<T>();

  const DataLayout data_layout = phi::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  funcs::ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  float scale_w = -1;
  if (size_tensor && size_tensor->size() > 0) {
    // have size tensor
    auto new_size = funcs::get_new_shape(size_tensor.get());
    out_w = new_size[0];
  } else {
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
      paddle::framework::TensorCopySync(
          *out_size, paddle::platform::CPUPlace(), &sizes);
      auto size_data = sizes.data<int>();
      out_w = size_data[0];
    }
  }
  PADDLE_ENFORCE_GT(
      out_w,
      0,
      errors::InvalidArgument("out_w in Attr(out_shape) of Op(interpolate) "
                              "should be greater than 0."));
  phi::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {n, c, out_w};
  } else {
    dim_out = {n, out_w, c};
  }
  output->Resize(dim_out);
  auto output_data = dev_ctx.template Alloc<T>(output);

  if (in_w == out_w) {
    paddle::framework::TensorCopy(input, dev_ctx.GetPlace(), output);
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

  backends::gpu::GpuLaunchConfig config =
      backends::gpu::GetGpuLaunchConfig1D(dev_ctx, pixelNum);

  if ("linear" == interp_method) {
    KeLinearInterpFw<T><<<config.block_per_grid,
                          config.thread_per_block,
                          0,
                          dev_ctx.stream()>>>(input_data,
                                              in_w,
                                              in_cw,
                                              output_data,
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
static void Interpolate2DCUDAFwd(
    const Context& dev_ctx,
    const DenseTensor& input,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout_str,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  auto* input_data = input.data<T>();

  const DataLayout data_layout = phi::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  funcs::ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  float scale_w = -1;
  float scale_h = -1;
  if (size_tensor && size_tensor->size() > 0) {
    // have size tensor
    auto new_size = funcs::get_new_shape(size_tensor.get());
    out_h = new_size[0];
    out_w = new_size[1];
  } else {
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
      paddle::framework::TensorCopySync(
          *out_size, paddle::platform::CPUPlace(), &sizes);
      auto size_data = sizes.data<int>();
      out_h = size_data[0];
      out_w = size_data[1];
    }
  }
  PADDLE_ENFORCE_GT(
      out_h,
      0,
      errors::InvalidArgument("out_h in Attr(out_shape) of Op(interpolate) "
                              "should be greater than 0."));
  PADDLE_ENFORCE_GT(
      out_w,
      0,
      errors::InvalidArgument("out_w in Attr(out_shape) of Op(interpolate) "
                              "should be greater than 0."));

  phi::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {n, c, out_h, out_w};
  } else {
    dim_out = {n, out_h, out_w, c};
  }
  output->Resize(dim_out);
  auto output_data = dev_ctx.template Alloc<T>(output);

  if (in_h == out_h && in_w == out_w) {
    paddle::framework::TensorCopy(input, dev_ctx.GetPlace(), output);
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
      KeNearestNeighborInterpNCHWFw<T><<<config_3d.block_per_grid,
                                         config_3d.thread_per_block,
                                         0,
                                         dev_ctx.stream()>>>(input_data,
                                                             in_h,
                                                             in_w,
                                                             output_data,
                                                             out_h,
                                                             out_w,
                                                             nc,
                                                             ratio_h,
                                                             ratio_w,
                                                             align_corners);
    } else {
      int64_t cw = c * out_w;
      auto interp_divmods = funcs::FastDivModForInterpolate(c, out_chw, cw);
      KeNearestNeighborInterpFw<T><<<config.block_per_grid,
                                     config.thread_per_block,
                                     0,
                                     dev_ctx.stream()>>>(input_data,
                                                         in_h,
                                                         in_w,
                                                         n,
                                                         in_chw,
                                                         output_data,
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
    dim3 thread_num = config.thread_per_block;
#ifdef WITH_NV_JETSON
    if (config.compute_capability == 53 || config.compute_capability == 62) {
      thread_num = 512;
    }
#endif
    const float align_type_value =
        (align_mode == 0 && !align_corners) ? 0.5f : 0.f;
    if (data_layout == DataLayout::kNCHW) {
      // get launch 3D config
      int nc = n * c;
      backends::gpu::GpuLaunchConfig config_3d =
          backends::gpu::GetGpuLaunchConfig3D(dev_ctx, nc, out_h, out_w);
      KeBilinearInterpNCHWFw<T><<<config_3d.block_per_grid,
                                  config_3d.thread_per_block,
                                  0,
                                  dev_ctx.stream()>>>(input_data,
                                                      in_h,
                                                      in_w,
                                                      output_data,
                                                      out_h,
                                                      out_w,
                                                      nc,
                                                      ratio_h,
                                                      ratio_w,
                                                      align_type_value);
    } else {
      int64_t cw = c * out_w;
      auto interp_divmods = funcs::FastDivModForInterpolate(c, out_chw, cw);
      KeBilinearInterpFw<T>
          <<<config.block_per_grid, thread_num, 0, dev_ctx.stream()>>>(
              input_data,
              in_h,
              in_w,
              n,
              in_chw,
              output_data,
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
#ifdef __HIPCC__
    constexpr int thread_per_block = 256;
#else
    constexpr int thread_per_block = 512;
#endif
    KeBicubicInterpFw<T>
        <<<config.block_per_grid, thread_per_block, 0, dev_ctx.stream()>>>(
            input_data,
            in_h,
            in_w,
            n,
            in_chw,
            output_data,
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
static void Interpolate3DCUDAFwd(
    const Context& dev_ctx,
    const DenseTensor& input,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout_str,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  auto* input_data = input.data<T>();

  const DataLayout data_layout = phi::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  funcs::ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  float scale_w = -1;
  float scale_d = -1;
  float scale_h = -1;
  if (size_tensor && size_tensor->size() > 0) {
    // have size tensor
    auto new_size = funcs::get_new_shape(size_tensor.get());
    out_d = new_size[0];
    out_h = new_size[1];
    out_w = new_size[2];
  } else {
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
      paddle::framework::TensorCopySync(
          *out_size, paddle::platform::CPUPlace(), &sizes);
      auto size_data = sizes.data<int>();
      out_d = size_data[0];
      out_h = size_data[1];
      out_w = size_data[2];
    }
  }
  PADDLE_ENFORCE_GT(
      out_d,
      0,
      errors::InvalidArgument("out_d in Attr(out_shape) of Op(interpolate) "
                              "should be greater than 0."));
  PADDLE_ENFORCE_GT(
      out_h,
      0,
      errors::InvalidArgument("out_h in Attr(out_shape) of Op(interpolate) "
                              "should be greater than 0."));
  PADDLE_ENFORCE_GT(
      out_w,
      0,
      errors::InvalidArgument("out_w in Attr(out_shape) of Op(interpolate) "
                              "should be greater than 0."));

  phi::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {n, c, out_d, out_h, out_w};
  } else {
    dim_out = {n, out_d, out_h, out_w, c};
  }
  output->Resize(dim_out);
  auto output_data = dev_ctx.template Alloc<T>(output);

  if (in_d == out_d && in_h == out_h && in_w == out_w) {
    paddle::framework::TensorCopy(input, dev_ctx.GetPlace(), output);
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
    KeTrilinearInterpFw<T><<<config.block_per_grid,
                             config.thread_per_block,
                             0,
                             dev_ctx.stream()>>>(input_data,
                                                 in_d,
                                                 in_h,
                                                 in_w,
                                                 n,
                                                 in_cdhw,
                                                 output_data,
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
    KeNearestNeighbor3DInterpFw<T><<<config.block_per_grid,
                                     config.thread_per_block,
                                     0,
                                     dev_ctx.stream()>>>(input_data,
                                                         in_d,
                                                         in_h,
                                                         in_w,
                                                         n,
                                                         in_cdhw,
                                                         output_data,
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
void InterpolateKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  auto input_dims = x.dims();
  if (input_dims.size() == 3) {  // 1D interpolation
    Interpolate1DCUDAFwd<T, Context>(dev_ctx,
                                     x,
                                     out_size,
                                     size_tensor,
                                     scale_tensor,
                                     data_layout,
                                     out_w,
                                     scale,
                                     interp_method,
                                     align_corners,
                                     align_mode,
                                     output);
  } else if (input_dims.size() == 4) {  // 2D interpolation
    Interpolate2DCUDAFwd<T, Context>(dev_ctx,
                                     x,
                                     out_size,
                                     size_tensor,
                                     scale_tensor,
                                     data_layout,
                                     out_h,
                                     out_w,
                                     scale,
                                     interp_method,
                                     align_corners,
                                     align_mode,
                                     output);
  } else if (input_dims.size() == 5) {  // 3D interpolation
    Interpolate3DCUDAFwd<T, Context>(dev_ctx,
                                     x,
                                     out_size,
                                     size_tensor,
                                     scale_tensor,
                                     data_layout,
                                     out_d,
                                     out_h,
                                     out_w,
                                     scale,
                                     interp_method,
                                     align_corners,
                                     align_mode,
                                     output);
  }
}

template <typename T, typename Context>
void BilinearInterpKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  InterpolateKernel<T, Context>(dev_ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale,
                                interp_method,
                                align_corners,
                                align_mode,
                                output);
}

template <typename T, typename Context>
void NearestInterpKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  InterpolateKernel<T, Context>(dev_ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale,
                                interp_method,
                                align_corners,
                                align_mode,
                                output);
}

template <typename T, typename Context>
void TrilinearInterpKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  InterpolateKernel<T, Context>(dev_ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale,
                                interp_method,
                                align_corners,
                                align_mode,
                                output);
}

template <typename T, typename Context>
void LinearInterpKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  InterpolateKernel<T, Context>(dev_ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale,
                                interp_method,
                                align_corners,
                                align_mode,
                                output);
}

template <typename T, typename Context>
void BicubicInterpKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  InterpolateKernel<T, Context>(dev_ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale,
                                interp_method,
                                align_corners,
                                align_mode,
                                output);
}

}  // namespace phi

PD_REGISTER_KERNEL(bilinear_interp,
                   GPU,
                   ALL_LAYOUT,
                   phi::BilinearInterpKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   int) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(nearest_interp,
                   GPU,
                   ALL_LAYOUT,
                   phi::NearestInterpKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(trilinear_interp,
                   GPU,
                   ALL_LAYOUT,
                   phi::TrilinearInterpKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   int) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(linear_interp,
                   GPU,
                   ALL_LAYOUT,
                   phi::LinearInterpKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   int) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(bicubic_interp,
                   GPU,
                   ALL_LAYOUT,
                   phi::BicubicInterpKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   int) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
