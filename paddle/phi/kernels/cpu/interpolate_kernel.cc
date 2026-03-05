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
#include <array>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "paddle/common/layout.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/interpolate_function.h"

namespace phi {

template <typename T>
static inline T cubic_interp(T x0, T x1, T x2, T x3, T t) {
  std::array<T, 4> coeffs;
  funcs::GetCubicUpsampleCoefficients<T>(coeffs.data(), t);

  return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

template <typename T>
static void LinearInterpolation(const DenseTensor& input,
                                DenseTensor* output,
                                const float ratio_w,
                                const int in_w,
                                const int n,
                                const int c,
                                const int out_w,
                                const bool align_corners,
                                const int align_mode,
                                const DataLayout data_layout) {
  auto input_t = EigenTensor<T, 3>::From(input);
  auto output_t = EigenTensor<T, 3>::From(*output);
  bool align_flag = (align_mode == 0 && !align_corners);
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

  std::vector<int> vx_w, vx_e;
  std::vector<MT> vd_w, vd_e;
  vx_w.reserve(out_w);
  vx_e.reserve(out_w);
  vd_w.reserve(out_w);
  vd_e.reserve(out_w);
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int l = 0; l < out_w; l++) {
    int x_w = static_cast<int>(align_flag ? (ratio_w * (l + 0.5) - 0.5)
                                          : ratio_w * l);
    x_w = (x_w > 0) ? x_w : 0;                       // w
    int x_e = (x_w < (in_w - 1)) ? (x_w + 1) : x_w;  // w_id

    MT idx_src_x = ratio_w * (l + 0.5) - 0.5;
    idx_src_x = (idx_src_x > 0) ? idx_src_x : 0;
    MT d_w = align_flag ? idx_src_x - x_w : ratio_w * l - x_w;  // w1lambda
    MT d_e = 1. - d_w;                                          // w2lambda
    {
      vx_w[l] = x_w;
      vx_e[l] = x_e;
      vd_w[l] = d_w;
      vd_e[l] = d_e;
    }
  }

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(3)
#endif
  for (int i = 0; i < n; i++) {    // loop for batches
    for (int j = 0; j < c; j++) {  // loop for channels
      for (int l = 0; l < out_w; l++) {
        // linear interpolation
        T out_t;
        if (data_layout == DataLayout::NCHW) {
          out_t =
              static_cast<T>(static_cast<MT>(input_t(i, j, vx_w[l])) * vd_e[l] +
                             static_cast<MT>(input_t(i, j, vx_e[l])) * vd_w[l]);
          output_t(i, j, l) = out_t;
        } else {
          out_t =
              static_cast<T>(static_cast<MT>(input_t(i, vx_w[l], j)) * vd_e[l] +
                             static_cast<MT>(input_t(i, vx_e[l], j)) * vd_w[l]);
          output_t(i, l, j) = out_t;
        }
      }
    }
  }
}

template <typename T>
static void BilinearInterpolation(const DenseTensor& input,
                                  DenseTensor* output,
                                  const float ratio_h,
                                  const float ratio_w,
                                  const int in_h,
                                  const int in_w,
                                  const int n,
                                  const int c,
                                  const int out_h,
                                  const int out_w,
                                  const bool align_corners,
                                  const int align_mode,
                                  const DataLayout data_layout) {
  auto input_t = EigenTensor<T, 4>::From(input);
  auto output_t = EigenTensor<T, 4>::From(*output);
  bool align_flag = (align_mode == 0 && !align_corners);
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

  std::vector<int> vy_n, vy_s;
  std::vector<float> vd_n, vd_s;
  vy_n.reserve(out_h);
  vy_s.reserve(out_h);
  vd_n.reserve(out_h);
  vd_s.reserve(out_h);
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int k = 0; k < out_h; k++) {
    int y_n = static_cast<int>(align_flag ? (ratio_h * (k + 0.5) - 0.5)
                                          : (ratio_h * static_cast<float>(k)));
    y_n = (y_n > 0) ? y_n : 0;
    int y_s = (y_n + 1) < (in_h - 1) ? (y_n + 1) : (in_h - 1);
    float idx_src_y = ratio_h * (k + 0.5) - 0.5;
    idx_src_y = (idx_src_y > 0) ? idx_src_y : 0;
    float d_n = align_flag
                    ? idx_src_y - static_cast<float>(y_n)
                    : ratio_h * static_cast<float>(k) - static_cast<float>(y_n);
    float d_s = 1.f - d_n;
    {
      vy_n[k] = y_n;
      vy_s[k] = y_s;
      vd_n[k] = d_n;
      vd_s[k] = d_s;
    }
  }

  std::vector<int> vx_w, vx_e;
  std::vector<float> vd_w, vd_e;
  vx_w.reserve(out_w);
  vx_e.reserve(out_w);
  vd_w.reserve(out_w);
  vd_e.reserve(out_w);
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int l = 0; l < out_w; l++) {
    int x_w = (align_mode == 0 && !align_corners)
                  ? static_cast<int>(ratio_w * (l + 0.5) - 0.5)
                  : static_cast<int>(ratio_w * static_cast<float>(l));
    x_w = (x_w > 0) ? x_w : 0;
    int x_e = (x_w + 1) < (in_w - 1) ? (x_w + 1) : (in_w - 1);
    float idx_src_x = ratio_w * (static_cast<float>(l) + 0.5f) - 0.5f;
    idx_src_x = (idx_src_x > 0) ? idx_src_x : 0;
    float d_w = align_flag
                    ? idx_src_x - static_cast<float>(x_w)
                    : ratio_w * static_cast<float>(l) - static_cast<float>(x_w);
    float d_e = 1.f - d_w;
    {
      vx_w[l] = x_w;
      vx_e[l] = x_e;
      vd_w[l] = d_w;
      vd_e[l] = d_e;
    }
  }

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(4)
#endif
  for (int i = 0; i < n; i++) {          // loop for batches
    for (int j = 0; j < c; j++) {        // loop for channels
      for (int k = 0; k < out_h; k++) {  // loop for images
        for (int l = 0; l < out_w; l++) {
          // bilinear interpolation
          T out_t;
          if (data_layout == DataLayout::NCHW) {
            out_t = static_cast<T>(
                static_cast<MT>(input_t(i, j, vy_n[k], vx_w[l])) * vd_s[k] *
                    vd_e[l] +
                static_cast<MT>(input_t(i, j, vy_s[k], vx_w[l])) * vd_n[k] *
                    vd_e[l] +
                static_cast<MT>(input_t(i, j, vy_n[k], vx_e[l])) * vd_s[k] *
                    vd_w[l] +
                static_cast<MT>(input_t(i, j, vy_s[k], vx_e[l])) * vd_n[k] *
                    vd_w[l]);
            output_t(i, j, k, l) = out_t;

          } else {
            out_t = static_cast<T>(
                static_cast<MT>(input_t(i, vy_n[k], vx_w[l], j)) * vd_s[k] *
                    vd_e[l] +
                static_cast<MT>(input_t(i, vy_s[k], vx_w[l], j)) * vd_n[k] *
                    vd_e[l] +
                static_cast<MT>(input_t(i, vy_n[k], vx_e[l], j)) * vd_s[k] *
                    vd_w[l] +
                static_cast<MT>(input_t(i, vy_s[k], vx_e[l], j)) * vd_n[k] *
                    vd_w[l]);
            output_t(i, k, l, j) = out_t;
          }
        }
      }
    }
  }
}

template <typename T>
static void NearestNeighborInterpolate(const DenseTensor& input,
                                       DenseTensor* output,
                                       const float ratio_h,
                                       const float ratio_w,
                                       const int n,
                                       const int c,
                                       const int out_h,
                                       const int out_w,
                                       const bool align_corners,
                                       const DataLayout& data_layout) {
  auto input_t = EigenTensor<T, 4>::From(input);
  auto output_t = EigenTensor<T, 4>::From(*output);

  for (int k = 0; k < out_h; k++) {  // loop for images
    int in_k =
        (align_corners)
            ? static_cast<int>(std::lround(ratio_h * static_cast<float>(k)))
            : static_cast<int>(ratio_h * static_cast<float>(k));

    for (int l = 0; l < out_w; l++) {
      int in_l =
          (align_corners)
              ? static_cast<int>(std::lround(ratio_w * static_cast<float>(l)))
              : static_cast<int>(ratio_w * static_cast<float>(l));

      for (int i = 0; i < n; i++) {    // loop for batches
        for (int j = 0; j < c; j++) {  // loop for channels
          if (data_layout == DataLayout::NCHW) {
            output_t(i, j, k, l) = input_t(i, j, in_k, in_l);
          } else {
            output_t(i, k, l, j) = input_t(i, in_k, in_l, j);
          }
        }
      }
    }
  }
}

template <typename T>
static void BicubicInterpolation(const DenseTensor& input,
                                 DenseTensor* output,
                                 const float ratio_h,
                                 const float ratio_w,
                                 const int in_h,
                                 const int in_w,
                                 const int n,
                                 const int c,
                                 const int out_h,
                                 const int out_w,
                                 const bool align_corners,
                                 const DataLayout data_layout) {
  auto input_t = EigenTensor<T, 4>::From(input);
  auto output_t = EigenTensor<T, 4>::From(*output);
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

  for (int k = 0; k < out_h; k++) {  // loop for images
    MT y_n = align_corners ? static_cast<MT>(ratio_h * static_cast<float>(k))
                           : static_cast<MT>(ratio_h * (k + 0.5) - 0.5);
    int input_y = floorf(y_n);
    const MT y_t = y_n - input_y;

    for (int l = 0; l < out_w; l++) {
      MT x_n = align_corners ? static_cast<MT>(ratio_w * static_cast<float>(l))
                             : static_cast<MT>(ratio_w * (l + 0.5) - 0.5);
      int input_x = floorf(x_n);
      const MT x_t = x_n - input_x;

      for (int i = 0; i < n; i++) {    // loop for batches
        for (int j = 0; j < c; j++) {  // loop for channels
          std::array<MT, 4> coefficients;
          // interp 4 times in x direction
          for (int ii = 0; ii < 4; ii++) {
            int access_y = std::max(std::min(input_y - 1 + ii, in_h - 1),
                                    static_cast<int>(0));
            int access_x_0 =
                std::max(std::min(input_x - 1, in_w - 1), static_cast<int>(0));
            int access_x_1 =
                std::max(std::min(input_x + 0, in_w - 1), static_cast<int>(0));
            int access_x_2 =
                std::max(std::min(input_x + 1, in_w - 1), static_cast<int>(0));
            int access_x_3 =
                std::max(std::min(input_x + 2, in_w - 1), static_cast<int>(0));
            if (data_layout == DataLayout::NCHW) {
              coefficients[ii] = cubic_interp<MT>(
                  static_cast<MT>(input_t(i, j, access_y, access_x_0)),
                  static_cast<MT>(input_t(i, j, access_y, access_x_1)),
                  static_cast<MT>(input_t(i, j, access_y, access_x_2)),
                  static_cast<MT>(input_t(i, j, access_y, access_x_3)),
                  x_t);
            } else {
              coefficients[ii] = cubic_interp<MT>(
                  static_cast<MT>(input_t(i, access_y, access_x_0, j)),
                  static_cast<MT>(input_t(i, access_y, access_x_1, j)),
                  static_cast<MT>(input_t(i, access_y, access_x_2, j)),
                  static_cast<MT>(input_t(i, access_y, access_x_3, j)),
                  x_t);
            }
          }

          // interp y direction
          if (data_layout == DataLayout::NCHW) {
            output_t(i, j, k, l) =
                static_cast<T>(cubic_interp<MT>(coefficients[0],
                                                coefficients[1],
                                                coefficients[2],
                                                coefficients[3],
                                                y_t));
          } else {
            output_t(i, k, l, j) =
                static_cast<T>(cubic_interp<MT>(coefficients[0],
                                                coefficients[1],
                                                coefficients[2],
                                                coefficients[3],
                                                y_t));
          }
        }
      }
    }
  }
}

template <typename T>
static void TrilinearInterpolation(const DenseTensor& input,
                                   DenseTensor* output,
                                   const float ratio_d,
                                   const float ratio_h,
                                   const float ratio_w,
                                   const int in_d,
                                   const int in_h,
                                   const int in_w,
                                   const int n,
                                   const int c,
                                   const int out_d,
                                   const int out_h,
                                   const int out_w,
                                   const bool align_corners,
                                   const int align_mode,
                                   const DataLayout& data_layout) {
  auto input_t = EigenTensor<T, 5>::From(input);
  auto output_t = EigenTensor<T, 5>::From(*output);
  bool align_flag = (align_mode == 0 && !align_corners);
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

  std::vector<int> vt_f, vt_b;
  std::vector<float> vd_f, vd_b;
  vt_f.reserve(out_d);
  vt_b.reserve(out_d);
  vd_f.reserve(out_d);
  vd_b.reserve(out_d);
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int j = 0; j < out_d; j++) {
    int t_f = align_flag ? static_cast<int>(ratio_d * (j + 0.5) - 0.5)
                         : static_cast<int>(ratio_d * static_cast<float>(j));
    t_f = (t_f > 0) ? t_f : 0;
    int t_b = (t_f + 1) < (in_d - 1) ? (t_f + 1) : (in_d - 1);
    float idx_src_t = ratio_d * (static_cast<float>(j) + 0.5f) - 0.5f;
    idx_src_t = (idx_src_t > 0) ? idx_src_t : 0;
    float d_f = align_flag
                    ? idx_src_t - static_cast<float>(t_f)
                    : ratio_d * static_cast<float>(j) - static_cast<float>(t_f);
    float d_b = 1.f - d_f;
    {
      vt_f[j] = t_f;
      vt_b[j] = t_b;
      vd_f[j] = d_f;
      vd_b[j] = d_b;
    }
  }

  std::vector<int> vy_n, vy_s;
  std::vector<float> vd_n, vd_s;
  vy_n.reserve(out_h);
  vy_s.reserve(out_h);
  vd_n.reserve(out_h);
  vd_s.reserve(out_h);
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int k = 0; k < out_h; k++) {
    int y_n = align_flag ? static_cast<int>(ratio_h * (k + 0.5) - 0.5)
                         : static_cast<int>(ratio_h * static_cast<float>(k));
    y_n = (y_n > 0) ? y_n : 0;
    int y_s = (y_n + 1) < (in_h - 1) ? (y_n + 1) : (in_h - 1);
    float idx_src_y = ratio_h * (static_cast<float>(k) + 0.5f) - 0.5f;
    idx_src_y = (idx_src_y > 0) ? idx_src_y : 0;
    float d_n = align_flag
                    ? idx_src_y - static_cast<float>(y_n)
                    : ratio_h * static_cast<float>(k) - static_cast<float>(y_n);
    float d_s = 1.f - d_n;
    {
      vy_n[k] = y_n;
      vy_s[k] = y_s;
      vd_n[k] = d_n;
      vd_s[k] = d_s;
    }
  }

  std::vector<int> vx_w, vx_e;
  std::vector<float> vd_w, vd_e;
  vx_w.reserve(out_w);
  vx_e.reserve(out_w);
  vd_w.reserve(out_w);
  vd_e.reserve(out_w);
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int l = 0; l < out_w; l++) {
    int x_w = (align_mode == 0 && !align_corners)
                  ? static_cast<int>(ratio_w * (l + 0.5) - 0.5)
                  : static_cast<int>(ratio_w * static_cast<float>(l));
    x_w = (x_w > 0) ? x_w : 0;
    int x_e = (x_w + 1) < (in_w - 1) ? (x_w + 1) : (in_w - 1);
    float idx_src_x = ratio_w * (static_cast<float>(l) + 0.5f) - 0.5f;
    idx_src_x = (idx_src_x > 0) ? idx_src_x : 0;
    float d_w = align_flag
                    ? idx_src_x - static_cast<float>(x_w)
                    : ratio_w * static_cast<float>(l) - static_cast<float>(x_w);
    float d_e = 1.f - d_w;
    {
      vx_w[l] = x_w;
      vx_e[l] = x_e;
      vd_w[l] = d_w;
      vd_e[l] = d_e;
    }
  }

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(5)
#endif
  for (int b = 0; b < n; b++) {          // loop for batches
    for (int i = 0; i < c; i++) {        // loop for channels
      for (int j = 0; j < out_d; j++) {  // loop for D, H, W
        for (int k = 0; k < out_h; k++) {
          for (int l = 0; l < out_w; l++) {
            // trilinear interpolation
            if (data_layout == DataLayout::NCHW) {
              T out_t = static_cast<T>(
                  static_cast<MT>(input_t(b, i, vt_f[j], vy_n[k], vx_w[l])) *
                      vd_b[j] * vd_s[k] * vd_e[l] +
                  static_cast<MT>(input_t(b, i, vt_f[j], vy_n[k], vx_e[l])) *
                      vd_b[j] * vd_s[k] * vd_w[l] +
                  static_cast<MT>(input_t(b, i, vt_f[j], vy_s[k], vx_w[l])) *
                      vd_b[j] * vd_n[k] * vd_e[l] +
                  static_cast<MT>(input_t(b, i, vt_f[j], vy_s[k], vx_e[l])) *
                      vd_b[j] * vd_n[k] * vd_w[l] +
                  static_cast<MT>(input_t(b, i, vt_b[j], vy_n[k], vx_w[l])) *
                      vd_f[j] * vd_s[k] * vd_e[l] +
                  static_cast<MT>(input_t(b, i, vt_b[j], vy_n[k], vx_e[l])) *
                      vd_f[j] * vd_s[k] * vd_w[l] +
                  static_cast<MT>(input_t(b, i, vt_b[j], vy_s[k], vx_w[l])) *
                      vd_f[j] * vd_n[k] * vd_e[l] +
                  static_cast<MT>(input_t(b, i, vt_b[j], vy_s[k], vx_e[l])) *
                      vd_f[j] * vd_n[k] * vd_w[l]);
              output_t(b, i, j, k, l) = out_t;
            } else {
              T out_t = static_cast<T>(
                  static_cast<MT>(input_t(b, vt_f[j], vy_n[k], vx_w[l], i)) *
                      vd_b[j] * vd_s[k] * vd_e[l] +
                  static_cast<MT>(input_t(b, vt_f[j], vy_n[k], vx_e[l], i)) *
                      vd_b[j] * vd_s[k] * vd_w[l] +
                  static_cast<MT>(input_t(b, vt_f[j], vy_s[k], vx_w[l], i)) *
                      vd_b[j] * vd_n[k] * vd_e[l] +
                  static_cast<MT>(input_t(b, vt_f[j], vy_s[k], vx_e[l], i)) *
                      vd_b[j] * vd_n[k] * vd_w[l] +
                  static_cast<MT>(input_t(b, vt_b[j], vy_n[k], vx_w[l], i)) *
                      vd_f[j] * vd_s[k] * vd_e[l] +
                  static_cast<MT>(input_t(b, vt_b[j], vy_n[k], vx_e[l], i)) *
                      vd_f[j] * vd_s[k] * vd_w[l] +
                  static_cast<MT>(input_t(b, vt_b[j], vy_s[k], vx_w[l], i)) *
                      vd_f[j] * vd_n[k] * vd_e[l] +
                  static_cast<MT>(input_t(b, vt_b[j], vy_s[k], vx_e[l], i)) *
                      vd_f[j] * vd_n[k] * vd_w[l]);
              output_t(b, j, k, l, i) = out_t;
            }
          }
        }
      }
    }
  }
}

template <typename T>
static void NearestNeighbor3DInterpolate(const DenseTensor& input,
                                         DenseTensor* output,
                                         const float ratio_d,
                                         const float ratio_h,
                                         const float ratio_w,
                                         const int n,
                                         const int c,
                                         const int out_d,
                                         const int out_h,
                                         const int out_w,
                                         const bool align_corners,
                                         const DataLayout& data_layout) {
  auto input_t = EigenTensor<T, 5>::From(input);
  auto output_t = EigenTensor<T, 5>::From(*output);
  for (int d = 0; d < out_d; d++) {  // loop for images
    int in_d =
        (align_corners)
            ? static_cast<int>(std::lround(ratio_d * static_cast<float>(d)))
            : static_cast<int>(ratio_d * static_cast<float>(d));
    for (int k = 0; k < out_h; k++) {
      int in_k =
          (align_corners)
              ? static_cast<int>(std::lround(ratio_h * static_cast<float>(k)))
              : static_cast<int>(ratio_h * static_cast<float>(k));

      for (int l = 0; l < out_w; l++) {
        int in_l =
            (align_corners)
                ? static_cast<int>(std::lround(ratio_w * static_cast<float>(l)))
                : static_cast<int>(ratio_w * static_cast<float>(l));

        for (int i = 0; i < n; i++) {    // loop for batches
          for (int j = 0; j < c; j++) {  // loop for channels
            if (data_layout == DataLayout::NCHW) {
              output_t(i, j, d, k, l) = input_t(i, j, in_d, in_k, in_l);
            } else {  // NDHWC
              output_t(i, d, k, l, j) = input_t(i, in_d, in_k, in_l, j);
            }
          }
        }
      }
    }
  }
}

template <typename T, typename Context>
static void Interpolate1DCPUFwd(
    const Context& dev_ctx,
    const DenseTensor& x,
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const std::string& data_layout_str,
    int out_w,
    const std::vector<double>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  const DataLayout data_layout = StringToDataLayout(data_layout_str);
  int64_t n = 0, c = 0, in_d = 0, in_h = 0, in_w = 0;
  funcs::ExtractNCDWH(x.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  double scale_w = -1.;
  if (size_tensor && !size_tensor->empty()) {
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
      if (!scale.empty()) {
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
      out_w = static_cast<int>(in_w * scale_w);  // NOLINT
    }
    if (out_size) {
      auto out_size_data =
          funcs::get_new_data_from_tensor<int>(out_size.get_ptr());
      out_w = out_size_data[0];
    }
  }
  PADDLE_ENFORCE_GT(
      out_w,
      0,
      errors::InvalidArgument("out_w in Attr(out_shape) of Op(interpolate) "
                              "should be greater than 0."));
  DDim dim_out;
  if (data_layout == DataLayout::NCHW) {
    dim_out = {n, c, out_w};
  } else {
    dim_out = {n, out_w, c};
  }
  output->Resize(dim_out);
  dev_ctx.template Alloc<T>(output);

  if (in_w == out_w) {
    Copy(dev_ctx, x, dev_ctx.GetPlace(), false, output);
    return;
  }

  float ratio_w =
      funcs::AreaPixelComputeScale<double>(in_w, out_w, align_corners, scale_w);
  if ("linear" == interp_method) {
    LinearInterpolation<T>(x,
                           output,
                           ratio_w,
                           in_w,
                           n,
                           c,
                           out_w,
                           align_corners,
                           align_mode,
                           data_layout);
  }
}

template <typename T, typename Context>
static void Interpolate2DCPUFwd(
    const Context& dev_ctx,
    const DenseTensor& x,
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const std::string& data_layout_str,
    int out_h,
    int out_w,
    const std::vector<double>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  const DataLayout data_layout = StringToDataLayout(data_layout_str);
  int64_t n = 0, c = 0, in_d = 0, in_h = 0, in_w = 0;
  funcs::ExtractNCDWH(x.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  double scale_h = -1;
  double scale_w = -1;

  if (size_tensor && !size_tensor->empty()) {
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
        scale_h = scale[0];
        scale_w = scale[1];

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
    if (scale_h > 0. && scale_w > 0.) {
      out_h = static_cast<int>(in_h * scale_h);  // NOLINT
      out_w = static_cast<int>(in_w * scale_w);  // NOLINT
    }
    if (out_size) {
      auto out_size_data =
          funcs::get_new_data_from_tensor<int>(out_size.get_ptr());
      out_h = out_size_data[0];
      out_w = out_size_data[1];
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
  DDim dim_out;
  if (data_layout == DataLayout::NCHW) {
    dim_out = {n, c, out_h, out_w};
  } else {
    dim_out = {n, out_h, out_w, c};
  }
  output->Resize(dim_out);
  dev_ctx.template Alloc<T>(output);

  if (in_h == out_h && in_w == out_w) {
    Copy(dev_ctx, x, dev_ctx.GetPlace(), false, output);
    return;
  }

  float ratio_h =
      funcs::AreaPixelComputeScale<float>(in_h, out_h, align_corners, scale_h);
  float ratio_w =
      funcs::AreaPixelComputeScale<float>(in_w, out_w, align_corners, scale_w);

  // TODO(zrr1999): to align xpu
  if (out_h <= 1) {
    ratio_h = 0;
  }
  if (out_w <= 1) {
    ratio_w = 0;
  }

  if ("bilinear" == interp_method) {
    BilinearInterpolation<T>(x,
                             output,
                             ratio_h,
                             ratio_w,
                             in_h,
                             in_w,
                             n,
                             c,
                             out_h,
                             out_w,
                             align_corners,
                             align_mode,
                             data_layout);
  } else if ("nearest" == interp_method) {
    NearestNeighborInterpolate<T>(x,
                                  output,
                                  ratio_h,
                                  ratio_w,
                                  n,
                                  c,
                                  out_h,
                                  out_w,
                                  align_corners,
                                  data_layout);
  } else if ("bicubic" == interp_method) {
    BicubicInterpolation<T>(x,
                            output,
                            ratio_h,
                            ratio_w,
                            in_h,
                            in_w,
                            n,
                            c,
                            out_h,
                            out_w,
                            align_corners,
                            data_layout);
  }
}

template <typename T, typename Context>
static void Interpolate3DCPUFwd(
    const Context& dev_ctx,
    const DenseTensor& x,
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const std::string& data_layout_str,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<double>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  const DataLayout data_layout = StringToDataLayout(data_layout_str);
  int64_t n = 0, c = 0, in_d = 0, in_h = 0, in_w = 0;
  funcs::ExtractNCDWH(x.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  double scale_d = -1;
  double scale_h = -1;
  double scale_w = -1;

  if (size_tensor && !size_tensor->empty()) {
    // have size tensor
    auto new_size = funcs::get_new_shape(size_tensor.get());
    out_d = new_size[0];
    out_h = new_size[1];
    out_w = new_size[2];
  } else {
    if (scale_tensor) {
      auto scale_data =
          funcs::get_new_data_from_tensor<float>(scale_tensor.get_ptr());
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
      if (scale.size() > 1) {
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
    if (scale_w > 0. && scale_h > 0. && scale_d > 0.) {
      out_d = static_cast<int>(in_d * scale_d);  // NOLINT
      out_h = static_cast<int>(in_h * scale_h);  // NOLINT
      out_w = static_cast<int>(in_w * scale_w);  // NOLINT
    }
    if (out_size) {
      auto out_size_data =
          funcs::get_new_data_from_tensor<int>(out_size.get_ptr());
      out_d = out_size_data[0];
      out_h = out_size_data[1];
      out_w = out_size_data[2];
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

  DDim dim_out;
  if (data_layout == DataLayout::NCHW) {
    dim_out = {n, c, out_d, out_h, out_w};
  } else {
    dim_out = {n, out_d, out_h, out_w, c};
  }

  output->Resize(dim_out);
  dev_ctx.template Alloc<T>(output);

  if (in_d == out_d && in_h == out_h && in_w == out_w) {
    Copy(dev_ctx, x, dev_ctx.GetPlace(), false, output);
    return;
  }

  float ratio_d =
      funcs::AreaPixelComputeScale<float>(in_d, out_d, align_corners, scale_d);
  float ratio_h =
      funcs::AreaPixelComputeScale<float>(in_h, out_h, align_corners, scale_h);
  float ratio_w =
      funcs::AreaPixelComputeScale<float>(in_w, out_w, align_corners, scale_w);

  if ("trilinear" == interp_method) {
    TrilinearInterpolation<T>(x,
                              output,
                              ratio_d,
                              ratio_h,
                              ratio_w,
                              in_d,
                              in_h,
                              in_w,
                              n,
                              c,
                              out_d,
                              out_h,
                              out_w,
                              align_corners,
                              align_mode,
                              data_layout);
  } else if ("nearest" == interp_method) {
    NearestNeighbor3DInterpolate<T>(x,
                                    output,
                                    ratio_d,
                                    ratio_h,
                                    ratio_w,
                                    n,
                                    c,
                                    out_d,
                                    out_h,
                                    out_w,
                                    align_corners,
                                    data_layout);
  }
}

template <typename T, typename Context>
void InterpolateKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<double>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  if (x.numel() == 0) {
    dev_ctx.template Alloc<T>(output);
    return;
  }

  auto input_dims = x.dims();
  if (input_dims.size() == 3) {  // 1D interpolation
    Interpolate1DCPUFwd<T, Context>(dev_ctx,
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
    Interpolate2DCPUFwd<T>(dev_ctx,
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
    Interpolate3DCPUFwd<T>(dev_ctx,
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
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<double>& scale,
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
void LegacyBilinearInterpKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    float scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  const auto& dim_x = x.dims();
  std::vector<double> scale_vec;
  if (scale > 0) {
    for (int i = 0; i < dim_x.size() - 2; i++) {
      scale_vec.push_back(scale);
    }
  }
  InterpolateKernel<T, Context>(dev_ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale_vec,
                                interp_method,
                                align_corners,
                                align_mode,
                                output);
}

template <typename T, typename Context>
void NearestInterpKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<double>& scale,
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
void LegacyNearestInterpKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    float scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  const auto& dim_x = x.dims();
  std::vector<double> scale_vec;
  if (scale > 0) {
    for (int i = 0; i < dim_x.size() - 2; i++) {
      scale_vec.push_back(scale);
    }
  }
  InterpolateKernel<T, Context>(dev_ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale_vec,
                                interp_method,
                                align_corners,
                                align_mode,
                                output);
}

template <typename T, typename Context>
void TrilinearInterpKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<double>& scale,
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
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<double>& scale,
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
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<double>& scale,
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

// =====================================================================
// CPU Antialias Interpolation Forward Implementation
// Separable 2-pass AA interpolation matching PyTorch's behavior exactly.
// =====================================================================

// CPU weight computation for antialias interpolation.
// Matches the GPU ComputeWeights function and PyTorch's weight computation.
template <typename WT, typename InterpFilter>
static void ComputeAAWeightsCPU(WT* wt_ptr,
                                const WT scale,
                                int interp_size,
                                const InterpFilter& interp_filter,
                                WT xmin_m_center,
                                int xsize) {
  WT invscale = (scale >= static_cast<WT>(1.0)) ? static_cast<WT>(1.0) / scale
                                                : static_cast<WT>(1.0);
  WT total_w = static_cast<WT>(0.0);
  int j = 0;
  for (j = 0; j < xsize; j++) {
    WT w = interp_filter((j + xmin_m_center + static_cast<WT>(0.5)) * invscale);
    wt_ptr[j] = w;
    total_w += w;
  }
  for (j = 0; j < xsize; j++) {
    if (total_w != static_cast<WT>(0.0)) {
      wt_ptr[j] /= total_w;
    }
  }
  for (; j < interp_size; j++) {
    wt_ptr[j] = static_cast<WT>(0.0);
  }
}

// CPU weight span computation matching the GPU ComputeWeightsSpan.
template <typename WT>
static void ComputeAAWeightsSpanCPU(const int i,
                                    const int input_size,
                                    const WT scale,
                                    const WT support,
                                    int* xmin,
                                    int* xsize,
                                    WT* center) {
  *center = scale * (i + static_cast<WT>(0.5));
  *xmin = std::max(
      static_cast<int>(std::floor(*center - support + static_cast<WT>(0.5))),
      0);
  *xsize = std::min(static_cast<int>(
                        std::floor(*center + support + static_cast<WT>(0.5))),
                    input_size) -
           *xmin;
}

// Single dimension AA interpolation for float types on CPU.
// Computes weighted sum: sum(src[j] * weights[j]) for j in [0, size).
template <typename T, typename WT>
static WT InterpolateAASingleDimCPU(const T* src, const WT* weights, int size) {
  WT output = static_cast<WT>(src[0]) * weights[0];
  for (int j = 1; j < size; j++) {
    output += static_cast<WT>(src[j]) * weights[j];
  }
  return output;
}

// Forward pass: separable 2-pass AA interpolation for float types, NCHW.
// Pass 1 (horizontal): input [N,C,H_in,W_in] -> temp [N,C,H_in,W_out]
// Pass 2 (vertical):   temp  [N,C,H_in,W_out] -> output [N,C,H_out,W_out]
template <typename T, typename InterpFilter>
static void AAInterpolation2DCPU_NCHW(const T* input_data,
                                      T* output_data,
                                      int64_t n,
                                      int64_t c,
                                      int in_h,
                                      int in_w,
                                      int out_h,
                                      int out_w,
                                      float ratio_h,
                                      float ratio_w,
                                      const InterpFilter& filter) {
  // Use MPTypeTrait to match GPU: float for float/float16/bfloat16, double for
  // double
  using WT = typename phi::dtype::MPTypeTrait<T>::Type;
  WT scale_h = static_cast<WT>(ratio_h);
  WT scale_w = static_cast<WT>(ratio_w);

  const WT half = static_cast<WT>(0.5);
  const WT support_h = (scale_h >= static_cast<WT>(1.0))
                           ? (filter.size * half) * scale_h
                           : filter.size * half;
  const WT support_w = (scale_w >= static_cast<WT>(1.0))
                           ? (filter.size * half) * scale_w
                           : filter.size * half;

  const int interp_height = static_cast<int>(std::ceil(support_h)) * 2 + 1;
  const int interp_width = static_cast<int>(std::ceil(support_w)) * 2 + 1;

  // Allocate temporary buffer for intermediate result [N, C, H_in, W_out]
  // and weight arrays
  std::vector<T> temp(static_cast<size_t>(n) * c * in_h * out_w);
  std::vector<WT> wx(interp_width);
  std::vector<WT> wy(interp_height);

  // Pre-compute horizontal weights and spans for each output column
  struct SpanInfo {
    int xmin;
    int xsize;
    WT center;
  };
  std::vector<SpanInfo> h_spans(out_w);
  std::vector<std::vector<WT>> h_weights(out_w);
  for (int ow = 0; ow < out_w; ow++) {
    ComputeAAWeightsSpanCPU<WT>(ow,
                                in_w,
                                scale_w,
                                support_w,
                                &h_spans[ow].xmin,
                                &h_spans[ow].xsize,
                                &h_spans[ow].center);
    h_weights[ow].resize(interp_width);
    ComputeAAWeightsCPU<WT>(
        h_weights[ow].data(),
        scale_w,
        interp_width,
        filter,
        static_cast<WT>(h_spans[ow].xmin) - h_spans[ow].center,
        h_spans[ow].xsize);
  }

  // Pre-compute vertical weights and spans for each output row
  std::vector<SpanInfo> v_spans(out_h);
  std::vector<std::vector<WT>> v_weights(out_h);
  for (int oh = 0; oh < out_h; oh++) {
    ComputeAAWeightsSpanCPU<WT>(oh,
                                in_h,
                                scale_h,
                                support_h,
                                &v_spans[oh].xmin,
                                &v_spans[oh].xsize,
                                &v_spans[oh].center);
    v_weights[oh].resize(interp_height);
    ComputeAAWeightsCPU<WT>(
        v_weights[oh].data(),
        scale_h,
        interp_height,
        filter,
        static_cast<WT>(v_spans[oh].xmin) - v_spans[oh].center,
        v_spans[oh].xsize);
  }

  // Pass 1: Horizontal interpolation
  // For each (batch, channel, input_row), interpolate across width
  for (int64_t nc_idx = 0; nc_idx < n * c; nc_idx++) {
    for (int ih = 0; ih < in_h; ih++) {
      const T* in_row = input_data + nc_idx * in_h * in_w + ih * in_w;
      T* temp_row = temp.data() + nc_idx * in_h * out_w + ih * out_w;

      for (int ow = 0; ow < out_w; ow++) {
        int xmin = h_spans[ow].xmin;
        int xsize = h_spans[ow].xsize;
        const WT* wts = h_weights[ow].data();

        WT result = static_cast<WT>(0);
        for (int j = 0; j < xsize; j++) {
          result += static_cast<WT>(in_row[xmin + j]) * wts[j];
        }
        temp_row[ow] = static_cast<T>(result);
      }
    }
  }

  // Pass 2: Vertical interpolation
  // For each (batch, channel, output_col), interpolate across height
  for (int64_t nc_idx = 0; nc_idx < n * c; nc_idx++) {
    for (int oh = 0; oh < out_h; oh++) {
      int ymin = v_spans[oh].xmin;
      int ysize = v_spans[oh].xsize;
      const WT* wts = v_weights[oh].data();

      T* out_row = output_data + nc_idx * out_h * out_w + oh * out_w;

      for (int ow = 0; ow < out_w; ow++) {
        WT result = static_cast<WT>(0);
        for (int j = 0; j < ysize; j++) {
          const T* temp_row =
              temp.data() + nc_idx * in_h * out_w + (ymin + j) * out_w;
          result += static_cast<WT>(temp_row[ow]) * wts[j];
        }
        out_row[ow] = static_cast<T>(result);
      }
    }
  }
}

// Forward pass: separable 2-pass AA interpolation for float types, NHWC.
template <typename T, typename InterpFilter>
static void AAInterpolation2DCPU_NHWC(const T* input_data,
                                      T* output_data,
                                      int64_t n,
                                      int64_t c,
                                      int in_h,
                                      int in_w,
                                      int out_h,
                                      int out_w,
                                      float ratio_h,
                                      float ratio_w,
                                      const InterpFilter& filter) {
  // Use MPTypeTrait to match GPU: float for float/float16/bfloat16, double for
  // double
  using WT = typename phi::dtype::MPTypeTrait<T>::Type;
  WT scale_h = static_cast<WT>(ratio_h);
  WT scale_w = static_cast<WT>(ratio_w);

  const WT half = static_cast<WT>(0.5);
  const WT support_h = (scale_h >= static_cast<WT>(1.0))
                           ? (filter.size * half) * scale_h
                           : filter.size * half;
  const WT support_w = (scale_w >= static_cast<WT>(1.0))
                           ? (filter.size * half) * scale_w
                           : filter.size * half;

  const int interp_height = static_cast<int>(std::ceil(support_h)) * 2 + 1;
  const int interp_width = static_cast<int>(std::ceil(support_w)) * 2 + 1;

  // Temporary buffer: [N, H_in, W_out, C]
  std::vector<T> temp(static_cast<size_t>(n) * in_h * out_w * c);

  // Pre-compute horizontal weights
  struct SpanInfo {
    int xmin;
    int xsize;
    WT center;
  };
  std::vector<SpanInfo> h_spans(out_w);
  std::vector<std::vector<WT>> h_weights(out_w);
  for (int ow = 0; ow < out_w; ow++) {
    ComputeAAWeightsSpanCPU<WT>(ow,
                                in_w,
                                scale_w,
                                support_w,
                                &h_spans[ow].xmin,
                                &h_spans[ow].xsize,
                                &h_spans[ow].center);
    h_weights[ow].resize(interp_width);
    ComputeAAWeightsCPU<WT>(
        h_weights[ow].data(),
        scale_w,
        interp_width,
        filter,
        static_cast<WT>(h_spans[ow].xmin) - h_spans[ow].center,
        h_spans[ow].xsize);
  }

  // Pre-compute vertical weights
  std::vector<SpanInfo> v_spans(out_h);
  std::vector<std::vector<WT>> v_weights(out_h);
  for (int oh = 0; oh < out_h; oh++) {
    ComputeAAWeightsSpanCPU<WT>(oh,
                                in_h,
                                scale_h,
                                support_h,
                                &v_spans[oh].xmin,
                                &v_spans[oh].xsize,
                                &v_spans[oh].center);
    v_weights[oh].resize(interp_height);
    ComputeAAWeightsCPU<WT>(
        v_weights[oh].data(),
        scale_h,
        interp_height,
        filter,
        static_cast<WT>(v_spans[oh].xmin) - v_spans[oh].center,
        v_spans[oh].xsize);
  }

  // Pass 1: Horizontal - input [N,H_in,W_in,C] -> temp [N,H_in,W_out,C]
  for (int64_t bi = 0; bi < n; bi++) {
    for (int ih = 0; ih < in_h; ih++) {
      for (int ow = 0; ow < out_w; ow++) {
        int xmin = h_spans[ow].xmin;
        int xsize = h_spans[ow].xsize;
        const WT* wts = h_weights[ow].data();

        for (int64_t ch = 0; ch < c; ch++) {
          WT result = static_cast<WT>(0);
          for (int j = 0; j < xsize; j++) {
            int64_t in_idx = ((bi * in_h + ih) * in_w + (xmin + j)) * c + ch;
            result += static_cast<WT>(input_data[in_idx]) * wts[j];
          }
          int64_t temp_idx = ((bi * in_h + ih) * out_w + ow) * c + ch;
          temp[temp_idx] = static_cast<T>(result);
        }
      }
    }
  }

  // Pass 2: Vertical - temp [N,H_in,W_out,C] -> output [N,H_out,W_out,C]
  for (int64_t bi = 0; bi < n; bi++) {
    for (int oh = 0; oh < out_h; oh++) {
      int ymin = v_spans[oh].xmin;
      int ysize = v_spans[oh].xsize;
      const WT* wts = v_weights[oh].data();

      for (int ow = 0; ow < out_w; ow++) {
        for (int64_t ch = 0; ch < c; ch++) {
          WT result = static_cast<WT>(0);
          for (int j = 0; j < ysize; j++) {
            int64_t temp_idx = ((bi * in_h + (ymin + j)) * out_w + ow) * c + ch;
            result += static_cast<WT>(temp[temp_idx]) * wts[j];
          }
          int64_t out_idx = ((bi * out_h + oh) * out_w + ow) * c + ch;
          output_data[out_idx] = static_cast<T>(result);
        }
      }
    }
  }
}

// Specialization for uint8_t: uses double weights, int16 quantization,
// int32 accumulation -- matching PyTorch's Pillow-compatible uint8 path.
template <typename InterpFilter>
static void AAInterpolation2DCPU_NCHW_UInt8(const uint8_t* input_data,
                                            uint8_t* output_data,
                                            int64_t n,
                                            int64_t c,
                                            int in_h,
                                            int in_w,
                                            int out_h,
                                            int out_w,
                                            float ratio_h,
                                            float ratio_w,
                                            const InterpFilter& filter) {
  using WT = double;
  WT scale_h = static_cast<WT>(ratio_h);
  WT scale_w = static_cast<WT>(ratio_w);

  const WT half = 0.5;
  const WT support_h =
      (scale_h >= 1.0) ? (filter.size * half) * scale_h : filter.size * half;
  const WT support_w =
      (scale_w >= 1.0) ? (filter.size * half) * scale_w : filter.size * half;

  const int interp_height = static_cast<int>(std::ceil(support_h)) * 2 + 1;
  const int interp_width = static_cast<int>(std::ceil(support_w)) * 2 + 1;

  struct SpanInfo {
    int xmin;
    int xsize;
    WT center;
  };

  // Helper: compute double weights, then quantize to int16 as PyTorch does
  auto compute_int16_weights = [&](const std::vector<WT>& dbl_weights,
                                   int xsize,
                                   std::vector<int16_t>& i16_weights,
                                   unsigned int& precision) {
    // Find maximum weight
    WT wt_max = 0.0;
    for (int j = 0; j < xsize; j++) {
      WT aw = dbl_weights[j] < 0 ? -dbl_weights[j] : dbl_weights[j];
      if (aw > wt_max) wt_max = aw;
    }
    // Find max precision P such that round(max_weight * 2^(P+1)) < 2^15
    unsigned int P = 0;
    for (P = 0; P < 22; ++P) {
      int next_value = static_cast<int>(0.5 + wt_max * (1 << (P + 1)));
      if (next_value >= (1 << 15)) break;
    }
    precision = P;
    i16_weights.resize(xsize);
    for (int j = 0; j < xsize; j++) {
      i16_weights[j] =
          static_cast<int16_t>(std::round(dbl_weights[j] * (1 << P)));
    }
  };

  // Pre-compute horizontal weights (double) and quantized int16 weights
  std::vector<SpanInfo> h_spans(out_w);
  std::vector<std::vector<WT>> h_dbl_weights(out_w);
  std::vector<std::vector<int16_t>> h_i16_weights(out_w);
  std::vector<unsigned int> h_precision(out_w);

  for (int ow = 0; ow < out_w; ow++) {
    ComputeAAWeightsSpanCPU<WT>(ow,
                                in_w,
                                scale_w,
                                support_w,
                                &h_spans[ow].xmin,
                                &h_spans[ow].xsize,
                                &h_spans[ow].center);
    h_dbl_weights[ow].resize(interp_width);
    ComputeAAWeightsCPU<WT>(
        h_dbl_weights[ow].data(),
        scale_w,
        interp_width,
        filter,
        static_cast<WT>(h_spans[ow].xmin) - h_spans[ow].center,
        h_spans[ow].xsize);
    compute_int16_weights(h_dbl_weights[ow],
                          h_spans[ow].xsize,
                          h_i16_weights[ow],
                          h_precision[ow]);
  }

  // Pre-compute vertical weights
  std::vector<SpanInfo> v_spans(out_h);
  std::vector<std::vector<WT>> v_dbl_weights(out_h);
  std::vector<std::vector<int16_t>> v_i16_weights(out_h);
  std::vector<unsigned int> v_precision(out_h);

  for (int oh = 0; oh < out_h; oh++) {
    ComputeAAWeightsSpanCPU<WT>(oh,
                                in_h,
                                scale_h,
                                support_h,
                                &v_spans[oh].xmin,
                                &v_spans[oh].xsize,
                                &v_spans[oh].center);
    v_dbl_weights[oh].resize(interp_height);
    ComputeAAWeightsCPU<WT>(
        v_dbl_weights[oh].data(),
        scale_h,
        interp_height,
        filter,
        static_cast<WT>(v_spans[oh].xmin) - v_spans[oh].center,
        v_spans[oh].xsize);
    compute_int16_weights(v_dbl_weights[oh],
                          v_spans[oh].xsize,
                          v_i16_weights[oh],
                          v_precision[oh]);
  }

  // Temporary buffer [N, C, H_in, W_out] as uint8
  std::vector<uint8_t> temp(static_cast<size_t>(n) * c * in_h * out_w);

  // Pass 1: Horizontal interpolation with int16 weights / int32 accumulation
  for (int64_t nc_idx = 0; nc_idx < n * c; nc_idx++) {
    for (int ih = 0; ih < in_h; ih++) {
      const uint8_t* in_row = input_data + nc_idx * in_h * in_w + ih * in_w;
      uint8_t* temp_row = temp.data() + nc_idx * in_h * out_w + ih * out_w;

      for (int ow = 0; ow < out_w; ow++) {
        int xmin = h_spans[ow].xmin;
        int xsize = h_spans[ow].xsize;
        unsigned int P = h_precision[ow];
        const int16_t* i16w = h_i16_weights[ow].data();

        int32_t accum = 1 << (P > 0 ? P - 1 : 0);  // rounding bias
        for (int j = 0; j < xsize; j++) {
          accum += static_cast<int32_t>(in_row[xmin + j]) *
                   static_cast<int32_t>(i16w[j]);
        }
        int32_t result = accum >> P;
        temp_row[ow] = static_cast<uint8_t>(std::max(0, std::min(255, result)));
      }
    }
  }

  // Pass 2: Vertical interpolation
  for (int64_t nc_idx = 0; nc_idx < n * c; nc_idx++) {
    for (int oh = 0; oh < out_h; oh++) {
      int ymin = v_spans[oh].xmin;
      int ysize = v_spans[oh].xsize;
      unsigned int P = v_precision[oh];
      const int16_t* i16w = v_i16_weights[oh].data();

      uint8_t* out_row = output_data + nc_idx * out_h * out_w + oh * out_w;

      for (int ow = 0; ow < out_w; ow++) {
        int32_t accum = 1 << (P > 0 ? P - 1 : 0);
        for (int j = 0; j < ysize; j++) {
          const uint8_t* temp_row =
              temp.data() + nc_idx * in_h * out_w + (ymin + j) * out_w;
          accum += static_cast<int32_t>(temp_row[ow]) *
                   static_cast<int32_t>(i16w[j]);
        }
        int32_t result = accum >> P;
        out_row[ow] = static_cast<uint8_t>(std::max(0, std::min(255, result)));
      }
    }
  }
}

// NHWC variant for uint8
template <typename InterpFilter>
static void AAInterpolation2DCPU_NHWC_UInt8(const uint8_t* input_data,
                                            uint8_t* output_data,
                                            int64_t n,
                                            int64_t c,
                                            int in_h,
                                            int in_w,
                                            int out_h,
                                            int out_w,
                                            float ratio_h,
                                            float ratio_w,
                                            const InterpFilter& filter) {
  using WT = double;
  WT scale_h = static_cast<WT>(ratio_h);
  WT scale_w = static_cast<WT>(ratio_w);

  const WT half = 0.5;
  const WT support_h =
      (scale_h >= 1.0) ? (filter.size * half) * scale_h : filter.size * half;
  const WT support_w =
      (scale_w >= 1.0) ? (filter.size * half) * scale_w : filter.size * half;

  const int interp_height = static_cast<int>(std::ceil(support_h)) * 2 + 1;
  const int interp_width = static_cast<int>(std::ceil(support_w)) * 2 + 1;

  struct SpanInfo {
    int xmin;
    int xsize;
    WT center;
  };

  auto compute_int16_weights = [&](const std::vector<WT>& dbl_weights,
                                   int xsize,
                                   std::vector<int16_t>& i16_weights,
                                   unsigned int& precision) {
    WT wt_max = 0.0;
    for (int j = 0; j < xsize; j++) {
      WT aw = dbl_weights[j] < 0 ? -dbl_weights[j] : dbl_weights[j];
      if (aw > wt_max) wt_max = aw;
    }
    unsigned int P = 0;
    for (P = 0; P < 22; ++P) {
      int next_value = static_cast<int>(0.5 + wt_max * (1 << (P + 1)));
      if (next_value >= (1 << 15)) break;
    }
    precision = P;
    i16_weights.resize(xsize);
    for (int j = 0; j < xsize; j++) {
      i16_weights[j] =
          static_cast<int16_t>(std::round(dbl_weights[j] * (1 << P)));
    }
  };

  std::vector<SpanInfo> h_spans(out_w);
  std::vector<std::vector<WT>> h_dbl_weights(out_w);
  std::vector<std::vector<int16_t>> h_i16_weights(out_w);
  std::vector<unsigned int> h_precision(out_w);
  for (int ow = 0; ow < out_w; ow++) {
    ComputeAAWeightsSpanCPU<WT>(ow,
                                in_w,
                                scale_w,
                                support_w,
                                &h_spans[ow].xmin,
                                &h_spans[ow].xsize,
                                &h_spans[ow].center);
    h_dbl_weights[ow].resize(interp_width);
    ComputeAAWeightsCPU<WT>(
        h_dbl_weights[ow].data(),
        scale_w,
        interp_width,
        filter,
        static_cast<WT>(h_spans[ow].xmin) - h_spans[ow].center,
        h_spans[ow].xsize);
    compute_int16_weights(h_dbl_weights[ow],
                          h_spans[ow].xsize,
                          h_i16_weights[ow],
                          h_precision[ow]);
  }

  std::vector<SpanInfo> v_spans(out_h);
  std::vector<std::vector<WT>> v_dbl_weights(out_h);
  std::vector<std::vector<int16_t>> v_i16_weights(out_h);
  std::vector<unsigned int> v_precision(out_h);
  for (int oh = 0; oh < out_h; oh++) {
    ComputeAAWeightsSpanCPU<WT>(oh,
                                in_h,
                                scale_h,
                                support_h,
                                &v_spans[oh].xmin,
                                &v_spans[oh].xsize,
                                &v_spans[oh].center);
    v_dbl_weights[oh].resize(interp_height);
    ComputeAAWeightsCPU<WT>(
        v_dbl_weights[oh].data(),
        scale_h,
        interp_height,
        filter,
        static_cast<WT>(v_spans[oh].xmin) - v_spans[oh].center,
        v_spans[oh].xsize);
    compute_int16_weights(v_dbl_weights[oh],
                          v_spans[oh].xsize,
                          v_i16_weights[oh],
                          v_precision[oh]);
  }

  // Temp buffer [N, H_in, W_out, C] as uint8
  std::vector<uint8_t> temp(static_cast<size_t>(n) * in_h * out_w * c);

  // Pass 1: Horizontal
  for (int64_t bi = 0; bi < n; bi++) {
    for (int ih = 0; ih < in_h; ih++) {
      for (int ow = 0; ow < out_w; ow++) {
        int xmin = h_spans[ow].xmin;
        int xsize = h_spans[ow].xsize;
        unsigned int P = h_precision[ow];
        const int16_t* i16w = h_i16_weights[ow].data();

        for (int64_t ch = 0; ch < c; ch++) {
          int32_t accum = 1 << (P > 0 ? P - 1 : 0);
          for (int j = 0; j < xsize; j++) {
            int64_t in_idx = ((bi * in_h + ih) * in_w + (xmin + j)) * c + ch;
            accum += static_cast<int32_t>(input_data[in_idx]) *
                     static_cast<int32_t>(i16w[j]);
          }
          int32_t result = accum >> P;
          int64_t temp_idx = ((bi * in_h + ih) * out_w + ow) * c + ch;
          temp[temp_idx] =
              static_cast<uint8_t>(std::max(0, std::min(255, result)));
        }
      }
    }
  }

  // Pass 2: Vertical
  for (int64_t bi = 0; bi < n; bi++) {
    for (int oh = 0; oh < out_h; oh++) {
      int ymin = v_spans[oh].xmin;
      int ysize = v_spans[oh].xsize;
      unsigned int P = v_precision[oh];
      const int16_t* i16w = v_i16_weights[oh].data();

      for (int ow = 0; ow < out_w; ow++) {
        for (int64_t ch = 0; ch < c; ch++) {
          int32_t accum = 1 << (P > 0 ? P - 1 : 0);
          for (int j = 0; j < ysize; j++) {
            int64_t temp_idx = ((bi * in_h + (ymin + j)) * out_w + ow) * c + ch;
            accum += static_cast<int32_t>(temp[temp_idx]) *
                     static_cast<int32_t>(i16w[j]);
          }
          int32_t result = accum >> P;
          int64_t out_idx = ((bi * out_h + oh) * out_w + ow) * c + ch;
          output_data[out_idx] =
              static_cast<uint8_t>(std::max(0, std::min(255, result)));
        }
      }
    }
  }
}

// Dispatcher: selects NCHW/NHWC and float/uint8 paths
template <typename T, typename InterpFilter>
static void AAInterpolation2DCPUDispatch(const T* input_data,
                                         T* output_data,
                                         int64_t n,
                                         int64_t c,
                                         int in_h,
                                         int in_w,
                                         int out_h,
                                         int out_w,
                                         float ratio_h,
                                         float ratio_w,
                                         const DataLayout data_layout,
                                         const InterpFilter& filter) {
  if (data_layout == DataLayout::NCHW) {
    AAInterpolation2DCPU_NCHW<T>(input_data,
                                 output_data,
                                 n,
                                 c,
                                 in_h,
                                 in_w,
                                 out_h,
                                 out_w,
                                 ratio_h,
                                 ratio_w,
                                 filter);
  } else {
    AAInterpolation2DCPU_NHWC<T>(input_data,
                                 output_data,
                                 n,
                                 c,
                                 in_h,
                                 in_w,
                                 out_h,
                                 out_w,
                                 ratio_h,
                                 ratio_w,
                                 filter);
  }
}

// Explicit specialization for uint8_t dispatch
template <typename InterpFilter>
static void AAInterpolation2DCPUDispatchUInt8(const uint8_t* input_data,
                                              uint8_t* output_data,
                                              int64_t n,
                                              int64_t c,
                                              int in_h,
                                              int in_w,
                                              int out_h,
                                              int out_w,
                                              float ratio_h,
                                              float ratio_w,
                                              const DataLayout data_layout,
                                              const InterpFilter& filter) {
  if (data_layout == DataLayout::NCHW) {
    AAInterpolation2DCPU_NCHW_UInt8(input_data,
                                    output_data,
                                    n,
                                    c,
                                    in_h,
                                    in_w,
                                    out_h,
                                    out_w,
                                    ratio_h,
                                    ratio_w,
                                    filter);
  } else {
    AAInterpolation2DCPU_NHWC_UInt8(input_data,
                                    output_data,
                                    n,
                                    c,
                                    in_h,
                                    in_w,
                                    out_h,
                                    out_w,
                                    ratio_h,
                                    ratio_w,
                                    filter);
  }
}

// Main CPU forward function for AA 2D interpolation.
// Parses output size from out_size/size_tensor/scale_tensor/scale params
// (same logic as GPU InterpolateAA2DCUDAFwd), then dispatches to the
// separable 2-pass interpolation.
template <typename T, typename Context>
static void InterpolateAA2DCPUFwd(
    const Context& dev_ctx,
    const DenseTensor& input,
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const std::string& data_layout_str,
    int out_h,
    int out_w,
    const std::vector<double>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  if (input.numel() == 0) {
    dev_ctx.template Alloc<T>(output);
    return;
  }
  auto* input_data = input.data<T>();

  const DataLayout data_layout = StringToDataLayout(data_layout_str);
  int64_t n, c, in_d, in_h, in_w;
  funcs::ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  double scale_w = -1;
  double scale_h = -1;
  if (size_tensor && !size_tensor->empty()) {
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
      auto out_size_data =
          funcs::get_new_data_from_tensor<int>(out_size.get_ptr());
      out_h = out_size_data[0];
      out_w = out_size_data[1];
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

  DDim dim_out;
  if (data_layout == DataLayout::NCHW) {
    dim_out = {n, c, out_h, out_w};
  } else {
    dim_out = {n, out_h, out_w, c};
  }
  output->Resize(dim_out);
  auto output_data = dev_ctx.template Alloc<T>(output);

  if (in_h == out_h && in_w == out_w) {
    Copy(dev_ctx, input, dev_ctx.GetPlace(), false, output);
    return;
  }

  // Use conditional type: float for integral/half types, double for double
  using MT =
      typename std::conditional_t<std::is_integral<T>::value,
                                  float,
                                  typename phi::dtype::MPTypeTrait<T>::Type>;
  MT ratio_h =
      funcs::AreaPixelComputeScale<MT>(in_h, out_h, align_corners, scale_h);
  MT ratio_w =
      funcs::AreaPixelComputeScale<MT>(in_w, out_w, align_corners, scale_w);

  // Dispatch based on interp_method and dtype
  auto launch_aa = [&](auto filter_functor) {
    if constexpr (std::is_same<T, uint8_t>::value) {
      AAInterpolation2DCPUDispatchUInt8(input_data,
                                        output_data,
                                        n,
                                        c,
                                        in_h,
                                        in_w,
                                        out_h,
                                        out_w,
                                        static_cast<float>(ratio_h),
                                        static_cast<float>(ratio_w),
                                        data_layout,
                                        filter_functor);
    } else {
      AAInterpolation2DCPUDispatch<T>(input_data,
                                      output_data,
                                      n,
                                      c,
                                      in_h,
                                      in_w,
                                      out_h,
                                      out_w,
                                      static_cast<float>(ratio_h),
                                      static_cast<float>(ratio_w),
                                      data_layout,
                                      filter_functor);
    }
  };

  if ("bilinear" == interp_method) {
    launch_aa(funcs::antialias::BilinearFilterFunctor{});
  } else if ("bicubic" == interp_method) {
    launch_aa(funcs::antialias::BicubicFilterFunctor{});
  }
}

template <typename T, typename Context>
void InterpAntialiasKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<double>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  InterpolateAA2DCPUFwd<T, Context>(dev_ctx,
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
}

}  // namespace phi

PD_REGISTER_KERNEL(bilinear_interp,
                   CPU,
                   ALL_LAYOUT,
                   phi::BilinearInterpKernel,
                   float,
                   double,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(legacy_bilinear_interp,
                   CPU,
                   ALL_LAYOUT,
                   phi::LegacyBilinearInterpKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(nearest_interp,
                   CPU,
                   ALL_LAYOUT,
                   phi::NearestInterpKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(legacy_nearest_interp,
                   CPU,
                   ALL_LAYOUT,
                   phi::LegacyNearestInterpKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(trilinear_interp,
                   CPU,
                   ALL_LAYOUT,
                   phi::TrilinearInterpKernel,
                   float,
                   double,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(linear_interp,
                   CPU,
                   ALL_LAYOUT,
                   phi::LinearInterpKernel,
                   float,
                   double,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(bicubic_interp,
                   CPU,
                   ALL_LAYOUT,
                   phi::BicubicInterpKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_KERNEL(interp_antialias,
                   CPU,
                   ALL_LAYOUT,
                   phi::InterpAntialiasKernel,
                   float,
                   double,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
