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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/interpolate_function.h"

namespace phi {

template <typename T>
static inline T cubic_interp(T x0, T x1, T x2, T x3, T t) {
  T coeffs[4];
  funcs::get_cubic_upsample_coefficients<T>(coeffs, t);

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
    int x_w = align_flag ? static_cast<int>(ratio_w * (l + 0.5) - 0.5)
                         : static_cast<int>(ratio_w * l);
    x_w = (x_w > 0) ? x_w : 0;                       // w
    int x_e = (x_w < (in_w - 1)) ? (x_w + 1) : x_w;  // w_id

    float idx_src_x = ratio_w * (l + 0.5) - 0.5;
    idx_src_x = (idx_src_x > 0) ? idx_src_x : 0;
    float d_w = align_flag ? idx_src_x - x_w : ratio_w * l - x_w;  // w1lambda
    float d_e = 1.f - d_w;                                         // w2lambda
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
        if (data_layout == DataLayout::kNCHW) {
          out_t = input_t(i, j, vx_w[l]) * vd_e[l] +
                  input_t(i, j, vx_e[l]) * vd_w[l];
          output_t(i, j, l) = out_t;
        } else {
          out_t = input_t(i, vx_w[l], j) * vd_e[l] +
                  input_t(i, vx_e[l], j) * vd_w[l];
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
                         : static_cast<int>(ratio_h * k);
    y_n = (y_n > 0) ? y_n : 0;
    int y_s = (y_n + 1) < (in_h - 1) ? (y_n + 1) : (in_h - 1);
    float idx_src_y = ratio_h * (k + 0.5) - 0.5;
    idx_src_y = (idx_src_y > 0) ? idx_src_y : 0;
    float d_n = align_flag ? idx_src_y - y_n : ratio_h * k - y_n;
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
                  : static_cast<int>(ratio_w * l);
    x_w = (x_w > 0) ? x_w : 0;
    int x_e = (x_w + 1) < (in_w - 1) ? (x_w + 1) : (in_w - 1);
    float idx_src_x = ratio_w * (l + 0.5) - 0.5;
    idx_src_x = (idx_src_x > 0) ? idx_src_x : 0;
    float d_w = align_flag ? idx_src_x - x_w : ratio_w * l - x_w;
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
          if (data_layout == DataLayout::kNCHW) {
            out_t = input_t(i, j, vy_n[k], vx_w[l]) * vd_s[k] * vd_e[l] +
                    input_t(i, j, vy_s[k], vx_w[l]) * vd_n[k] * vd_e[l] +
                    input_t(i, j, vy_n[k], vx_e[l]) * vd_s[k] * vd_w[l] +
                    input_t(i, j, vy_s[k], vx_e[l]) * vd_n[k] * vd_w[l];
            output_t(i, j, k, l) = out_t;

          } else {
            out_t = input_t(i, vy_n[k], vx_w[l], j) * vd_s[k] * vd_e[l] +
                    input_t(i, vy_s[k], vx_w[l], j) * vd_n[k] * vd_e[l] +
                    input_t(i, vy_n[k], vx_e[l], j) * vd_s[k] * vd_w[l] +
                    input_t(i, vy_s[k], vx_e[l], j) * vd_n[k] * vd_w[l];
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
    int in_k = (align_corners) ? static_cast<int>(ratio_h * k + 0.5)
                               : static_cast<int>(ratio_h * k);

    for (int l = 0; l < out_w; l++) {
      int in_l = (align_corners) ? static_cast<int>(ratio_w * l + 0.5)
                                 : static_cast<int>(ratio_w * l);

      for (int i = 0; i < n; i++) {    // loop for batches
        for (int j = 0; j < c; j++) {  // loop for channels
          if (data_layout == DataLayout::kNCHW) {
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

  for (int k = 0; k < out_h; k++) {  // loop for images
    T y_n = align_corners ? static_cast<T>(ratio_h * k)
                          : static_cast<T>(ratio_h * (k + 0.5) - 0.5);
    int input_y = floorf(y_n);
    const T y_t = y_n - input_y;

    for (int l = 0; l < out_w; l++) {
      T x_n = align_corners ? static_cast<T>(ratio_w * l)
                            : static_cast<T>(ratio_w * (l + 0.5) - 0.5);
      int input_x = floorf(x_n);
      const T x_t = x_n - input_x;

      for (int i = 0; i < n; i++) {    // loop for batches
        for (int j = 0; j < c; j++) {  // loop for channels
          T coefficients[4];
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
            if (data_layout == DataLayout::kNCHW) {
              coefficients[ii] =
                  cubic_interp<T>(input_t(i, j, access_y, access_x_0),
                                  input_t(i, j, access_y, access_x_1),
                                  input_t(i, j, access_y, access_x_2),
                                  input_t(i, j, access_y, access_x_3),
                                  x_t);
            } else {
              coefficients[ii] =
                  cubic_interp<T>(input_t(i, access_y, access_x_0, j),
                                  input_t(i, access_y, access_x_1, j),
                                  input_t(i, access_y, access_x_2, j),
                                  input_t(i, access_y, access_x_3, j),
                                  x_t);
            }
          }

          // interp y direction
          if (data_layout == DataLayout::kNCHW) {
            output_t(i, j, k, l) = cubic_interp<T>(coefficients[0],
                                                   coefficients[1],
                                                   coefficients[2],
                                                   coefficients[3],
                                                   y_t);
          } else {
            output_t(i, k, l, j) = cubic_interp<T>(coefficients[0],
                                                   coefficients[1],
                                                   coefficients[2],
                                                   coefficients[3],
                                                   y_t);
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
                         : static_cast<int>(ratio_d * j);
    t_f = (t_f > 0) ? t_f : 0;
    int t_b = (t_f + 1) < (in_d - 1) ? (t_f + 1) : (in_d - 1);
    float idx_src_t = ratio_d * (j + 0.5) - 0.5;
    idx_src_t = (idx_src_t > 0) ? idx_src_t : 0;
    float d_f = align_flag ? idx_src_t - t_f : ratio_d * j - t_f;
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
                         : static_cast<int>(ratio_h * k);
    y_n = (y_n > 0) ? y_n : 0;
    int y_s = (y_n + 1) < (in_h - 1) ? (y_n + 1) : (in_h - 1);
    float idx_src_y = ratio_h * (k + 0.5) - 0.5;
    idx_src_y = (idx_src_y > 0) ? idx_src_y : 0;
    float d_n = align_flag ? idx_src_y - y_n : ratio_h * k - y_n;
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
                  : static_cast<int>(ratio_w * l);
    x_w = (x_w > 0) ? x_w : 0;
    int x_e = (x_w + 1) < (in_w - 1) ? (x_w + 1) : (in_w - 1);
    float idx_src_x = ratio_w * (l + 0.5) - 0.5;
    idx_src_x = (idx_src_x > 0) ? idx_src_x : 0;
    float d_w = align_flag ? idx_src_x - x_w : ratio_w * l - x_w;
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
            if (data_layout == DataLayout::kNCHW) {
              T out_t = input_t(b, i, vt_f[j], vy_n[k], vx_w[l]) * vd_b[j] *
                            vd_s[k] * vd_e[l] +
                        input_t(b, i, vt_f[j], vy_n[k], vx_e[l]) * vd_b[j] *
                            vd_s[k] * vd_w[l] +
                        input_t(b, i, vt_f[j], vy_s[k], vx_w[l]) * vd_b[j] *
                            vd_n[k] * vd_e[l] +
                        input_t(b, i, vt_f[j], vy_s[k], vx_e[l]) * vd_b[j] *
                            vd_n[k] * vd_w[l] +
                        input_t(b, i, vt_b[j], vy_n[k], vx_w[l]) * vd_f[j] *
                            vd_s[k] * vd_e[l] +
                        input_t(b, i, vt_b[j], vy_n[k], vx_e[l]) * vd_f[j] *
                            vd_s[k] * vd_w[l] +
                        input_t(b, i, vt_b[j], vy_s[k], vx_w[l]) * vd_f[j] *
                            vd_n[k] * vd_e[l] +
                        input_t(b, i, vt_b[j], vy_s[k], vx_e[l]) * vd_f[j] *
                            vd_n[k] * vd_w[l];
              output_t(b, i, j, k, l) = out_t;
            } else {
              T out_t = input_t(b, vt_f[j], vy_n[k], vx_w[l], i) * vd_b[j] *
                            vd_s[k] * vd_e[l] +
                        input_t(b, vt_f[j], vy_n[k], vx_e[l], i) * vd_b[j] *
                            vd_s[k] * vd_w[l] +
                        input_t(b, vt_f[j], vy_s[k], vx_w[l], i) * vd_b[j] *
                            vd_n[k] * vd_e[l] +
                        input_t(b, vt_f[j], vy_s[k], vx_e[l], i) * vd_b[j] *
                            vd_n[k] * vd_w[l] +
                        input_t(b, vt_b[j], vy_n[k], vx_w[l], i) * vd_f[j] *
                            vd_s[k] * vd_e[l] +
                        input_t(b, vt_b[j], vy_n[k], vx_e[l], i) * vd_f[j] *
                            vd_s[k] * vd_w[l] +
                        input_t(b, vt_b[j], vy_s[k], vx_w[l], i) * vd_f[j] *
                            vd_n[k] * vd_e[l] +
                        input_t(b, vt_b[j], vy_s[k], vx_e[l], i) * vd_f[j] *
                            vd_n[k] * vd_w[l];
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
    int in_d = (align_corners) ? static_cast<int>(ratio_d * d + 0.5)
                               : static_cast<int>(ratio_d * d);
    for (int k = 0; k < out_h; k++) {
      int in_k = (align_corners) ? static_cast<int>(ratio_h * k + 0.5)
                                 : static_cast<int>(ratio_h * k);

      for (int l = 0; l < out_w; l++) {
        int in_l = (align_corners) ? static_cast<int>(ratio_w * l + 0.5)
                                   : static_cast<int>(ratio_w * l);

        for (int i = 0; i < n; i++) {    // loop for batches
          for (int j = 0; j < c; j++) {  // loop for channels
            if (data_layout == DataLayout::kNCHW) {
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
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,
    paddle::optional<const DenseTensor&> scale_tensor,
    const std::string& data_layout_str,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  const DataLayout data_layout =
      paddle::framework::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  funcs::ExtractNCDWH(x.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  float scale_w = -1.;
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
  phi::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {n, c, out_w};
  } else {
    dim_out = {n, out_w, c};
  }
  output->Resize(dim_out);
  dev_ctx.template Alloc<T>(output);

  if (in_w == out_w) {
    paddle::framework::TensorCopy(x, dev_ctx.GetPlace(), output);
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
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,
    paddle::optional<const DenseTensor&> scale_tensor,
    const std::string& data_layout_str,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  const DataLayout data_layout =
      paddle::framework::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  funcs::ExtractNCDWH(x.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  float scale_h = -1;
  float scale_w = -1;

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
  phi::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {n, c, out_h, out_w};
  } else {
    dim_out = {n, out_h, out_w, c};
  }
  output->Resize(dim_out);
  dev_ctx.template Alloc<T>(output);

  if (in_h == out_h && in_w == out_w) {
    paddle::framework::TensorCopy(x, dev_ctx.GetPlace(), output);
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
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,
    paddle::optional<const DenseTensor&> scale_tensor,
    const std::string& data_layout_str,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  const DataLayout data_layout =
      paddle::framework::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  funcs::ExtractNCDWH(x.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  float scale_d = -1;
  float scale_h = -1;
  float scale_w = -1;

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
      out_d = static_cast<int>(in_d * scale_d);
      out_h = static_cast<int>(in_h * scale_h);
      out_w = static_cast<int>(in_w * scale_w);
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

  phi::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {n, c, out_d, out_h, out_w};
  } else {
    dim_out = {n, out_d, out_h, out_w, c};
  }

  output->Resize(dim_out);
  dev_ctx.template Alloc<T>(output);

  if (in_d == out_d && in_h == out_h && in_w == out_w) {
    paddle::framework::TensorCopy(x, dev_ctx.GetPlace(), output);
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
    const Context& ctx,
    const DenseTensor& x,
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,
    paddle::optional<const DenseTensor&> scale_tensor,
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
    Interpolate1DCPUFwd<T, Context>(ctx,
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
    Interpolate2DCPUFwd<T>(ctx,
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
    Interpolate3DCPUFwd<T>(ctx,
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
    const Context& ctx,
    const DenseTensor& x,
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,
    paddle::optional<const DenseTensor&> scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  InterpolateKernel<T, Context>(ctx,
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
    const Context& ctx,
    const DenseTensor& x,
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,
    paddle::optional<const DenseTensor&> scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  InterpolateKernel<T, Context>(ctx,
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
    const Context& ctx,
    const DenseTensor& x,
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,
    paddle::optional<const DenseTensor&> scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  InterpolateKernel<T, Context>(ctx,
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
    const Context& ctx,
    const DenseTensor& x,
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,
    paddle::optional<const DenseTensor&> scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  InterpolateKernel<T, Context>(ctx,
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
    const Context& ctx,
    const DenseTensor& x,
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,
    paddle::optional<const DenseTensor&> scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  InterpolateKernel<T, Context>(ctx,
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

PD_REGISTER_KERNEL(bilinear_interp_v2,
                   CPU,
                   ALL_LAYOUT,
                   phi::BilinearInterpKernel,
                   float,
                   double,
                   uint8_t) {}
PD_REGISTER_KERNEL(nearest_interp_v2,
                   CPU,
                   ALL_LAYOUT,
                   phi::NearestInterpKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   uint8_t) {}
PD_REGISTER_KERNEL(trilinear_interp_v2,
                   CPU,
                   ALL_LAYOUT,
                   phi::TrilinearInterpKernel,
                   float,
                   double,
                   uint8_t) {}
PD_REGISTER_KERNEL(linear_interp_v2,
                   CPU,
                   ALL_LAYOUT,
                   phi::LinearInterpKernel,
                   float,
                   double,
                   uint8_t) {}
PD_REGISTER_KERNEL(bicubic_interp_v2,
                   CPU,
                   ALL_LAYOUT,
                   phi::BicubicInterpKernel,
                   float,
                   double) {}
