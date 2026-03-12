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
#include <array>
#include <cmath>
#include <type_traits>
#include <vector>

#include "paddle/common/layout.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/interpolate_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T>
static void LinearInterpolationGrad(const DenseTensor& output_grad,
                                    DenseTensor* input_grad,
                                    const float ratio_w,
                                    const int64_t in_w,
                                    const int64_t n,
                                    const int64_t c,
                                    const int out_w,
                                    const bool align_corners,
                                    const int align_mode,
                                    const DataLayout data_layout) {
  auto input_grad_t = EigenTensor<T, 3>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 3>::From(output_grad);
  bool align_flag = (align_mode == 0 && !align_corners);
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  for (int l = 0; l < out_w; l++) {
    int x_w = static_cast<int>(align_flag ? (ratio_w * (l + 0.5) - 0.5)
                                          : (ratio_w * static_cast<float>(l)));
    x_w = (x_w > 0) ? x_w : 0;                       // w
    int x_e = (x_w < (in_w - 1)) ? (x_w + 1) : x_w;  // w_id

    float idx_src_x = ratio_w * (static_cast<float>(l) + 0.5f) - 0.5f;
    idx_src_x = (idx_src_x > 0) ? idx_src_x : 0;
    float d_w = static_cast<float>(
        align_flag ? idx_src_x - static_cast<float>(x_w)
                   : ratio_w * static_cast<float>(l) -
                         static_cast<float>(x_w));  // w1lambda
    float d_e = 1.f - d_w;                          // w2lambda

    for (int i = 0; i < n; i++) {    // loop for batches
      for (int j = 0; j < c; j++) {  // loop for channels
        // linear interpolation grad
        if (data_layout == DataLayout::NCHW) {
          const MT grad = static_cast<MT>(output_grad_t(i, j, l));
          input_grad_t(i, j, x_w) += static_cast<T>(grad * d_e);
          input_grad_t(i, j, x_e) += static_cast<T>(grad * d_w);
        } else {
          const MT grad = static_cast<MT>(output_grad_t(i, l, j));
          input_grad_t(i, x_w, j) += static_cast<T>(grad * d_e);
          input_grad_t(i, x_e, j) += static_cast<T>(grad * d_w);
        }
      }
    }
  }
}

template <typename T>
static void BilinearInterpolationGrad(const DenseTensor& output_grad,
                                      DenseTensor* input_grad,
                                      const float ratio_h,
                                      const float ratio_w,
                                      const int64_t in_h,
                                      const int64_t in_w,
                                      const int64_t n,
                                      const int64_t c,
                                      const int out_h,
                                      const int out_w,
                                      const bool align_corners,
                                      const int align_mode,
                                      const DataLayout data_layout) {
  auto input_grad_t = EigenTensor<T, 4>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 4>::From(output_grad);
  bool align_flag = (align_mode == 0 && !align_corners);

  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

  for (int k = 0; k < out_h; k++) {  // loop for images
    int y_n = static_cast<int>(align_flag ? (ratio_h * (k + 0.5) - 0.5)
                                          : (ratio_h * static_cast<float>(k)));
    y_n = (y_n > 0) ? y_n : 0;
    int y_s = (y_n + 1) < (in_h - 1) ? (y_n + 1) : (in_h - 1);
    float idx_src_y = ratio_h * (static_cast<float>(k) + 0.5f) - 0.5f;
    idx_src_y = (idx_src_y > 0) ? idx_src_y : 0;
    float d_n = align_flag
                    ? idx_src_y - static_cast<float>(y_n)
                    : ratio_h * static_cast<float>(k) - static_cast<float>(y_n);
    float d_s = 1.f - d_n;

    for (int l = 0; l < out_w; l++) {
      int x_w = static_cast<int>(
          align_flag ? (ratio_w * (static_cast<float>(l) + 0.5f) - 0.5f)
                     : (ratio_w * static_cast<float>(l)));
      x_w = (x_w > 0) ? x_w : 0;
      int x_e = (x_w + 1) < (in_w - 1) ? (x_w + 1) : (in_w - 1);
      float idx_src_x = ratio_w * (static_cast<float>(l) + 0.5f) - 0.5f;
      idx_src_x = (idx_src_x > 0) ? idx_src_x : 0;
      float d_w = align_flag ? idx_src_x - static_cast<float>(x_w)
                             : ratio_w * static_cast<float>(l) -
                                   static_cast<float>(x_w);
      float d_e = 1.f - d_w;

      for (int i = 0; i < n; i++) {    // loop for batches
        for (int j = 0; j < c; j++) {  // loop for channels
          // bilinear interpolation grad
          if (data_layout == DataLayout::NCHW) {
            const MT grad = static_cast<MT>(output_grad_t(i, j, k, l));
            input_grad_t(i, j, y_n, x_w) += static_cast<T>(grad * d_s * d_e);
            input_grad_t(i, j, y_s, x_w) += static_cast<T>(grad * d_n * d_e);
            input_grad_t(i, j, y_n, x_e) += static_cast<T>(grad * d_s * d_w);
            input_grad_t(i, j, y_s, x_e) += static_cast<T>(grad * d_n * d_w);
          } else {
            const MT grad = static_cast<MT>(output_grad_t(i, k, l, j));
            input_grad_t(i, y_n, x_w, j) += static_cast<T>(grad * d_s * d_e);
            input_grad_t(i, y_s, x_w, j) += static_cast<T>(grad * d_n * d_e);
            input_grad_t(i, y_n, x_e, j) += static_cast<T>(grad * d_s * d_w);
            input_grad_t(i, y_s, x_e, j) += static_cast<T>(grad * d_n * d_w);
          }
        }
      }
    }
  }
}

template <typename T>
static void NearestNeighborInterpolateGrad(const DenseTensor& output_grad,
                                           DenseTensor* input_grad,
                                           const float ratio_h,
                                           const float ratio_w,
                                           const int64_t n,
                                           const int64_t c,
                                           const int out_h,
                                           const int out_w,
                                           const bool align_corners,
                                           const DataLayout data_layout) {
  auto input_grad_t = EigenTensor<T, 4>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 4>::From(output_grad);

  for (int k = 0; k < out_h; k++) {  // loop for images
    int in_k = static_cast<int>(align_corners
                                    ? (ratio_h * static_cast<float>(k) + 0.5f)
                                    : (ratio_h * static_cast<float>(k)));

    for (int l = 0; l < out_w; l++) {
      int in_l = static_cast<int>(align_corners
                                      ? (ratio_w * static_cast<float>(l) + 0.5f)
                                      : (ratio_w * static_cast<float>(l)));

      for (int i = 0; i < n; i++) {    // loop for batches
        for (int j = 0; j < c; j++) {  // loop for channels
          if (data_layout == DataLayout::NCHW) {
            input_grad_t(i, j, in_k, in_l) += output_grad_t(i, j, k, l);
          } else {
            input_grad_t(i, in_k, in_l, j) += output_grad_t(i, k, l, j);
          }
        }
      }
    }
  }
}

template <typename T>
static void BicubicInterpolationGrad(const DenseTensor& output_grad,
                                     DenseTensor* input_grad,
                                     const float ratio_h,
                                     const float ratio_w,
                                     const int64_t in_h,
                                     const int64_t in_w,
                                     const int64_t n,
                                     const int64_t c,
                                     const int out_h,
                                     const int out_w,
                                     const bool align_corners,
                                     const DataLayout data_layout) {
  auto input_grad_t = EigenTensor<T, 4>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 4>::From(output_grad);
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

  for (int k = 0; k < out_h; k++) {  // loop for images
    MT y_n = align_corners ? ratio_h * static_cast<float>(k)
                           : ratio_h * (static_cast<float>(k) + 0.5f) - 0.5f;
    int64_t input_y = floorf(y_n);
    MT y_t = y_n - input_y;

    for (int l = 0; l < out_w; l++) {
      MT x_n = align_corners ? ratio_w * static_cast<float>(l)
                             : ratio_w * (static_cast<float>(l) + 0.5f) - 0.5f;
      int64_t input_x = floorf(x_n);
      MT x_t = x_n - input_x;

      std::array<MT, 4> x_coeffs;
      std::array<MT, 4> y_coeffs;

      funcs::GetCubicUpsampleCoefficients<MT>(x_coeffs.data(), x_t);
      funcs::GetCubicUpsampleCoefficients<MT>(y_coeffs.data(), y_t);

      for (int i = 0; i < n; i++) {    // loop for batches
        for (int j = 0; j < c; j++) {  // loop for channels
          // bicubic interpolation grad
          for (int ii = 0; ii < 4; ii++) {
            for (int jj = 0; jj < 4; jj++) {
              int access_x = std::max(std::min(input_x - 1 + ii, in_w - 1),
                                      static_cast<int64_t>(0));
              int access_y = std::max(std::min(input_y - 1 + jj, in_h - 1),
                                      static_cast<int64_t>(0));
              if (data_layout == DataLayout::NCHW) {
                MT grad = static_cast<MT>(output_grad_t(i, j, k, l));
                input_grad_t(i, j, access_y, access_x) +=
                    static_cast<T>(grad * y_coeffs[jj] * x_coeffs[ii]);
              } else {
                MT grad = static_cast<MT>(output_grad_t(i, k, l, j));
                input_grad_t(i, access_y, access_x, j) +=
                    static_cast<T>(grad * y_coeffs[jj] * x_coeffs[ii]);
              }
            }
          }
        }
      }
    }
  }
}

template <typename T>
static void TrilinearInterpolationGrad(const DenseTensor& output_grad,
                                       DenseTensor* input_grad,
                                       const float ratio_d,
                                       const float ratio_h,
                                       const float ratio_w,
                                       const int64_t in_d,
                                       const int64_t in_h,
                                       const int64_t in_w,
                                       const int64_t n,
                                       const int64_t c,
                                       const int out_d,
                                       const int out_h,
                                       const int out_w,
                                       const bool align_corners,
                                       const int align_mode,
                                       const DataLayout data_layout) {
  auto input_grad_t = EigenTensor<T, 5>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 5>::From(output_grad);
  bool align_flag = (align_mode == 0 && !align_corners);
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  for (int j = 0; j < out_d; j++) {  // loop for D
    int t_f = static_cast<int>(
        align_flag ? (ratio_d * (static_cast<float>(j) + 0.5f) - 0.5f)
                   : (ratio_d * static_cast<float>(j)));
    t_f = (t_f > 0) ? t_f : 0;
    int t_b = (t_f + 1) < (in_d - 1) ? (t_f + 1) : (in_d - 1);
    float idx_src_t = ratio_d * (static_cast<float>(j) + 0.5f) - 0.5f;
    idx_src_t = (idx_src_t > 0) ? idx_src_t : 0;
    float d_f = align_flag
                    ? idx_src_t - static_cast<float>(t_f)
                    : ratio_d * static_cast<float>(j) - static_cast<float>(t_f);
    float d_b = 1.f - d_f;

    for (int k = 0; k < out_h; k++) {  // loop for H
      int y_n = static_cast<int>(
          align_flag ? (ratio_h * (static_cast<float>(k) + 0.5f) - 0.5f)
                     : (ratio_h * static_cast<float>(k)));
      y_n = (y_n > 0) ? y_n : 0;
      int y_s = (y_n + 1) < (in_h - 1) ? (y_n + 1) : (in_h - 1);
      float idx_src_y = ratio_h * (static_cast<float>(k) + 0.5f) - 0.5f;
      idx_src_y = (idx_src_y > 0) ? idx_src_y : 0;
      float d_n = align_flag ? idx_src_y - static_cast<float>(y_n)
                             : ratio_h * static_cast<float>(k) -
                                   static_cast<float>(y_n);
      float d_s = 1.f - d_n;

      for (int l = 0; l < out_w; l++) {  // loop for W
        int x_w = static_cast<int>(
            align_flag ? (ratio_w * (static_cast<float>(l) + 0.5f) - 0.5f)
                       : (ratio_w * static_cast<float>(l)));
        x_w = (x_w > 0) ? x_w : 0;
        int x_e = (x_w + 1) < (in_w - 1) ? (x_w + 1) : (in_w - 1);
        float idx_src_x = ratio_w * (static_cast<float>(l) + 0.5f) - 0.5f;
        idx_src_x = (idx_src_x > 0) ? idx_src_x : 0;
        float d_w = align_flag ? idx_src_x - static_cast<float>(x_w)
                               : ratio_w * static_cast<float>(l) -
                                     static_cast<float>(x_w);
        float d_e = 1.f - d_w;

        for (int b = 0; b < n; b++) {    // loop for batches
          for (int i = 0; i < c; i++) {  // loop for channels
            // trilinear interpolation grad
            if (data_layout == DataLayout::NCHW) {
              const MT grad = static_cast<MT>(output_grad_t(b, i, j, k, l));
              input_grad_t(b, i, t_f, y_n, x_w) +=
                  static_cast<T>(grad * d_b * d_s * d_e);
              input_grad_t(b, i, t_f, y_n, x_e) +=
                  static_cast<T>(grad * d_b * d_s * d_w);
              input_grad_t(b, i, t_f, y_s, x_w) +=
                  static_cast<T>(grad * d_b * d_n * d_e);
              input_grad_t(b, i, t_f, y_s, x_e) +=
                  static_cast<T>(grad * d_b * d_n * d_w);
              input_grad_t(b, i, t_b, y_n, x_w) +=
                  static_cast<T>(grad * d_f * d_s * d_e);
              input_grad_t(b, i, t_b, y_n, x_e) +=
                  static_cast<T>(grad * d_f * d_s * d_w);
              input_grad_t(b, i, t_b, y_s, x_w) +=
                  static_cast<T>(grad * d_f * d_n * d_e);
              input_grad_t(b, i, t_b, y_s, x_e) +=
                  static_cast<T>(grad * d_f * d_n * d_w);
            } else {
              const MT grad = static_cast<MT>(output_grad_t(b, j, k, l, i));
              input_grad_t(b, t_f, y_n, x_w, i) +=
                  static_cast<T>(grad * d_b * d_s * d_e);
              input_grad_t(b, t_f, y_n, x_e, i) +=
                  static_cast<T>(grad * d_b * d_s * d_w);
              input_grad_t(b, t_f, y_s, x_w, i) +=
                  static_cast<T>(grad * d_b * d_n * d_e);
              input_grad_t(b, t_f, y_s, x_e, i) +=
                  static_cast<T>(grad * d_b * d_n * d_w);
              input_grad_t(b, t_b, y_n, x_w, i) +=
                  static_cast<T>(grad * d_f * d_s * d_e);
              input_grad_t(b, t_b, y_n, x_e, i) +=
                  static_cast<T>(grad * d_f * d_s * d_w);
              input_grad_t(b, t_b, y_s, x_w, i) +=
                  static_cast<T>(grad * d_f * d_n * d_e);
              input_grad_t(b, t_b, y_s, x_e, i) +=
                  static_cast<T>(grad * d_f * d_n * d_w);
            }
          }
        }
      }
    }
  }
}

template <typename T>
static void NearestNeighbor3DInterpolateGrad(const DenseTensor& output_grad,
                                             DenseTensor* input_grad,
                                             const float ratio_d,
                                             const float ratio_h,
                                             const float ratio_w,
                                             const int64_t n,
                                             const int64_t c,
                                             const int out_d,
                                             const int out_h,
                                             const int out_w,
                                             const bool align_corners,
                                             const DataLayout data_layout) {
  auto input_grad_t = EigenTensor<T, 5>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 5>::From(output_grad);

  for (int d = 0; d < out_d; d++) {
    int in_d = static_cast<int>(
        align_corners
            ? static_cast<float>(std::lround(ratio_d * static_cast<float>(d)))
            : (ratio_d * static_cast<float>(d)));
    for (int k = 0; k < out_h; k++) {  // loop for images
      int in_k = static_cast<int>(
          align_corners
              ? static_cast<float>(std::lround(ratio_h * static_cast<float>(k)))
              : (ratio_h * static_cast<float>(k)));

      for (int l = 0; l < out_w; l++) {
        int in_l = static_cast<int>(align_corners
                                        ? static_cast<float>(std::lround(
                                              ratio_w * static_cast<float>(l)))
                                        : (ratio_w * static_cast<float>(l)));

        for (int i = 0; i < n; i++) {    // loop for batches
          for (int j = 0; j < c; j++) {  // loop for channels
            if (data_layout == DataLayout::NCHW) {
              input_grad_t(i, j, in_d, in_k, in_l) +=
                  output_grad_t(i, j, d, k, l);
            } else {
              input_grad_t(i, in_d, in_k, in_l, j) +=
                  output_grad_t(i, d, k, l, j);
            }
          }
        }
      }
    }
  }
}

template <typename T, typename Context>
static void Interpolate1DCPUBwd(
    const Context& dev_ctx,
    const DenseTensor& input,
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const DenseTensor& output_grad,
    const std::string& data_layout_str,
    int out_w,
    const std::vector<double>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* input_grad) {
  const DataLayout data_layout = StringToDataLayout(data_layout_str);
  int64_t n = 0, c = 0, in_d = 0, in_h = 0, in_w = 0;
  funcs::ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  double scale_w = -1.0;
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
    out_w = static_cast<int>(static_cast<float>(in_w) * scale_w);
  }
  if (out_size) {
    auto out_size_data =
        funcs::get_new_data_from_tensor<int>(out_size.get_ptr());
    out_w = out_size_data[0];
  }
  if (size_tensor && !size_tensor->empty()) {
    // have size tensor
    auto new_size = funcs::get_new_shape(size_tensor.get());
    out_w = new_size[0];
  }

  DDim dim_grad;
  if (data_layout == DataLayout::NCHW) {
    dim_grad = {n, c, in_w};
  } else {
    dim_grad = {n, in_w, c};
  }

  input_grad->Resize(dim_grad);
  dev_ctx.template Alloc<T>(input_grad);

  funcs::SetConstant<Context, T> zero;
  zero(dev_ctx, input_grad, static_cast<T>(0.0));

  if (in_w == out_w) {
    Copy(dev_ctx, output_grad, dev_ctx.GetPlace(), false, input_grad);
    return;
  }

  float ratio_w = 0.f;
  if (out_w > 1) {
    float new_scale_w = 0.f;
    new_scale_w = static_cast<float>(
        scale_w > 0 ? (1.f / scale_w)
                    : static_cast<float>(in_w) / static_cast<float>(out_w));
    ratio_w =
        static_cast<float>(align_corners ? (static_cast<float>(in_w) - 1.f) /
                                               (static_cast<float>(out_w) - 1.f)
                                         : new_scale_w);
  }
  if ("linear" == interp_method) {
    LinearInterpolationGrad<T>(output_grad,
                               input_grad,
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
static void Interpolate2DCPUBwd(
    const Context& dev_ctx,
    const DenseTensor& input,
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const DenseTensor& output_grad,
    const std::string& data_layout_str,
    int out_h,
    int out_w,
    const std::vector<double>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* input_grad) {
  const DataLayout data_layout = StringToDataLayout(data_layout_str);
  int64_t n = 0, c = 0, in_d = 0, in_h = 0, in_w = 0;
  funcs::ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  double scale_h = -1;
  double scale_w = -1;
  if (scale_tensor) {
    auto scale_data =
        funcs::get_new_data_from_tensor<float>(scale_tensor.get_ptr());
    if (scale_data.size() > 1) {
      scale_h = scale_data[0];
      scale_w = scale_data[1];
    } else {
      scale_w = scale_data[0];
      scale_h = scale_data[0];
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
  if (size_tensor && !size_tensor->empty()) {
    // have size tensor
    auto new_size = funcs::get_new_shape(size_tensor.get());
    out_h = new_size[0];
    out_w = new_size[1];
  }

  DDim dim_grad;
  if (data_layout == DataLayout::NCHW) {
    dim_grad = {n, c, in_h, in_w};
  } else {
    dim_grad = {n, in_h, in_w, c};
  }

  input_grad->Resize(dim_grad);
  dev_ctx.template Alloc<T>(input_grad);

  funcs::SetConstant<Context, T> zero;
  zero(dev_ctx, input_grad, static_cast<T>(0.0));

  if (in_h == out_h && in_w == out_w) {
    Copy(dev_ctx, output_grad, dev_ctx.GetPlace(), false, input_grad);
    return;
  }

  double ratio_h =
      funcs::AreaPixelComputeScale<float>(in_h, out_h, align_corners, scale_h);
  double ratio_w =
      funcs::AreaPixelComputeScale<float>(in_w, out_w, align_corners, scale_w);

  // TODO(zrr1999): to align xpu
  if (out_h <= 1) {
    ratio_h = 0;
  }
  if (out_w <= 1) {
    ratio_w = 0;
  }

  if ("bilinear" == interp_method) {
    BilinearInterpolationGrad<T>(output_grad,
                                 input_grad,
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
    NearestNeighborInterpolateGrad<T>(output_grad,
                                      input_grad,
                                      ratio_h,
                                      ratio_w,
                                      n,
                                      c,
                                      out_h,
                                      out_w,
                                      align_corners,
                                      data_layout);
  } else if ("bicubic" == interp_method) {
    BicubicInterpolationGrad<T>(output_grad,
                                input_grad,
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
static void Interpolate3DCPUBwd(
    const Context& dev_ctx,
    const DenseTensor& input,
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const DenseTensor& output_grad,
    const std::string& data_layout_str,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<double>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* input_grad) {
  const DataLayout data_layout = StringToDataLayout(data_layout_str);
  int64_t n = 0, c = 0, in_d = 0, in_h = 0, in_w = 0;
  funcs::ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  double scale_d = -1;
  double scale_h = -1;
  double scale_w = -1;
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
  if (scale_d > 0. && scale_h > 0. && scale_w > 0.) {
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
  if (size_tensor && !size_tensor->empty()) {
    // have size tensor
    auto new_size = funcs::get_new_shape(size_tensor.get());
    out_d = new_size[0];
    out_h = new_size[1];
    out_w = new_size[2];
  }

  DDim dim_grad;
  if (data_layout == DataLayout::NCHW) {
    dim_grad = {n, c, in_d, in_h, in_w};
  } else {
    dim_grad = {n, in_d, in_h, in_w, c};
  }
  input_grad->Resize(dim_grad);
  dev_ctx.template Alloc<T>(input_grad);

  funcs::SetConstant<Context, T> zero;
  zero(dev_ctx, input_grad, static_cast<T>(0.0));

  if (in_d == out_d && in_h == out_h && in_w == out_w) {
    Copy(dev_ctx, output_grad, dev_ctx.GetPlace(), false, input_grad);
    return;
  }

  double ratio_d =
      funcs::AreaPixelComputeScale<float>(in_d, out_d, align_corners, scale_d);
  double ratio_h =
      funcs::AreaPixelComputeScale<float>(in_h, out_h, align_corners, scale_h);
  double ratio_w =
      funcs::AreaPixelComputeScale<float>(in_w, out_w, align_corners, scale_w);

  if ("trilinear" == interp_method) {
    TrilinearInterpolationGrad<T>(output_grad,
                                  input_grad,
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
    NearestNeighbor3DInterpolateGrad<T>(output_grad,
                                        input_grad,
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
void InterpolateGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const DenseTensor& output_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<double>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* x_grad) {
  if (x_grad && x_grad->numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    return;
  }
  auto output_grad_dims = output_grad.dims();
  if (output_grad_dims.size() == 3) {  // 1D interpolation grad
    Interpolate1DCPUBwd<T, Context>(dev_ctx,
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
    Interpolate2DCPUBwd<T, Context>(dev_ctx,
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
    Interpolate3DCPUBwd<T, Context>(dev_ctx,
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
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<double>& scale,
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
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
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
  std::vector<double> scale_vec;
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
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<double>& scale,
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
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
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
  std::vector<double> scale_vec;
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
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<double>& scale,
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
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<double>& scale,
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
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<double>& scale,
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

// CPU weight computation for antialias interpolation (backward uses same
// weights).
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

// =====================================================================
// CPU Antialias Interpolation Backward Implementation
// The backward pass of separable 2-pass AA interpolation.
// For the forward: output = W_v * W_h * input (separable)
// The backward: input_grad += W_h^T * W_v^T * output_grad
// Since it's separable, we reverse the passes:
//   Pass 1 (vertical backward): grad_output [N,C,H_out,W_out] -> temp
//   [N,C,H_in,W_out] Pass 2 (horizontal backward): temp [N,C,H_in,W_out] ->
//   input_grad [N,C,H_in,W_in]
// =====================================================================

// Backward pass for float types, NCHW layout.
template <typename T, typename InterpFilter>
static void AAInterpolation2DGradCPU_NCHW(const T* output_grad_data,
                                          T* input_grad_data,
                                          int64_t n,
                                          int64_t c,
                                          int in_h,
                                          int in_w,
                                          int out_h,
                                          int out_w,
                                          float ratio_h,
                                          float ratio_w,
                                          const InterpFilter& filter) {
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

  struct SpanInfo {
    int xmin;
    int xsize;
    WT center;
  };

  // Pre-compute horizontal weights
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

  // Temporary buffer for intermediate gradient [N, C, H_in, W_out]
  std::vector<T> temp_grad(static_cast<size_t>(n) * c * in_h * out_w,
                           static_cast<T>(0));

  // Backward Pass 1: Vertical backward (transpose of vertical forward)
  // Forward was: output[oh] = sum_j(temp[ymin+j] * wy[j])
  // Backward: temp_grad[ymin+j] += output_grad[oh] * wy[j]
  for (int64_t nc_idx = 0; nc_idx < n * c; nc_idx++) {
    for (int oh = 0; oh < out_h; oh++) {
      int ymin = v_spans[oh].xmin;
      int ysize = v_spans[oh].xsize;
      const WT* wts = v_weights[oh].data();

      for (int ow = 0; ow < out_w; ow++) {
        WT grad_val = static_cast<WT>(
            output_grad_data[nc_idx * out_h * out_w + oh * out_w + ow]);

        for (int j = 0; j < ysize; j++) {
          T* temp_ptr = temp_grad.data() + nc_idx * in_h * out_w +
                        (ymin + j) * out_w + ow;
          *temp_ptr =
              static_cast<T>(static_cast<WT>(*temp_ptr) + grad_val * wts[j]);
        }
      }
    }
  }

  // Backward Pass 2: Horizontal backward (transpose of horizontal forward)
  // Forward was: temp[ow] = sum_j(input[xmin+j] * wx[j])
  // Backward: input_grad[xmin+j] += temp_grad[ow] * wx[j]
  for (int64_t nc_idx = 0; nc_idx < n * c; nc_idx++) {
    for (int ih = 0; ih < in_h; ih++) {
      for (int ow = 0; ow < out_w; ow++) {
        int xmin = h_spans[ow].xmin;
        int xsize = h_spans[ow].xsize;
        const WT* wts = h_weights[ow].data();

        WT grad_val =
            static_cast<WT>(temp_grad[nc_idx * in_h * out_w + ih * out_w + ow]);

        for (int j = 0; j < xsize; j++) {
          T* ig_ptr =
              input_grad_data + nc_idx * in_h * in_w + ih * in_w + (xmin + j);
          *ig_ptr =
              static_cast<T>(static_cast<WT>(*ig_ptr) + grad_val * wts[j]);
        }
      }
    }
  }
}

// Backward pass for float types, NHWC layout.
template <typename T, typename InterpFilter>
static void AAInterpolation2DGradCPU_NHWC(const T* output_grad_data,
                                          T* input_grad_data,
                                          int64_t n,
                                          int64_t c,
                                          int in_h,
                                          int in_w,
                                          int out_h,
                                          int out_w,
                                          float ratio_h,
                                          float ratio_w,
                                          const InterpFilter& filter) {
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

  struct SpanInfo {
    int xmin;
    int xsize;
    WT center;
  };

  // Pre-compute horizontal weights
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

  // Temporary buffer [N, H_in, W_out, C]
  std::vector<T> temp_grad(static_cast<size_t>(n) * in_h * out_w * c,
                           static_cast<T>(0));

  // Backward Pass 1: Vertical backward
  for (int64_t bi = 0; bi < n; bi++) {
    for (int oh = 0; oh < out_h; oh++) {
      int ymin = v_spans[oh].xmin;
      int ysize = v_spans[oh].xsize;
      const WT* wts = v_weights[oh].data();

      for (int ow = 0; ow < out_w; ow++) {
        for (int64_t ch = 0; ch < c; ch++) {
          int64_t og_idx = ((bi * out_h + oh) * out_w + ow) * c + ch;
          WT grad_val = static_cast<WT>(output_grad_data[og_idx]);

          for (int j = 0; j < ysize; j++) {
            int64_t temp_idx = ((bi * in_h + (ymin + j)) * out_w + ow) * c + ch;
            temp_grad[temp_idx] = static_cast<T>(
                static_cast<WT>(temp_grad[temp_idx]) + grad_val * wts[j]);
          }
        }
      }
    }
  }

  // Backward Pass 2: Horizontal backward
  for (int64_t bi = 0; bi < n; bi++) {
    for (int ih = 0; ih < in_h; ih++) {
      for (int ow = 0; ow < out_w; ow++) {
        int xmin = h_spans[ow].xmin;
        int xsize = h_spans[ow].xsize;
        const WT* wts = h_weights[ow].data();

        for (int64_t ch = 0; ch < c; ch++) {
          int64_t temp_idx = ((bi * in_h + ih) * out_w + ow) * c + ch;
          WT grad_val = static_cast<WT>(temp_grad[temp_idx]);

          for (int j = 0; j < xsize; j++) {
            int64_t ig_idx = ((bi * in_h + ih) * in_w + (xmin + j)) * c + ch;
            input_grad_data[ig_idx] = static_cast<T>(
                static_cast<WT>(input_grad_data[ig_idx]) + grad_val * wts[j]);
          }
        }
      }
    }
  }
}

// Dispatcher for backward
template <typename T, typename InterpFilter>
static void AAInterpolation2DGradCPUDispatch(const T* output_grad_data,
                                             T* input_grad_data,
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
    AAInterpolation2DGradCPU_NCHW<T>(output_grad_data,
                                     input_grad_data,
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
    AAInterpolation2DGradCPU_NHWC<T>(output_grad_data,
                                     input_grad_data,
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

// Main CPU backward function for AA 2D interpolation.
template <typename T, typename Context>
static void InterpolateAA2DCPUBwd(
    const Context& dev_ctx,
    const DenseTensor& input,
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const DenseTensor& output_grad,
    const std::string& data_layout_str,
    int out_h,
    int out_w,
    const std::vector<double>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* input_grad) {
  if (input_grad && input_grad->numel() == 0) {
    dev_ctx.template Alloc<T>(input_grad);
    return;
  }

  const DataLayout data_layout = StringToDataLayout(data_layout_str);
  int64_t n, c, in_d, in_h, in_w;
  funcs::ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  double scale_h = -1;
  double scale_w = -1;
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

  auto* output_grad_data = output_grad.data<T>();
  DDim dim_grad;
  if (data_layout == DataLayout::NCHW) {
    dim_grad = {n, c, in_h, in_w};
  } else {
    dim_grad = {n, in_h, in_w, c};
  }
  input_grad->Resize(dim_grad);
  auto* input_grad_data = dev_ctx.template Alloc<T>(input_grad);
  funcs::SetConstant<Context, T> zero;
  zero(dev_ctx, input_grad, static_cast<T>(0.0));

  if (in_h == out_h && in_w == out_w) {
    Copy(dev_ctx, output_grad, dev_ctx.GetPlace(), false, input_grad);
    return;
  }

  // Use conditional type matching GPU: float for integral/half types, double
  // for double
  using MT =
      typename std::conditional_t<std::is_integral<T>::value,
                                  float,
                                  typename phi::dtype::MPTypeTrait<T>::Type>;
  MT ratio_h =
      funcs::AreaPixelComputeScale<MT>(in_h, out_h, align_corners, scale_h);
  MT ratio_w =
      funcs::AreaPixelComputeScale<MT>(in_w, out_w, align_corners, scale_w);

  auto launch_aa_bw = [&](auto filter_functor) {
    AAInterpolation2DGradCPUDispatch<T>(output_grad_data,
                                        input_grad_data,
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
  };

  if ("bilinear" == interp_method) {
    launch_aa_bw(funcs::antialias::BilinearFilterFunctor{});
  } else if ("bicubic" == interp_method) {
    launch_aa_bw(funcs::antialias::BicubicFilterFunctor{});
  }
}

template <typename T, typename Context>
void InterpAntialiasGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const optional<DenseTensor>& out_size,
    const optional<std::vector<const DenseTensor*>>& size_tensor,
    const optional<DenseTensor>& scale_tensor,
    const DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<double>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* x_grad) {
  InterpolateAA2DCPUBwd<T, Context>(dev_ctx,
                                    x,
                                    out_size,
                                    size_tensor,
                                    scale_tensor,
                                    out_grad,
                                    data_layout,
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
                   CPU,
                   ALL_LAYOUT,
                   phi::BilinearInterpGradKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(legacy_bilinear_interp_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::LegacyBilinearInterpGradKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(nearest_interp_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::NearestInterpGradKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(legacy_nearest_interp_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::LegacyNearestInterpGradKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(trilinear_interp_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::TrilinearInterpGradKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(linear_interp_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::LinearInterpGradKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(bicubic_interp_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::BicubicInterpGradKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_KERNEL(interp_antialias_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::InterpAntialiasGradKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(1).SetBackend(phi::Backend::CPU);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
