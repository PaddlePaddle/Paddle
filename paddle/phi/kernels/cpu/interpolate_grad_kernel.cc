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
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/interpolate_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T>
static void LinearInterpolationGrad(const DenseTensor& output_grad,
                                    DenseTensor* input_grad,
                                    const float ratio_w,
                                    const int in_w,
                                    const int n,
                                    const int c,
                                    const int out_w,
                                    const bool align_corners,
                                    const int align_mode,
                                    const DataLayout data_layout) {
  auto input_grad_t = EigenTensor<T, 3>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 3>::From(output_grad);
  bool align_flag = (align_mode == 0 && !align_corners);
  for (int l = 0; l < out_w; l++) {
    int x_w = align_flag ? static_cast<int>(ratio_w * (l + 0.5) - 0.5)
                         : static_cast<int>(ratio_w * l);
    x_w = (x_w > 0) ? x_w : 0;                       // w
    int x_e = (x_w < (in_w - 1)) ? (x_w + 1) : x_w;  // w_id

    float idx_src_x = ratio_w * (l + 0.5) - 0.5;
    idx_src_x = (idx_src_x > 0) ? idx_src_x : 0;
    float d_w = align_flag ? idx_src_x - x_w : ratio_w * l - x_w;  // w1lambda
    float d_e = 1.f - d_w;                                         // w2lambda

    for (int i = 0; i < n; i++) {    // loop for batches
      for (int j = 0; j < c; j++) {  // loop for channels
        // linear interpolation grad
        if (data_layout == DataLayout::kNCHW) {
          const T grad = output_grad_t(i, j, l);
          input_grad_t(i, j, x_w) += static_cast<T>(grad * d_e);
          input_grad_t(i, j, x_e) += static_cast<T>(grad * d_w);
        } else {
          const T grad = output_grad_t(i, l, j);
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
                                      const int in_h,
                                      const int in_w,
                                      const int n,
                                      const int c,
                                      const int out_h,
                                      const int out_w,
                                      const bool align_corners,
                                      const int align_mode,
                                      const DataLayout data_layout) {
  auto input_grad_t = EigenTensor<T, 4>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 4>::From(output_grad);
  bool align_flag = (align_mode == 0 && !align_corners);
  for (int k = 0; k < out_h; k++) {  // loop for images
    int y_n = align_flag ? static_cast<int>(ratio_h * (k + 0.5) - 0.5)
                         : static_cast<int>(ratio_h * k);
    y_n = (y_n > 0) ? y_n : 0;
    int y_s = (y_n + 1) < (in_h - 1) ? (y_n + 1) : (in_h - 1);
    float idx_src_y = ratio_h * (k + 0.5) - 0.5;
    idx_src_y = (idx_src_y > 0) ? idx_src_y : 0;
    float d_n = align_flag ? idx_src_y - y_n : ratio_h * k - y_n;
    float d_s = 1.f - d_n;

    for (int l = 0; l < out_w; l++) {
      int x_w = align_flag ? static_cast<int>(ratio_w * (l + 0.5) - 0.5)
                           : static_cast<int>(ratio_w * l);
      x_w = (x_w > 0) ? x_w : 0;
      int x_e = (x_w + 1) < (in_w - 1) ? (x_w + 1) : (in_w - 1);
      float idx_src_x = ratio_w * (l + 0.5) - 0.5;
      idx_src_x = (idx_src_x > 0) ? idx_src_x : 0;
      float d_w = align_flag ? idx_src_x - x_w : ratio_w * l - x_w;
      float d_e = 1.f - d_w;

      for (int i = 0; i < n; i++) {    // loop for batches
        for (int j = 0; j < c; j++) {  // loop for channels
          // bilinear interpolation grad
          if (data_layout == DataLayout::kNCHW) {
            const T grad = output_grad_t(i, j, k, l);
            input_grad_t(i, j, y_n, x_w) += static_cast<T>(grad * d_s * d_e);
            input_grad_t(i, j, y_s, x_w) += static_cast<T>(grad * d_n * d_e);
            input_grad_t(i, j, y_n, x_e) += static_cast<T>(grad * d_s * d_w);
            input_grad_t(i, j, y_s, x_e) += static_cast<T>(grad * d_n * d_w);
          } else {
            const T grad = output_grad_t(i, k, l, j);
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
                                           const int n,
                                           const int c,
                                           const int out_h,
                                           const int out_w,
                                           const bool align_corners,
                                           const DataLayout data_layout) {
  auto input_grad_t = EigenTensor<T, 4>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 4>::From(output_grad);

  for (int k = 0; k < out_h; k++) {  // loop for images
    int in_k = (align_corners) ? static_cast<int>(ratio_h * k + 0.5)
                               : static_cast<int>(ratio_h * k);

    for (int l = 0; l < out_w; l++) {
      int in_l = (align_corners) ? static_cast<int>(ratio_w * l + 0.5)
                                 : static_cast<int>(ratio_w * l);

      for (int i = 0; i < n; i++) {    // loop for batches
        for (int j = 0; j < c; j++) {  // loop for channels
          if (data_layout == DataLayout::kNCHW) {
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
                                     const int in_h,
                                     const int in_w,
                                     const int n,
                                     const int c,
                                     const int out_h,
                                     const int out_w,
                                     const bool align_corners,
                                     const DataLayout data_layout) {
  auto input_grad_t = EigenTensor<T, 4>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 4>::From(output_grad);

  for (int k = 0; k < out_h; k++) {  // loop for images
    T y_n = align_corners ? static_cast<T>(ratio_h * k)
                          : static_cast<T>(ratio_h * (k + 0.5) - 0.5);
    int input_y = floorf(y_n);
    T y_t = y_n - input_y;

    for (int l = 0; l < out_w; l++) {
      T x_n = align_corners ? static_cast<T>(ratio_w * l)
                            : static_cast<T>(ratio_w * (l + 0.5) - 0.5);
      int input_x = floorf(x_n);
      T x_t = x_n - input_x;

      T x_coeffs[4];
      T y_coeffs[4];

      funcs::get_cubic_upsample_coefficients<T>(x_coeffs, x_t);
      funcs::get_cubic_upsample_coefficients<T>(y_coeffs, y_t);

      for (int i = 0; i < n; i++) {    // loop for batches
        for (int j = 0; j < c; j++) {  // loop for channels
          // bicubic interpolation grad
          for (int ii = 0; ii < 4; ii++) {
            for (int jj = 0; jj < 4; jj++) {
              int access_x = std::max(std::min(input_x - 1 + ii, in_w - 1),
                                      static_cast<int>(0));
              int access_y = std::max(std::min(input_y - 1 + jj, in_h - 1),
                                      static_cast<int>(0));
              if (data_layout == DataLayout::kNCHW) {
                T grad = output_grad_t(i, j, k, l);
                input_grad_t(i, j, access_y, access_x) +=
                    grad * y_coeffs[jj] * x_coeffs[ii];
              } else {
                T grad = output_grad_t(i, k, l, j);
                input_grad_t(i, access_y, access_x, j) +=
                    grad * y_coeffs[jj] * x_coeffs[ii];
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
                                       const DataLayout data_layout) {
  auto input_grad_t = EigenTensor<T, 5>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 5>::From(output_grad);
  bool align_flag = (align_mode == 0 && !align_corners);
  for (int j = 0; j < out_d; j++) {  // loop for D
    int t_f = align_flag ? static_cast<int>(ratio_d * (j + 0.5) - 0.5)
                         : static_cast<int>(ratio_d * j);
    t_f = (t_f > 0) ? t_f : 0;
    int t_b = (t_f + 1) < (in_d - 1) ? (t_f + 1) : (in_d - 1);
    float idx_src_t = ratio_d * (j + 0.5) - 0.5;
    idx_src_t = (idx_src_t > 0) ? idx_src_t : 0;
    float d_f = align_flag ? idx_src_t - t_f : ratio_d * j - t_f;
    float d_b = 1.f - d_f;

    for (int k = 0; k < out_h; k++) {  // loop for H
      int y_n = align_flag ? static_cast<int>(ratio_h * (k + 0.5) - 0.5)
                           : static_cast<int>(ratio_h * k);
      y_n = (y_n > 0) ? y_n : 0;
      int y_s = (y_n + 1) < (in_h - 1) ? (y_n + 1) : (in_h - 1);
      float idx_src_y = ratio_h * (k + 0.5) - 0.5;
      idx_src_y = (idx_src_y > 0) ? idx_src_y : 0;
      float d_n = align_flag ? idx_src_y - y_n : ratio_h * k - y_n;
      float d_s = 1.f - d_n;

      for (int l = 0; l < out_w; l++) {  // loop for W
        int x_w = align_flag ? static_cast<int>(ratio_w * (l + 0.5) - 0.5)
                             : static_cast<int>(ratio_w * l);
        x_w = (x_w > 0) ? x_w : 0;
        int x_e = (x_w + 1) < (in_w - 1) ? (x_w + 1) : (in_w - 1);
        float idx_src_x = ratio_w * (l + 0.5) - 0.5;
        idx_src_x = (idx_src_x > 0) ? idx_src_x : 0;
        float d_w = align_flag ? idx_src_x - x_w : ratio_w * l - x_w;
        float d_e = 1.f - d_w;

        for (int b = 0; b < n; b++) {    // loop for batches
          for (int i = 0; i < c; i++) {  // loop for channels
            // trilinear interpolation grad
            if (data_layout == DataLayout::kNCHW) {
              const T grad = output_grad_t(b, i, j, k, l);
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
              const T grad = output_grad_t(b, j, k, l, i);
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
                                             const int n,
                                             const int c,
                                             const int out_d,
                                             const int out_h,
                                             const int out_w,
                                             const bool align_corners,
                                             const DataLayout data_layout) {
  auto input_grad_t = EigenTensor<T, 5>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 5>::From(output_grad);

  for (int d = 0; d < out_d; d++) {
    int in_d = (align_corners) ? static_cast<int>(ratio_d * d + 0.5)
                               : static_cast<int>(ratio_d * d);
    for (int k = 0; k < out_h; k++) {  // loop for images
      int in_k = (align_corners) ? static_cast<int>(ratio_h * k + 0.5)
                                 : static_cast<int>(ratio_h * k);

      for (int l = 0; l < out_w; l++) {
        int in_l = (align_corners) ? static_cast<int>(ratio_w * l + 0.5)
                                   : static_cast<int>(ratio_w * l);

        for (int i = 0; i < n; i++) {    // loop for batches
          for (int j = 0; j < c; j++) {  // loop for channels
            if (data_layout == DataLayout::kNCHW) {
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
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,
    paddle::optional<const DenseTensor&> scale_tensor,
    const DenseTensor& output_grad,
    const std::string& data_layout_str,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* input_grad) {
  const DataLayout data_layout =
      paddle::framework::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  funcs::ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  float scale_w = -1.0;
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
  if (size_tensor && size_tensor->size() > 0) {
    // have size tensor
    auto new_size = funcs::get_new_shape(size_tensor.get());
    out_w = new_size[0];
  }

  phi::DDim dim_grad;
  if (data_layout == DataLayout::kNCHW) {
    dim_grad = {n, c, in_w};
  } else {
    dim_grad = {n, in_w, c};
  }

  input_grad->Resize(dim_grad);
  dev_ctx.template Alloc<T>(input_grad);

  phi::funcs::SetConstant<Context, T> zero;
  zero(dev_ctx, input_grad, static_cast<T>(0.0));

  if (in_w == out_w) {
    paddle::framework::TensorCopy(output_grad, dev_ctx.GetPlace(), input_grad);
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
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,
    paddle::optional<const DenseTensor&> scale_tensor,
    const DenseTensor& output_grad,
    const std::string& data_layout_str,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* input_grad) {
  const DataLayout data_layout =
      paddle::framework::StringToDataLayout(data_layout_str);
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
    out_h = static_cast<int>(in_h * scale_h);
    out_w = static_cast<int>(in_w * scale_w);
  }
  if (out_size) {
    auto out_size_data =
        funcs::get_new_data_from_tensor<int>(out_size.get_ptr());
    out_h = out_size_data[0];
    out_w = out_size_data[1];
  }
  if (size_tensor && size_tensor->size() > 0) {
    // have size tensor
    auto new_size = funcs::get_new_shape(size_tensor.get());
    out_h = new_size[0];
    out_w = new_size[1];
  }

  phi::DDim dim_grad;
  if (data_layout == DataLayout::kNCHW) {
    dim_grad = {n, c, in_h, in_w};
  } else {
    dim_grad = {n, in_h, in_w, c};
  }

  input_grad->Resize(dim_grad);
  dev_ctx.template Alloc<T>(input_grad);

  phi::funcs::SetConstant<Context, T> zero;
  zero(dev_ctx, input_grad, static_cast<T>(0.0));

  if (in_h == out_h && in_w == out_w) {
    paddle::framework::TensorCopy(output_grad, dev_ctx.GetPlace(), input_grad);
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
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,
    paddle::optional<const DenseTensor&> scale_tensor,
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
  const DataLayout data_layout =
      paddle::framework::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  funcs::ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  float scale_d = -1;
  float scale_h = -1;
  float scale_w = -1;
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
  if (size_tensor && size_tensor->size() > 0) {
    // have size tensor
    auto new_size = funcs::get_new_shape(size_tensor.get());
    out_d = new_size[0];
    out_h = new_size[1];
    out_w = new_size[2];
  }

  phi::DDim dim_grad;
  if (data_layout == DataLayout::kNCHW) {
    dim_grad = {n, c, in_d, in_h, in_w};
  } else {
    dim_grad = {n, in_d, in_h, in_w, c};
  }
  input_grad->Resize(dim_grad);
  dev_ctx.template Alloc<T>(input_grad);

  phi::funcs::SetConstant<Context, T> zero;
  zero(dev_ctx, input_grad, static_cast<T>(0.0));

  if (in_d == out_d && in_h == out_h && in_w == out_w) {
    paddle::framework::TensorCopy(output_grad, dev_ctx.GetPlace(), input_grad);
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
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,
    paddle::optional<const DenseTensor&> scale_tensor,
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
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,
    paddle::optional<const DenseTensor&> scale_tensor,
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
void NearestInterpGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,
    paddle::optional<const DenseTensor&> scale_tensor,
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
void TrilinearInterpGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,
    paddle::optional<const DenseTensor&> scale_tensor,
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
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,
    paddle::optional<const DenseTensor&> scale_tensor,
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
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,
    paddle::optional<const DenseTensor&> scale_tensor,
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

PD_REGISTER_KERNEL(bilinear_interp_v2_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::BilinearInterpGradKernel,
                   float,
                   double) {}
PD_REGISTER_KERNEL(nearest_interp_v2_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::NearestInterpGradKernel,
                   float,
                   double) {}
PD_REGISTER_KERNEL(trilinear_interp_v2_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::TrilinearInterpGradKernel,
                   float,
                   double) {}
PD_REGISTER_KERNEL(linear_interp_v2_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::LinearInterpGradKernel,
                   float,
                   double) {}
PD_REGISTER_KERNEL(bicubic_interp_v2_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::BicubicInterpGradKernel,
                   float,
                   double) {}
