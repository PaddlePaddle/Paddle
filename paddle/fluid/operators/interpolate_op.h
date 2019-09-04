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

#pragma once
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
using Tensor = framework::Tensor;

template <typename T>
static void NearestNeighborInterpolate(const Tensor& input, Tensor* output,
                                       const float ratio_h, const float ratio_w,
                                       const int n, const int c,
                                       const int out_h, const int out_w,
                                       const bool align_corners) {
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
          output_t(i, j, k, l) = input_t(i, j, in_k, in_l);
        }
      }
    }
  }
}

template <typename T>
static void BilinearInterpolation(const Tensor& input, Tensor* output,
                                  const float ratio_h, const float ratio_w,
                                  const int in_h, const int in_w, const int n,
                                  const int c, const int out_h, const int out_w,
                                  const bool align_corners,
                                  const bool align_mode) {
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
          T out_t = input_t(i, j, vy_n[k], vx_w[l]) * vd_s[k] * vd_e[l] +
                    input_t(i, j, vy_s[k], vx_w[l]) * vd_n[k] * vd_e[l] +
                    input_t(i, j, vy_n[k], vx_e[l]) * vd_s[k] * vd_w[l] +
                    input_t(i, j, vy_s[k], vx_e[l]) * vd_n[k] * vd_w[l];
          output_t(i, j, k, l) = out_t;
        }
      }
    }
  }
}

template <typename T>
static void TrilinearInterpolation(
    const Tensor& input, Tensor* output, const float ratio_d,
    const float ratio_h, const float ratio_w, const int in_d, const int in_h,
    const int in_w, const int n, const int c, const int out_d, const int out_h,
    const int out_w, const bool align_corners, const bool align_mode) {
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
          }
        }
      }
    }
  }
}

template <typename T>
static void NearestNeighborInterpolateGrad(
    const Tensor& output_grad, Tensor* input_grad, const float ratio_h,
    const float ratio_w, const int n, const int c, const int out_h,
    const int out_w, const bool align_corners) {
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
          input_grad_t(i, j, in_k, in_l) += output_grad_t(i, j, k, l);
        }
      }
    }
  }
}

template <typename T>
static void BilinearInterpolationGrad(const Tensor& output_grad,
                                      Tensor* input_grad, const float ratio_h,
                                      const float ratio_w, const int in_h,
                                      const int in_w, const int n, const int c,
                                      const int out_h, const int out_w,
                                      const bool align_corners,
                                      const int align_mode) {
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
          const T grad = output_grad_t(i, j, k, l);
          input_grad_t(i, j, y_n, x_w) += static_cast<T>(grad * d_s * d_e);
          input_grad_t(i, j, y_s, x_w) += static_cast<T>(grad * d_n * d_e);
          input_grad_t(i, j, y_n, x_e) += static_cast<T>(grad * d_s * d_w);
          input_grad_t(i, j, y_s, x_e) += static_cast<T>(grad * d_n * d_w);
        }
      }
    }
  }
}

template <typename T>
static void TrilinearInterpolationGrad(
    const Tensor& output_grad, Tensor* input_grad, const float ratio_d,
    const float ratio_h, const float ratio_w, const int in_d, const int in_h,
    const int in_w, const int n, const int c, const int out_d, const int out_h,
    const int out_w, const bool align_corners, const int align_mode) {
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
          }
        }
      }
    }
  }
}

template <typename T>
static void Interpolate2DCPUFwd(const framework::ExecutionContext& ctx,
                                const Tensor& input, Tensor* output) {
  const int n = input.dims()[0];
  const int c = input.dims()[1];
  const int in_h = input.dims()[2];
  const int in_w = input.dims()[3];

  auto interp_method = ctx.Attr<std::string>("interp_method");
  bool align_corners = ctx.Attr<bool>("align_corners");
  int align_mode = ctx.Attr<int>("align_mode");

  int out_h = ctx.Attr<int>("out_h");
  int out_w = ctx.Attr<int>("out_w");
  float scale = ctx.Attr<float>("scale");
  if (scale > 0) {
    out_h = static_cast<int>(in_h * scale);
    out_w = static_cast<int>(in_w * scale);
  }

  auto out_size = ctx.Input<Tensor>("OutSize");
  if (out_size != nullptr) {
    auto out_size_data = out_size->data<int>();
    out_h = out_size_data[0];
    out_w = out_size_data[1];
  }

  output->mutable_data<T>({n, c, out_h, out_w}, ctx.GetPlace());

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

  if ("bilinear" == interp_method) {
    BilinearInterpolation<T>(input, output, ratio_h, ratio_w, in_h, in_w, n, c,
                             out_h, out_w, align_corners, align_mode);
  } else if ("nearest" == interp_method) {
    NearestNeighborInterpolate<T>(input, output, ratio_h, ratio_w, n, c, out_h,
                                  out_w, align_corners);
  }
}

template <typename T>
static void Interpolate3DCPUFwd(const framework::ExecutionContext& ctx,
                                const Tensor& input, Tensor* output) {
  const int n = input.dims()[0];
  const int c = input.dims()[1];
  const int in_d = input.dims()[2];
  const int in_h = input.dims()[3];
  const int in_w = input.dims()[4];

  auto interp_method = ctx.Attr<std::string>("interp_method");
  bool align_corners = ctx.Attr<bool>("align_corners");
  int align_mode = ctx.Attr<int>("align_mode");

  int out_d = ctx.Attr<int>("out_d");
  int out_h = ctx.Attr<int>("out_h");
  int out_w = ctx.Attr<int>("out_w");
  float scale = ctx.Attr<float>("scale");
  if (scale > 0) {
    out_d = static_cast<int>(in_d * scale);
    out_h = static_cast<int>(in_h * scale);
    out_w = static_cast<int>(in_w * scale);
  }

  auto out_size = ctx.Input<Tensor>("OutSize");
  if (out_size != nullptr) {
    auto out_size_data = out_size->data<int>();
    out_d = out_size_data[0];
    out_h = out_size_data[1];
    out_w = out_size_data[2];
  }

  output->mutable_data<T>({n, c, out_d, out_h, out_w}, ctx.GetPlace());

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

  if ("trilinear" == interp_method) {
    TrilinearInterpolation<T>(input, output, ratio_d, ratio_h, ratio_w, in_d,
                              in_h, in_w, n, c, out_d, out_h, out_w,
                              align_corners, align_mode);
  }
}

template <typename T>
static void Interpolate2DCPUBwd(const framework::ExecutionContext& ctx,
                                Tensor* input_grad, const Tensor& output_grad) {
  auto* input = ctx.Input<Tensor>("X");
  const int n = input->dims()[0];
  const int c = input->dims()[1];
  const int in_h = input->dims()[2];
  const int in_w = input->dims()[3];

  auto interp_method = ctx.Attr<std::string>("interp_method");
  bool align_corners = ctx.Attr<bool>("align_corners");
  int align_mode = ctx.Attr<int>("align_mode");

  int out_h = ctx.Attr<int>("out_h");
  int out_w = ctx.Attr<int>("out_w");
  float scale = ctx.Attr<float>("scale");
  if (scale > 0) {
    out_h = static_cast<int>(in_h * scale);
    out_w = static_cast<int>(in_w * scale);
  }

  auto out_size = ctx.Input<Tensor>("OutSize");
  if (out_size != nullptr) {
    auto out_size_data = out_size->data<int>();
    out_h = out_size_data[0];
    out_w = out_size_data[1];
  }

  input_grad->mutable_data<T>({n, c, in_h, in_w}, ctx.GetPlace());
  auto& device_ctx = ctx.template device_context<platform::CPUDeviceContext>();
  math::SetConstant<platform::CPUDeviceContext, T> zero;
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

  if ("bilinear" == interp_method) {
    BilinearInterpolationGrad<T>(output_grad, input_grad, ratio_h, ratio_w,
                                 in_h, in_w, n, c, out_h, out_w, align_corners,
                                 align_mode);
  } else if ("nearest" == interp_method) {
    NearestNeighborInterpolateGrad<T>(output_grad, input_grad, ratio_h, ratio_w,
                                      n, c, out_h, out_w, align_corners);
  }
}

template <typename T>
static void Interpolate3DCPUBwd(const framework::ExecutionContext& ctx,
                                Tensor* input_grad, const Tensor output_grad) {
  auto* input = ctx.Input<Tensor>("X");
  const int n = input->dims()[0];
  const int c = input->dims()[1];
  const int in_d = input->dims()[2];
  const int in_h = input->dims()[3];
  const int in_w = input->dims()[4];

  auto interp_method = ctx.Attr<std::string>("interp_method");
  bool align_corners = ctx.Attr<bool>("align_corners");
  int align_mode = ctx.Attr<int>("align_mode");

  int out_d = ctx.Attr<int>("out_d");
  int out_h = ctx.Attr<int>("out_h");
  int out_w = ctx.Attr<int>("out_w");
  float scale = ctx.Attr<float>("scale");
  if (scale > 0) {
    out_d = static_cast<int>(in_d * scale);
    out_h = static_cast<int>(in_h * scale);
    out_w = static_cast<int>(in_w * scale);
  }

  auto out_size = ctx.Input<Tensor>("OutSize");
  if (out_size != nullptr) {
    auto out_size_data = out_size->data<int>();
    out_d = out_size_data[0];
    out_h = out_size_data[1];
    out_w = out_size_data[2];
  }

  input_grad->mutable_data<T>({n, c, in_d, in_h, in_w}, ctx.GetPlace());
  auto& device_ctx = ctx.template device_context<platform::CPUDeviceContext>();
  math::SetConstant<platform::CPUDeviceContext, T> zero;
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

  if ("trilinear" == interp_method) {
    TrilinearInterpolationGrad<T>(output_grad, input_grad, ratio_d, ratio_h,
                                  ratio_w, in_d, in_h, in_w, n, c, out_d, out_h,
                                  out_w, align_corners, align_mode);
  }
}

template <typename T>
class InterpolateKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");

    auto input_dims = input->dims();
    if (input_dims.size() == 4) {  // 2D interpolation
      Interpolate2DCPUFwd<T>(ctx, *input, output);
    } else if (input_dims.size() == 5) {  // 3D interpolation
      Interpolate3DCPUFwd<T>(ctx, *input, output);
    }
  }
};

template <typename T>
class InterpolateGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto output_grad_dims = output_grad->dims();
    if (output_grad_dims.size() == 4) {  // 2D interpolation grad
      Interpolate2DCPUBwd<T>(ctx, input_grad, *output_grad);
    } else if (output_grad_dims.size() == 5) {  // 3D interpolation grad
      Interpolate3DCPUBwd<T>(ctx, input_grad, *output_grad);
    }
  }
};

}  // namespace operators
}  // namespace paddle
