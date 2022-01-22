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
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/pten/core/hostdevice.h"

namespace paddle {
namespace operators {

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;

inline std::vector<int> get_new_shape(
    const std::vector<const Tensor*>& list_new_shape_tensor) {
  // get tensor from
  std::vector<int> vec_new_shape;
  for (size_t i = 0; i < list_new_shape_tensor.size(); ++i) {
    auto tensor = list_new_shape_tensor[i];
    PADDLE_ENFORCE_EQ(tensor->dims(), framework::make_ddim({1}),
                      platform::errors::InvalidArgument(
                          "The shape of dimension tensor should be [1],"
                          "but received d%.",
                          tensor->dims()));
    if (platform::is_gpu_place(tensor->place())) {
      framework::Tensor temp;
      paddle::framework::TensorCopySync(*tensor, platform::CPUPlace(), &temp);
      vec_new_shape.push_back(static_cast<int32_t>(*temp.data<int32_t>()));
    } else {
      vec_new_shape.push_back(static_cast<int32_t>(*tensor->data<int32_t>()));
    }
  }

  return vec_new_shape;
}

template <typename T>
inline std::vector<T> get_new_data_from_tensor(const Tensor* new_data_tensor) {
  std::vector<T> vec_new_data;
  auto* new_data = new_data_tensor->data<T>();
  framework::Tensor cpu_starts_tensor;
  if (platform::is_gpu_place(new_data_tensor->place())) {
    paddle::framework::TensorCopySync(*new_data_tensor, platform::CPUPlace(),
                                      &cpu_starts_tensor);
    new_data = cpu_starts_tensor.data<T>();
  }
  vec_new_data = std::vector<T>(new_data, new_data + new_data_tensor->numel());
  return vec_new_data;
}

inline void ExtractNCDWH(const framework::DDim& dims,
                         const DataLayout& data_layout, int* N, int* C, int* D,
                         int* H, int* W) {
  *N = dims[0];

  if (dims.size() == 3) {
    *C = data_layout == DataLayout::kNCHW ? dims[1] : dims[2];
    *D = 1;
    *H = 1;
    *W = data_layout == DataLayout::kNCHW ? dims[2] : dims[1];
  } else if (dims.size() == 4) {
    *C = data_layout == DataLayout::kNCHW ? dims[1] : dims[3];
    *D = 1;
    *H = data_layout == DataLayout::kNCHW ? dims[2] : dims[1];
    *W = data_layout == DataLayout::kNCHW ? dims[3] : dims[2];
  } else {
    *C = data_layout == DataLayout::kNCHW ? dims[1] : dims[4];
    *D = data_layout == DataLayout::kNCHW ? dims[2] : dims[1];
    *H = data_layout == DataLayout::kNCHW ? dims[3] : dims[2];
    *W = data_layout == DataLayout::kNCHW ? dims[4] : dims[3];
  }
}

template <typename T>
static void NearestNeighborInterpolate(const Tensor& input, Tensor* output,
                                       const float ratio_h, const float ratio_w,
                                       const int n, const int c,
                                       const int out_h, const int out_w,
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
static void LinearInterpolation(const Tensor& input, Tensor* output,
                                const float ratio_w, const int in_w,
                                const int n, const int c, const int out_w,
                                const bool align_corners, const bool align_mode,
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
static void LinearInterpolationGrad(const Tensor& output_grad,
                                    Tensor* input_grad, const float ratio_w,
                                    const int in_w, const int n, const int c,
                                    const int out_w, const bool align_corners,
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
static void BilinearInterpolation(const Tensor& input, Tensor* output,
                                  const float ratio_h, const float ratio_w,
                                  const int in_h, const int in_w, const int n,
                                  const int c, const int out_h, const int out_w,
                                  const bool align_corners,
                                  const bool align_mode,
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
static void TrilinearInterpolation(
    const Tensor& input, Tensor* output, const float ratio_d,
    const float ratio_h, const float ratio_w, const int in_d, const int in_h,
    const int in_w, const int n, const int c, const int out_d, const int out_h,
    const int out_w, const bool align_corners, const bool align_mode,
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
HOSTDEVICE inline T cubic_convolution1(T x, T A) {
  return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template <typename T>
HOSTDEVICE inline T cubic_convolution2(T x, T A) {
  return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template <typename T>
HOSTDEVICE inline void get_cubic_upsample_coefficients(T coeffs[4], T t) {
  T A = -0.75;

  T x1 = t;
  coeffs[0] = cubic_convolution2<T>(x1 + 1.0, A);
  coeffs[1] = cubic_convolution1<T>(x1, A);

  // opposite coefficients
  T x2 = 1.0 - t;
  coeffs[2] = cubic_convolution1<T>(x2, A);
  coeffs[3] = cubic_convolution2<T>(x2 + 1.0, A);
}

template <typename T>
static inline T cubic_interp(T x0, T x1, T x2, T x3, T t) {
  T coeffs[4];
  get_cubic_upsample_coefficients<T>(coeffs, t);

  return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

template <typename T>
static void BicubicInterpolation(const Tensor& input, Tensor* output,
                                 const float ratio_h, const float ratio_w,
                                 const int in_h, const int in_w, const int n,
                                 const int c, const int out_h, const int out_w,
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
                                  input_t(i, j, access_y, access_x_3), x_t);
            } else {
              coefficients[ii] =
                  cubic_interp<T>(input_t(i, access_y, access_x_0, j),
                                  input_t(i, access_y, access_x_1, j),
                                  input_t(i, access_y, access_x_2, j),
                                  input_t(i, access_y, access_x_3, j), x_t);
            }
          }

          // interp y direction
          if (data_layout == DataLayout::kNCHW) {
            output_t(i, j, k, l) =
                cubic_interp<T>(coefficients[0], coefficients[1],
                                coefficients[2], coefficients[3], y_t);
          } else {
            output_t(i, k, l, j) =
                cubic_interp<T>(coefficients[0], coefficients[1],
                                coefficients[2], coefficients[3], y_t);
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
    const int out_w, const bool align_corners, const DataLayout data_layout) {
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
static void BilinearInterpolationGrad(
    const Tensor& output_grad, Tensor* input_grad, const float ratio_h,
    const float ratio_w, const int in_h, const int in_w, const int n,
    const int c, const int out_h, const int out_w, const bool align_corners,
    const int align_mode, const DataLayout data_layout) {
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
static void TrilinearInterpolationGrad(
    const Tensor& output_grad, Tensor* input_grad, const float ratio_d,
    const float ratio_h, const float ratio_w, const int in_d, const int in_h,
    const int in_w, const int n, const int c, const int out_d, const int out_h,
    const int out_w, const bool align_corners, const int align_mode,
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
static void BicubicInterpolationGrad(const Tensor& output_grad,
                                     Tensor* input_grad, const float ratio_h,
                                     const float ratio_w, const int in_h,
                                     const int in_w, const int n, const int c,
                                     const int out_h, const int out_w,
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

      get_cubic_upsample_coefficients<T>(x_coeffs, x_t);
      get_cubic_upsample_coefficients<T>(y_coeffs, y_t);

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
static void Interpolate1DCPUFwd(const framework::ExecutionContext& ctx,
                                const Tensor& input, Tensor* output) {
  const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
  const DataLayout data_layout = framework::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  auto interp_method = ctx.Attr<std::string>("interp_method");
  bool align_corners = ctx.Attr<bool>("align_corners");
  int align_mode = ctx.Attr<int>("align_mode");

  int out_w = ctx.Attr<int>("out_w");
  auto list_new_size_tensor = ctx.MultiInput<framework::Tensor>("SizeTensor");
  if (list_new_size_tensor.size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(list_new_size_tensor);
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
      auto out_size_data = get_new_data_from_tensor<int>(out_size);
      out_w = out_size_data[0];
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
  output->mutable_data<T>(dim_out, ctx.GetPlace());

  if (in_w == out_w) {
    framework::TensorCopy(input, ctx.GetPlace(), output);
    return;
  }

  float ratio_w = 0.f;
  if (out_w > 1) {
    ratio_w = (align_corners) ? static_cast<float>(in_w - 1) / (out_w - 1)
                              : static_cast<float>(in_w) / out_w;
  }
  if ("linear" == interp_method) {
    LinearInterpolation<T>(input, output, ratio_w, in_w, n, c, out_w,
                           align_corners, align_mode, data_layout);
  }
}

template <typename T>
static void Interpolate2DCPUFwd(const framework::ExecutionContext& ctx,
                                const Tensor& input, Tensor* output) {
  const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
  const DataLayout data_layout = framework::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  auto interp_method = ctx.Attr<std::string>("interp_method");
  bool align_corners = ctx.Attr<bool>("align_corners");
  int align_mode = ctx.Attr<int>("align_mode");

  int out_h = ctx.Attr<int>("out_h");
  int out_w = ctx.Attr<int>("out_w");

  auto list_new_size_tensor = ctx.MultiInput<framework::Tensor>("SizeTensor");
  if (list_new_size_tensor.size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(list_new_size_tensor);
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
      auto out_size_data = get_new_data_from_tensor<int>(out_size);
      out_h = out_size_data[0];
      out_w = out_size_data[1];
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
  output->mutable_data<T>(dim_out, ctx.GetPlace());

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
                             out_h, out_w, align_corners, align_mode,
                             data_layout);
  } else if ("nearest" == interp_method) {
    NearestNeighborInterpolate<T>(input, output, ratio_h, ratio_w, n, c, out_h,
                                  out_w, align_corners, data_layout);
  } else if ("bicubic" == interp_method) {
    BicubicInterpolation<T>(input, output, ratio_h, ratio_w, in_h, in_w, n, c,
                            out_h, out_w, align_corners, data_layout);
  }
}

template <typename T>
static void Interpolate3DCPUFwd(const framework::ExecutionContext& ctx,
                                const Tensor& input, Tensor* output) {
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

  auto list_new_size_tensor = ctx.MultiInput<framework::Tensor>("SizeTensor");
  if (list_new_size_tensor.size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(list_new_size_tensor);
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
      auto out_size_data = get_new_data_from_tensor<int>(out_size);
      out_d = out_size_data[0];
      out_h = out_size_data[1];
      out_w = out_size_data[2];
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

  output->mutable_data<T>(dim_out, ctx.GetPlace());

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
                              align_corners, align_mode, data_layout);
  }
}

template <typename T>
static void Interpolate1DCPUBwd(const framework::ExecutionContext& ctx,
                                Tensor* input_grad, const Tensor& output_grad) {
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
    auto out_size_data = get_new_data_from_tensor<int>(out_size);
    out_w = out_size_data[0];
  }
  auto list_new_size_tensor = ctx.MultiInput<framework::Tensor>("SizeTensor");
  if (list_new_size_tensor.size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(list_new_size_tensor);
    out_w = new_size[0];
  }

  framework::DDim dim_grad;
  if (data_layout == DataLayout::kNCHW) {
    dim_grad = {n, c, in_w};
  } else {
    dim_grad = {n, in_w, c};
  }
  input_grad->mutable_data<T>(dim_grad, ctx.GetPlace());

  auto& device_ctx = ctx.template device_context<platform::CPUDeviceContext>();
  math::SetConstant<platform::CPUDeviceContext, T> zero;
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
  if ("linear" == interp_method) {
    LinearInterpolationGrad<T>(output_grad, input_grad, ratio_w, in_w, n, c,
                               out_w, align_corners, align_mode, data_layout);
  }
}

template <typename T>
static void Interpolate2DCPUBwd(const framework::ExecutionContext& ctx,
                                Tensor* input_grad, const Tensor& output_grad) {
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
    auto out_size_data = get_new_data_from_tensor<int>(out_size);
    out_h = out_size_data[0];
    out_w = out_size_data[1];
  }
  auto list_new_size_tensor = ctx.MultiInput<framework::Tensor>("SizeTensor");
  if (list_new_size_tensor.size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(list_new_size_tensor);
    out_h = new_size[0];
    out_w = new_size[1];
  }

  framework::DDim dim_grad;
  if (data_layout == DataLayout::kNCHW) {
    dim_grad = {n, c, in_h, in_w};
  } else {
    dim_grad = {n, in_h, in_w, c};
  }
  input_grad->mutable_data<T>(dim_grad, ctx.GetPlace());

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
                                 align_mode, data_layout);
  } else if ("nearest" == interp_method) {
    NearestNeighborInterpolateGrad<T>(output_grad, input_grad, ratio_h, ratio_w,
                                      n, c, out_h, out_w, align_corners,
                                      data_layout);
  } else if ("bicubic" == interp_method) {
    BicubicInterpolationGrad<T>(output_grad, input_grad, ratio_h, ratio_w, in_h,
                                in_w, n, c, out_h, out_w, align_corners,
                                data_layout);
  }
}

template <typename T>
static void Interpolate3DCPUBwd(const framework::ExecutionContext& ctx,
                                Tensor* input_grad, const Tensor output_grad) {
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
    auto out_size_data = get_new_data_from_tensor<int>(out_size);
    out_d = out_size_data[0];
    out_h = out_size_data[1];
    out_w = out_size_data[2];
  }
  auto list_new_size_tensor = ctx.MultiInput<framework::Tensor>("SizeTensor");
  if (list_new_size_tensor.size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(list_new_size_tensor);
    out_d = new_size[0];
    out_h = new_size[1];
    out_w = new_size[2];
  }

  framework::DDim dim_grad;
  if (data_layout == DataLayout::kNCHW) {
    dim_grad = {n, c, in_d, in_h, in_w};
  } else {
    dim_grad = {n, in_d, in_h, in_w, c};
  }
  input_grad->mutable_data<T>(dim_grad, ctx.GetPlace());
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
    TrilinearInterpolationGrad<T>(
        output_grad, input_grad, ratio_d, ratio_h, ratio_w, in_d, in_h, in_w, n,
        c, out_d, out_h, out_w, align_corners, align_mode, data_layout);
  }
}

template <typename T>
class InterpolateKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");

    auto input_dims = input->dims();
    if (input_dims.size() == 3) {  // 1D interpolation
      Interpolate1DCPUFwd<T>(ctx, *input, output);
    } else if (input_dims.size() == 4) {  // 2D interpolation
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
    if (output_grad_dims.size() == 3) {  // 1D interpolation grad
      Interpolate1DCPUBwd<T>(ctx, input_grad, *output_grad);
    } else if (output_grad_dims.size() == 4) {  // 2D interpolation grad
      Interpolate2DCPUBwd<T>(ctx, input_grad, *output_grad);
    } else if (output_grad_dims.size() == 5) {  // 3D interpolation grad
      Interpolate3DCPUBwd<T>(ctx, input_grad, *output_grad);
    }
  }
};

}  // namespace operators
}  // namespace paddle
