/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <math.h>
#include <stdlib.h>

#include "paddle/fluid/platform/device_context.h"
#include "paddle/pten/core/hostdevice.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T, bool same_row>
struct CosSimFunctor {
  CosSimFunctor(const T* x, const T* y, T* x_norm, T* y_norm, T* z, int cols)
      : x_norm_(x_norm),
        y_norm_(y_norm),
        x_(x),
        y_(y),
        z_(z),
        cols_(static_cast<size_t>(cols)) {}

  inline HOSTDEVICE void operator()(size_t row_id) const {
    auto* x = x_ + cols_ * row_id;
    T xx = 0, xy = 0, yy = 0;
    T eps = 1e-8;
    if (same_row) {
      auto* y = y_ + cols_ * row_id;
      T tep_x, tep_y;
      for (size_t i = 0; i < cols_; ++i) {
        tep_x = x[i];
        tep_y = y[i];
        xx += tep_x * tep_x;

        yy += tep_y * tep_y;
        xy += tep_x * tep_y;
      }
      xx = xx > eps ? xx : eps;
      yy = yy > eps ? yy : eps;
      xx = sqrt(xx);
      yy = sqrt(yy);
      y_norm_[row_id] = yy;
      x_norm_[row_id] = xx;
      z_[row_id] = xy / (xx * yy);
    } else {  // This can be wrote in a better way.
      T tep_x, tep_y;
      for (size_t i = 0; i < cols_; ++i) {
        tep_x = x[i];
        tep_y = y_[i];
        xx += tep_x * tep_x;
        yy += tep_y * tep_y;
        xy += tep_x * tep_y;
      }
      xx = xx > eps ? xx : eps;
      yy = yy > eps ? yy : eps;
      xx = sqrt(xx);
      yy = sqrt(yy);
      if (row_id == 0) y_norm_[0] = yy;
      x_norm_[row_id] = xx;
      z_[row_id] = xy / (xx * yy);
    }
  }

  T* x_norm_;
  T* y_norm_;
  const T* x_;
  const T* y_;
  T* z_;
  const size_t cols_;
};

template <typename T>
struct CosSimGradFunctor {
  CosSimGradFunctor(const T* x_norm, const T* y_norm, const T* x, const T* y,
                    const T* z, const T* dz, T* dx, int cols)
      : x_norm_(x_norm),
        y_norm_(y_norm),
        x_(x),
        y_(y),
        z_(z),
        dz_(dz),
        dx_(dx),
        cols_(static_cast<size_t>(cols)) {}

  inline HOSTDEVICE void operator()(size_t row_id) const {
    auto x_norm_square = x_norm_[row_id] * x_norm_[row_id];
    auto xy_norm_prod = x_norm_[row_id] * y_norm_[row_id];
    auto dz = dz_[row_id];
    auto z = z_[row_id];

    auto* dx = dx_ + cols_ * row_id;
    auto* x = x_ + cols_ * row_id;
    auto* y = y_ + cols_ * row_id;

    auto reciprocal_xy_norm_prod = 1 / xy_norm_prod;
    auto reciprocal_x_norm_square = 1 / x_norm_square;
    for (size_t i = 0; i < cols_; ++i) {
      dx[i] = dz * (y[i] * reciprocal_xy_norm_prod -
                    z * x[i] * reciprocal_x_norm_square);
    }
  }

  const T* x_norm_;
  const T* y_norm_;
  const T* x_;
  const T* y_;
  const T* z_;
  const T* dz_;
  T* dx_;
  const size_t cols_;
};

template <typename T>
struct CosSimDxFunctor {
  CosSimDxFunctor(const T* x_norm, const T* y_norm, const T* x, const T* y,
                  const T* z, const T* dz, T* dx, int cols)
      : x_norm_(x_norm),
        y_norm_(y_norm),
        x_(x),
        y_(y),
        z_(z),
        dz_(dz),
        dx_(dx),
        cols_(static_cast<size_t>(cols)) {}

  inline HOSTDEVICE void operator()(size_t row_id) const {
    auto xy_norm_prod = x_norm_[row_id] * y_norm_[0];
    auto dz = dz_[row_id];
    auto z = z_[row_id];
    auto* x = x_ + cols_ * row_id;
    auto reciprocal_xy_norm_prod = 1 / xy_norm_prod;
    auto x_norm_square = x_norm_[row_id] * x_norm_[row_id];
    auto* dx = dx_ + cols_ * row_id;
    auto reciprocal_x_norm_square = 1 / x_norm_square;

    for (size_t i = 0; i < cols_; ++i) {
      dx[i] = dz * (y_[i] * reciprocal_xy_norm_prod -
                    z * x[i] * reciprocal_x_norm_square);
    }
  }
  const T* x_norm_;
  const T* y_norm_;
  const T* x_;
  const T* y_;
  const T* z_;
  const T* dz_;
  T* dx_;
  const size_t cols_;
};

template <typename DeviceContext, typename T>
struct CosSimDyFunctor {
  void operator()(const DeviceContext& ctx, const T* x_norm, const T* y_norm,
                  const T* x, const T* y, const T* z, const T* dz,
                  const size_t rows, const size_t cols, T* dy) const;
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
