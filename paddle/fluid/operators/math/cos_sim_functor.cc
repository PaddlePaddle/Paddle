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

#include "paddle/fluid/operators/math/cos_sim_functor.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct CosSimDyFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx, const T* x_norm,
                  const T* y_norm, const T* x, const T* y, const T* z,
                  const T* dz, const size_t rows, const size_t cols,
                  T* dy) const {
    for (size_t row_id = 0; row_id < rows; ++row_id) {
      auto xy_norm_prod = x_norm[row_id] * y_norm[0];
      auto dz_data = dz[row_id];
      auto z_data = z[row_id];
      auto* x_data = x + cols * row_id;
      auto reciprocal_xy_norm_prod = 1 / xy_norm_prod;

      auto y_norm_square = y_norm[0] * y_norm[0];
      auto reciprocal_y_norm_square = 1 / y_norm_square;
      for (size_t i = 0; i < cols; ++i) {
        dy[i] += dz_data * (x_data[i] * reciprocal_xy_norm_prod -
                            z_data * y[i] * reciprocal_y_norm_square);
      }
    }
  }
};

template struct CosSimDyFunctor<platform::CPUDeviceContext, float>;
template struct CosSimDyFunctor<platform::CPUDeviceContext, double>;
}  // namespace math
}  // namespace operators
}  // namespace paddle
