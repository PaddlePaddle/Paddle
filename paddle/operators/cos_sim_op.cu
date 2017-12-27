/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#define EIGEN_USE_GPU
#include "paddle/operators/cos_sim_op.h"

namespace paddle {
namespace operators {

template <typename T>
struct CosSimDyFunctor<platform::CUDADeviceContext, T> {
  CosSimDyFunctor(const T* x_norm, const T* y_norm, const T* x, const T* y,
                  const T* z, const T* dz, T* dy, int cols)
      : x_norm_(x_norm),
        y_norm_(y_norm),
        x_(x),
        y_(y),
        z_(z),
        dz_(dz),
        dy_(dy),
        cols_(static_cast<size_t>(cols)) {}

  inline void operator()(size_t offset) const {
    auto xy_norm_prod = x_norm_[offset] * y_norm_[0];
    auto dz = dz_[offset];
    auto z = z_[offset];
    auto* x = x_ + cols_ * offset;
    auto reciprocal_xy_norm_prod = 1 / xy_norm_prod;

    auto y_norm_square = y_norm_[0] * y_norm_[0];
    auto reciprocal_y_norm_square = 1 / y_norm_square;
    for (size_t i = 0; i < cols_; ++i) {
      T dy = dz * (x[i] * reciprocal_xy_norm_prod -
                   z * y_[i] * reciprocal_y_norm_square);
      paddle::paddleAtomicAdd(dy_ + i, dy)
    }
  }

  const T* x_norm_;
  const T* y_norm_;
  const T* x_;
  const T* y_;
  const T* z_;
  const T* dz_;
  T* dy_;
  const size_t cols_;
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    cos_sim, ops::CosSimKernel<paddle::platform::CUDADeviceContext, float>);
REGISTER_OP_CUDA_KERNEL(
    cos_sim_grad,
    ops::CosSimGradKernel<paddle::platform::CUDADeviceContext, float>);
