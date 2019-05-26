// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <Eigen/Core>
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/operators/fc_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename T>
void fc_compute_eigen(const T* x, int x_h, int x_w,  //
                      const T* w, int w_h, int w_w,  //
                      const T* b,                    //
                      T* out) {
  using matrix_t =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  Eigen::Map<const matrix_t> X(x, x_h, x_w);
  Eigen::Map<const matrix_t> W(w, w_h, w_w);
  Eigen::Map<matrix_t> Out(out, x_h, w_w);

  Out = X * W;

  if (b) {
    Eigen::Map<const Eigen::Matrix<T, 1, Eigen::Dynamic>> B(b, w_w);
    Out = Out.array().rowwise() + B.array();
  }
}

class FcCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::FcParam;

  void Run() override;

  TargetType target() const override;
  PrecisionType precision() const override;

  virtual ~FcCompute() = default;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
