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
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/types.h"
#include "paddle/fluid/lite/cuda/blas.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T>
void mul_compute(const lite::cuda::Blas<float>& blas, const T* x, int x_h,
                 int x_w, const T* y, int y_h, int y_w, T* out) {
  blas.sgemm(CUBLAS_OP_N, CUBLAS_OP_N, x_w, x_h, y_w, nullptr, x, 0, y, 0,
             nullptr, out, 0);
}

class MulCompute : public OpKernel<TARGET(kHost), PRECISION(kFloat)> {
 public:
  using param_t = operators::MulParam;

  void Run() override {}

  virtual ~MulCompute() = default;
};

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
