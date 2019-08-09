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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/type_system.h"
#include "paddle/fluid/lite/operators/fc_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

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
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> B(b, w_w);
    Out = Out.array().rowwise() + B.transpose().array();
  }
}

template <typename T>
void fc_compute_naive(const T* x, int x_h, int x_w,  //
                      const T* w, int w_h, int w_w,  //
                      const T* b,                    //
                      T* out) {
  CHECK_EQ(x_w, w_h);
  // out shape: (x_h, w_w)
  memset(out, 0, x_h * w_w * sizeof(T));
  for (int i = 0; i < x_h; i++) {
    for (int j = 0; j < w_w; j++) {
      T tmp = static_cast<T>(0);
      for (int k = 0; k < x_w; k++) {
        tmp += x[i * x_w + k] * w[k * w_w + j];
      }
      out[i * w_w + j] = tmp + b[j];
    }
  }
}

template <typename T>
class FcCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::FcParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    CHECK_GE(param.input->dims().size(), 2UL);
    CHECK_EQ(param.output->dims().size(), 2UL);

    fc_compute_eigen(
        param.input->data<T>(),  // x
        param.input->dims().Slice(0, param.in_num_col_dims).production(),
        param.input->dims()
            .Slice(param.in_num_col_dims, param.input->dims().size())
            .production(),
        param.w->data<T>(),     // w
        param.w->dims()[0],     // w_h
        param.w->dims()[1],     // w_w
        param.bias->data<T>(),  // b
        param.output->mutable_data<T>());
  }

  virtual ~FcCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
