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
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/types.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

using Tensor = framework::Tensor;

template <typename T>
class MulCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::MulParam;

  void Run() override {
    auto& context = ctx_->As<X86Context>();
    auto& param = *param_.get_mutable<operators::MulParam>();
    CHECK(context.x86_device_context());

    param.output->template mutable_data<T>();

    auto* x = &param.x->raw_tensor();
    auto* y = &param.y->raw_tensor();

    Tensor x_matrix, y_matrix;

    if (x->dims().size() > 2) {
      x_matrix = framework::ReshapeToMatrix(*x, param.x_num_col_dims);
    } else {
      x_matrix = *x;
    }

    if (y->dims().size() > 2) {
      y_matrix = framework::ReshapeToMatrix(*y, param.y_num_col_dims);

    } else {
      y_matrix = *y;
    }

    auto* z = &param.output->raw_tensor();
    auto z_dim = z->dims();
    if (z_dim.size() != 2) {
      z->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
    }

    auto blas = paddle::operators::math::GetBlas<platform::CPUDeviceContext, T>(
        *context.x86_device_context());

    blas.MatMul(x_matrix, y_matrix, z);
    if (z_dim.size() != 2) {
      z->Resize(z_dim);
    }
  }

  virtual ~MulCompute() = default;
};

template <typename T>
class MulGradCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override {
    auto& context = ctx_->As<X86Context>();
    auto& param = *param_.get_mutable<operators::MulGradParam>();
    CHECK(context.x86_device_context());

    auto* x = &param.x->raw_tensor();
    auto* y = &param.y->raw_tensor();

    Tensor x_matrix, y_matrix;

    if (x->dims().size() > 2) {
      x_matrix = framework::ReshapeToMatrix(*x, param.x_num_col_dims);
    } else {
      x_matrix = *x;
    }

    if (y->dims().size() > 2) {
      y_matrix = framework::ReshapeToMatrix(*y, param.y_num_col_dims);

    } else {
      y_matrix = *y;
    }

    auto* dout = &param.output_grad->raw_tensor();

    Tensor dout_mat;
    dout_mat.ShareDataWith(*dout);
    dout_mat.Resize(
        {framework::flatten_to_2d(x->dims(), param.x_num_col_dims)[0],
         framework::flatten_to_2d(y->dims(), param.y_num_col_dims)[1]});

    auto* dx = &param.x_grad->raw_tensor();
    auto* dy = &param.y_grad->raw_tensor();

    if (dx != nullptr) {
      dx->set_lod(x->lod());
    }
    if (dy != nullptr) {
      dy->set_lod(y->lod());
    }

    auto blas = paddle::operators::math::GetBlas<platform::CPUDeviceContext, T>(
        *context.x86_device_context());
    if (dx) {
      // dx->mutable_data<T>(context.x86_device_context->GetPlace());
      param.x_grad->template mutable_data<T>();
      Tensor dx_matrix = dx->dims().size() > 2 ? framework::ReshapeToMatrix(
                                                     *dx, param.x_num_col_dims)
                                               : *dx;

      // dx = dout * y'. dx: M x K, dout : M x N, y : K x N
      blas.MatMul(dout_mat, false, y_matrix, true, &dx_matrix);
    }
    if (dy) {
      // dy->yutable_data<T>(context.x86_device_context->GetPlace());
      param.y_grad->template mutable_data<T>();
      Tensor dy_matrix = dy->dims().size() > 2 ? framework::ReshapeToMatrix(
                                                     *dy, param.y_num_col_dims)
                                               : *dy;
      // dy = x' * dout. dy K x N, dout : M x N, x : M x K
      blas.MatMul(x_matrix, true, dout_mat, false, &dy_matrix);
    }
  }

  virtual ~MulGradCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
