/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#ifndef PADDLE_FLUID_OPERATORS_BMM_OP_H_
#define PADDLE_FLUID_OPERATORS_BMM_OP_H_

#include <algorithm>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/pten/kernels/funcs/math_function.h"
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

static void ReshapeTensorIntoMatrixSequence(
    framework::Tensor *x, const math::MatDescriptor &descriptor) {
  int64_t h, w;
  h = descriptor.height_;
  w = descriptor.width_;
  if (descriptor.trans_) {
    std::swap(w, h);
  }

  x->Resize({descriptor.batch_size_, h, w});
}

static void ReshapeXYOutIntoMatrixSequence(framework::Tensor *x,
                                           framework::Tensor *y,
                                           framework::Tensor *out, bool trans_x,
                                           bool trans_y) {
  auto x_dim = x->dims();
  auto y_dim = y->dims();
  auto mat_dim_x = math::CreateMatrixDescriptor(x_dim, 0, false);
  auto mat_dim_y = math::CreateMatrixDescriptor(y_dim, 0, false);

  out->Resize({std::max(mat_dim_x.batch_size_, mat_dim_y.batch_size_),
               mat_dim_x.height_, mat_dim_y.width_});

  ReshapeTensorIntoMatrixSequence(x, mat_dim_x);
  ReshapeTensorIntoMatrixSequence(y, mat_dim_y);
}

template <typename DeviceContext, typename T>
class BmmKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor &x = *context.Input<Tensor>("X");
    const Tensor &y = *context.Input<Tensor>("Y");
    Tensor *out = context.Output<Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    if (x.numel() == 0 || y.numel() == 0) {
      return;
    }

    auto blas = math::GetBlas<DeviceContext, T>(context);

    auto mat_dim_a = math::CreateMatrixDescriptor(x.dims(), 0, false);
    auto mat_dim_b = math::CreateMatrixDescriptor(y.dims(), 0, false);

    // auto scale = static_cast<T>(context.Attr<float>("alpha"));
    blas.MatMul(x, mat_dim_a, y, mat_dim_b, T(1), out, T(0));
  }
};

template <typename DeviceContext, typename T>
class BmmGradKernel : public framework::OpKernel<T> {
 public:
  void MatMul(const framework::ExecutionContext &context,
              const framework::Tensor &a, bool trans_a,
              const framework::Tensor &b, bool trans_b,
              framework::Tensor *out) const {
    out->mutable_data<T>(context.GetPlace());
    auto blas = math::GetBlas<DeviceContext, T>(context);
    auto mat_dim_a = math::CreateMatrixDescriptor(a.dims(), 0, trans_a);
    auto mat_dim_b = math::CreateMatrixDescriptor(b.dims(), 0, trans_b);

    blas.MatMul(a, mat_dim_a, b, mat_dim_b, T(1), out, T(0));
  }
  void CalcInputGrad(const framework::ExecutionContext &context,
                     const framework::Tensor &a, bool trans_a,
                     const framework::Tensor &b, bool trans_b,
                     framework::Tensor *out) const {
    if (out == nullptr) return;
    MatMul(context, a, trans_a, b, trans_b, out);
  }
  void Compute(const framework::ExecutionContext &context) const override {
    auto x = *context.Input<framework::Tensor>("X");
    auto y = *context.Input<framework::Tensor>("Y");
    auto dout =
        *context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *dx = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto *dy = context.Output<framework::Tensor>(framework::GradVarName("Y"));

    ReshapeXYOutIntoMatrixSequence(&x, &y, &dout, false, false);
    framework::DDim dx_dims;
    if (dx) {
      dx_dims = dx->dims();
      if (dx_dims != x.dims()) {
        dx->Resize(x.dims());
      }
    }

    framework::DDim dy_dims;
    if (dy) {
      dy_dims = dy->dims();
      if (dy_dims != y.dims()) {
        dy->Resize(y.dims());
      }
    }

    CalcInputGrad(context, dout, false, y, true, dx);
    CalcInputGrad(context, x, true, dout, false, dy);

    if (dx) {
      if (dx_dims != x.dims()) {
        dx->Resize(dx_dims);
      }
    }
    if (dy) {
      if (dy_dims != y.dims()) {
        dy->Resize(dy_dims);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
#endif  // PADDLE_FLUID_OPERATORS_BMM_OP_H_
