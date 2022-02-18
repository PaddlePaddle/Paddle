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

#include <boost/preprocessor/repetition/repeat.hpp>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;

using Array1 = Eigen::DSizes<Eigen::DenseIndex, 1>;
using Array2 = Eigen::DSizes<Eigen::DenseIndex, 2>;

using Tensor = framework::Tensor;

constexpr int kMULMKLDNNINT8 = 1;

template <typename DeviceContext, typename T>
class AddMMKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    const Tensor* x = context.Input<Tensor>("X");
    const Tensor* y = context.Input<Tensor>("Y");

    auto input_dims = input->dims();
    auto x_dims = x->dims();
    auto y_dims = y->dims();

    // broadcast mode check
    if (x_dims[0] != input_dims[0]) {
      PADDLE_ENFORCE_EQ(input_dims[0], 1,
                        platform::errors::InvalidArgument(
                            "When x_dims[0] is not equal with input_dims[0], "
                            "input_dims[0] must be 1 but got %s",
                            input_dims[0]));
      PADDLE_ENFORCE_EQ(
          y_dims[1] == input_dims[1] || input_dims[1] == 1, true,
          platform::errors::InvalidArgument(
              "The input tensor shape mismatch, input shape=[%s], "
              "x shape=[%s], y shape=[%s]",
              input_dims, x_dims, y_dims));
    }
    // broadcast mode check
    if (y_dims[1] != input_dims[1]) {
      PADDLE_ENFORCE_EQ(input_dims[1], 1,
                        platform::errors::InvalidArgument(
                            "When y_dims[1] is not equal with input_dims[0], "
                            "input_dims[0] must be 1 but got %s",
                            input_dims[1]));
      PADDLE_ENFORCE_EQ(
          x_dims[0] == input_dims[0] || input_dims[0] == 1, true,
          platform::errors::InvalidArgument(
              "The input tensor shape mismatch, input shape=[%s], "
              "x shape=[%s], y shape=[%s]",
              input_dims, x_dims, y_dims));
    }
    // broadcast mode check
    PADDLE_ENFORCE_EQ(
        x_dims[1], y_dims[0],
        platform::errors::InvalidArgument(
            "The input tensor X's width must be equal with matrix Y' height. "
            "But received X's shape = [%s], Y's shape = [%s].",
            x_dims[1], y_dims[0]));

    auto* out = context.Output<Tensor>("Out");
    out->mutable_data<T>({x_dims[0], y_dims[1]}, context.GetPlace());

    float alpha = context.template Attr<float>("Alpha");
    float beta = context.template Attr<float>("Beta");

    auto blas = math::GetBlas<DeviceContext, T>(context);

    // calc broadcast dim
    Array2 bcast_dims;
    bcast_dims[0] = x_dims[0] / input_dims[0];
    bcast_dims[1] = y_dims[1] / input_dims[1];
    VLOG(3) << "bcast_dims=[" << bcast_dims[0] << "," << bcast_dims[1] << "]";
    // broadcast using eigen
    auto eigen_input = EigenTensor<T, 2>::From(*input);
    auto eigen_out = EigenTensor<T, 2>::From(*out);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    EigenBroadcast<std::decay_t<decltype(place)>, T, 2>::Eval(
        place, eigen_out, eigen_input, bcast_dims);

    blas.GEMM(false, false, x_dims[0], y_dims[1], x_dims[1], alpha,
              x->data<T>(), x_dims[1], y->data<T>(), y_dims[1], beta,
              out->data<T>(), y_dims[1]);
  }
};

template <typename DeviceContext, typename T>
class AddMMGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* dout = ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto in_dims = ctx.Input<framework::LoDTensor>("Input")->dims();
    auto* dinput =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("Input"));
    auto* dx = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<framework::LoDTensor>(framework::GradVarName("Y"));

    float alpha = ctx.Attr<float>("Alpha");
    float beta = ctx.Attr<float>("Beta");

    int total_elems = 0;

    VLOG(3) << "alpha: " << alpha << " beta: " << beta;

    if (dinput != nullptr) {
      dinput->set_lod(dout->lod());
    }
    if (dx != nullptr) {
      dx->set_lod(x->lod());
    }
    if (dy != nullptr) {
      dy->set_lod(y->lod());
    }

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    if (dinput) {
      dinput->mutable_data<T>(ctx.GetPlace());
      total_elems = in_dims[0] * in_dims[1];
      auto& place =
          *ctx.template device_context<DeviceContext>().eigen_device();
      auto eigen_dout = EigenTensor<T, 2>::From(*dout);
      auto eigen_dinput = EigenTensor<T, 2>::From(*dinput);

      bool row_compress = in_dims[0] != dout->dims()[0];
      bool col_compress = in_dims[1] != dout->dims()[1];
      auto eigen_dinput_shape = Array2(dinput->dims()[0], dinput->dims()[1]);

      if (row_compress && col_compress) {
        eigen_dinput.device(place) =
            eigen_dout.sum().eval().reshape(eigen_dinput_shape);
      } else if (row_compress) {
        eigen_dinput.device(place) =
            eigen_dout.sum(Array1(0)).eval().reshape(eigen_dinput_shape);
      } else if (col_compress) {
        eigen_dinput.device(place) =
            eigen_dout.sum(Array1(1)).eval().reshape(eigen_dinput_shape);
      } else {
        blas.VCOPY(total_elems, dout->data<T>(), dinput->data<T>());
      }

      blas.SCAL(total_elems, beta, dinput->data<T>());
    }
    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
      total_elems = x->dims()[0] * x->dims()[1];
      // dx = dout * y'. dx: M x K, dout : M x N, y : K x N
      blas.MatMul(*dout, false, *y, true, dx);
      blas.SCAL(total_elems, alpha, dx->data<T>());
    }
    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());
      total_elems = x->dims()[1] * y->dims()[1];
      // dy = x' * dout. dy K x N, dout : M x N, x : M x K
      blas.MatMul(*x, true, *dout, false, dy);
      blas.SCAL(total_elems, alpha, dy->data<T>());
    }
  }
};

}  // namespace operators
}  // namespace paddle
