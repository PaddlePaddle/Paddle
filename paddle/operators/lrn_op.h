/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   You may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class LRNKernel : public framework::OpKernel<T> {
 public:
  using Tensor = framework::Tensor;

  // f(x) = x * ( k + alpha * SUM((x)^2) )^(-beta)
  // x represents inputs
  // f(x) represents outputs
  void Compute(const framework::ExecutionContext& ctx) const override {
    // input
    const Tensor* x = ctx.Input<Tensor>("X");
    auto x_dims = x->dims();

    // NCHW
    int N = x_dims[0];
    int C = x_dims[1];
    int H = x_dims[2];
    int W = x_dims[3];

    Tensor* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    // MidOut save the intermediate result for backward
    Tensor* mid = ctx.Output<Tensor>("MidOut");
    mid->mutable_data<T>(ctx.GetPlace());

    int n = ctx.Attr<int>("n");
    T alpha = ctx.Attr<float>("alpha");
    T beta = ctx.Attr<float>("beta");
    T k = ctx.Attr<float>("k");

    PADDLE_ENFORCE(n > 0, "n should >= 0");
    PADDLE_ENFORCE(alpha >= 0.0, "alpha should >= 0.0");
    PADDLE_ENFORCE(beta >= 0.0, "beta should >= 0.0");
    PADDLE_ENFORCE(k >= 0.0, "k should >= 0.0");

    auto x_v = framework::EigenVector<T>::Flatten(*x);

    const int start = -(n - 1) / 2;
    const int end = start + n;

    auto e_mid = framework::EigenTensor<T, 4>::From(*mid);
    e_mid.device(ctx.GetEigenDevice<Place>()) = e_mid.constant(k);

    auto e_x = framework::EigenTensor<T, 4>::From(*x);
    for (int m = 0; m < N; m++) {
      for (int i = 0; i < C; i++) {
        for (int c = start; c <= end; c++) {
          int ch = i + c;
          if (ch >= 0 && ch < C) {
            auto s = e_mid.slice(Eigen::array<int, 4>({{m, i, 0, 0}}),
                                 Eigen::array<int, 4>({{1, 1, H, W}}));

            auto r = e_x.slice(Eigen::array<int, 4>({{m, ch, 0, 0}}),
                               Eigen::array<int, 4>({{1, 1, H, W}}));

            s.device(ctx.GetEigenDevice<Place>()) += alpha * r.square();
          }
        }
      }
    }

    auto out_e = framework::EigenVector<T>::Flatten(*out);
    out_e.device(ctx.GetEigenDevice<Place>()) =
        x_v * e_mid.reshape(Eigen::DSizes<int, 1>(e_mid.size())).pow(-beta);
  }
};

/**
 * \brief Backward calculation for normalization with across maps.
 *
 * Function implementation:
 *
 * The implementation of this Function is derived from the
 * CrossMapNormalFunc implementation.
 *
 * InputGrad = OutputGrad * denoms ^ (-beta)
 *    -- upper
 *  + > (OutputGrad * OutputValue * (-2 * alpha * beta) / MidOut) * InputValue
 *    -- lower
 *
 * The data of inputs/outputs format is the same as the forward interface
 * and is NCHW.
 *
 * The upper and lower is the same as forward. The logic of the sum
 * is also the same as forward.
 */
template <typename Place, typename T>
class LRNGradKernel : public framework::OpKernel<T> {
 public:
  using Tensor = framework::Tensor;
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* x = ctx.Input<Tensor>("X");
    const Tensor* out = ctx.Input<Tensor>("Out");
    const Tensor* out_g = ctx.Input<Tensor>(framework::GradVarName("Out"));
    const Tensor* mid = ctx.Input<Tensor>("MidOut");

    auto x_g = ctx.Output<Tensor>(framework::GradVarName("X"));
    x_g->mutable_data<T>(ctx.GetPlace());

    auto x_g_e = framework::EigenVector<T>::Flatten(*x_g);
    x_g_e.device(ctx.GetEigenDevice<Place>()) = x_g_e.constant(0.0);

    auto x_dims = x->dims();
    int N = x_dims[0];
    int C = x_dims[1];
    int H = x_dims[2];
    int W = x_dims[3];

    int n = ctx.Attr<int>("n");
    T alpha = ctx.Attr<T>("alpha");
    T beta = ctx.Attr<T>("beta");
    T ratio = -2 * alpha * beta;

    auto e_x = framework::EigenTensor<T, 4>::From(*x);
    auto e_x_g = framework::EigenTensor<T, 4>::From(*x_g);
    auto e_out = framework::EigenTensor<T, 4>::From(*out);
    auto e_out_g = framework::EigenTensor<T, 4>::From(*out_g);
    auto e_mid = framework::EigenTensor<T, 4>::From(*mid);

    const int start = -(n - 1) / 2;
    const int end = start + n;
    for (int m = 0; m < N; m++) {
      for (int i = 0; i < C; i++) {
        auto i_x = e_x.slice(Eigen::array<int, 4>({{m, i, 0, 0}}),
                             Eigen::array<int, 4>({{1, 1, H, W}}));

        auto i_x_g = e_x_g.slice(Eigen::array<int, 4>({{m, i, 0, 0}}),
                                 Eigen::array<int, 4>({{1, 1, H, W}}));

        auto i_out_g = e_out_g.slice(Eigen::array<int, 4>({{m, i, 0, 0}}),
                                     Eigen::array<int, 4>({{1, 1, H, W}}));

        auto i_mid = e_mid.slice(Eigen::array<int, 4>({{m, i, 0, 0}}),
                                 Eigen::array<int, 4>({{1, 1, H, W}}));

        i_x_g.device(ctx.GetEigenDevice<Place>()) = i_mid.pow(-beta) * i_out_g;
        for (int c = start; c <= end; c++) {
          int ch = i + c;
          if (ch < 0 || ch >= C) {
            continue;
          }

          auto c_out = e_out.slice(Eigen::array<int, 4>({{m, ch, 0, 0}}),
                                   Eigen::array<int, 4>({{1, 1, H, W}}));

          auto c_mid = e_mid.slice(Eigen::array<int, 4>({{m, ch, 0, 0}}),
                                   Eigen::array<int, 4>({{1, 1, H, W}}));

          auto c_out_g = e_out_g.slice(Eigen::array<int, 4>({{m, ch, 0, 0}}),
                                       Eigen::array<int, 4>({{1, 1, H, W}}));

          i_x_g.device(ctx.GetEigenDevice<Place>()) +=
              ratio * c_out_g * c_out * i_x / c_mid;
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
