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
class LRNKernel : public framework::OpKernel {
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
    int one_img_size = H * W;
    int one_sample_size = C * one_img_size;
    // printf("N:%d C:%d H:%d W:%d\n", N, C, H, W);

    Tensor* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    // mid_out save the intermediate result for backward
    Tensor* mid_out = ctx.Output<Tensor>("mid_out");
    mid_out->mutable_data<float>(ctx.GetPlace());

    int n = ctx.Attr<int>("n");
    float alpha = ctx.Attr<float>("alpha");
    float beta = ctx.Attr<float>("beta");
    float k = ctx.Attr<float>("k");
    // printf("n:%d alpha:%f beta:%f k:%f\n", n, alpha, beta, k);

    PADDLE_ENFORCE(n > 0, "n should >= 0");
    PADDLE_ENFORCE(alpha >= 0.0, "alpha should >= 0.0");
    PADDLE_ENFORCE(beta >= 0.0, "beta should >= 0.0");
    PADDLE_ENFORCE(k >= 0.0, "k should >= 0.0");

    auto mid_out_v = framework::EigenVector<T>::Flatten(*mid_out);
    mid_out_v.setConstant(k);

    auto x_v = framework::EigenVector<T>::Flatten(*x);
    const T* data = x_v.data();

    const int start = -(n - 1) / 2;
    const int end = start + n;
    // std::cout << "start:" << start << "\tend:" << end << std::endl;
    for (int m = 0; m < N; m++) {
      const T* sample = data + m * one_sample_size;

      for (int i = 0; i < C; i++) {
        auto mid_data =
            mid_out_v.data() + m * one_sample_size + i * one_img_size;
        framework::EigenTensor<float, 1>::Type s =
            framework::EigenTensor<float, 1>::From(
                mid_data, framework::make_ddim({one_img_size}));

        for (int c = start; c <= end; c++) {
          int ch = i + c;
          if (ch >= 0 && ch < C) {
            // printf("m:%d i:%d ch:%d\n", m, i, ch);
            auto input = framework::EigenTensor<T, 1>::From(
                (sample + ch * one_img_size),
                framework::make_ddim({one_img_size}));
            s += (input.square() * alpha);
          }
        }
      }
    }

    auto out_e = framework::EigenVector<T>::Flatten(*out);
    out_e.device(ctx.GetEigenDevice<Place>()) = x_v * mid_out_v.pow(-beta);
  }
};

template <typename T>
inline void detail(std::string name, T t) {
  printf("in_c:%s %f %f %f %f %f\n", name.c_str(), t(0), t(1), t(2), t(3),
         t(4));
}

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
 *  + > (OutputGrad * OutputValue * (-2 * alpha * beta) / mid_out) * InputValue
 *    -- lower
 *
 * The data of inputs/outputs format is the same as the forward interface
 * and is NCHW.
 *
 * The upper and lower is the same as forward. The logic of the sum
 * is also the same as forward.
 */
template <typename Place, typename T>
class LRNGradKernel : public framework::OpKernel {
 public:
  using Tensor = framework::Tensor;
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* x = ctx.Input<Tensor>("X");
    const Tensor* out = ctx.Input<Tensor>("Out");
    const Tensor* out_g = ctx.Input<Tensor>(framework::GradVarName("Out"));
    const Tensor* mid = ctx.Input<Tensor>("mid_out");

    auto x_g = ctx.Output<Tensor>(framework::GradVarName("X"));
    x_g->mutable_data<T>(ctx.GetPlace());
    auto x_g_e = framework::EigenVector<float>::Flatten(*x_g);
    x_g_e.setConstant(0.0);

    auto x_dims = x->dims();
    int N = x_dims[0];
    int C = x_dims[1];
    int H = x_dims[2];
    int W = x_dims[3];
    // printf("N:%d C:%d H:%d W:%d\n", N, C, H, W);

    int n = ctx.Attr<int>("n");
    float alpha = ctx.Attr<float>("alpha");
    float beta = ctx.Attr<float>("beta");
    // float k = ctx.Attr<float>("k");
    // printf("n:%d alpha:%f beta:%f k:%f\n", n, alpha, beta, k);

    auto x_e = framework::EigenVector<float>::Flatten(*x);
    auto out_e = framework::EigenVector<float>::Flatten(*out);
    auto out_g_e = framework::EigenVector<float>::Flatten(*out_g);
    auto mid_e = framework::EigenVector<float>::Flatten(*mid);

    /*
    detail("x_e", x_e);
    detail("out_e", out_e);
    detail("out_g_e", out_g_e);
    detail("mid_e", mid_e);
    */

    int one_img_size = H * W;
    int one_sample_size = C * one_img_size;

    float ratio = -2 * alpha * beta;

    const int start = -(n - 1) / 2;
    const int end = start + n;
    for (int m = 0; m < N; m++) {
      int m_pos = m * one_sample_size;
      for (int i = 0; i < C; i++) {
        int i_pos = m_pos + i * one_img_size;

        framework::EigenTensor<float, 1>::ConstType i_x =
            framework::EigenTensor<float, 1>::From(
                x_e.data() + i_pos, framework::make_ddim({one_img_size}));

        framework::EigenTensor<float, 1>::Type i_x_g =
            framework::EigenTensor<float, 1>::From(
                x_g_e.data() + i_pos, framework::make_ddim({one_img_size}));

        framework::EigenTensor<float, 1>::ConstType i_out_g =
            framework::EigenTensor<float, 1>::From(
                out_g_e.data() + i_pos, framework::make_ddim({one_img_size}));

        /*
        framework::EigenTensor<float, 1>::ConstType i_out =
            framework::EigenTensor<float, 1>::From(
                out_e.data() + i_pos, framework::make_ddim({one_img_size}));
                */

        framework::EigenTensor<float, 1>::ConstType i_mid =
            framework::EigenTensor<float, 1>::From(
                mid_e.data() + i_pos, framework::make_ddim({one_img_size}));

        // std::cout << i_x_g(0) << std::endl;
        i_x_g = i_mid.pow(-beta) * i_out_g;
        for (int c = start; c <= end; c++) {
          int ch = i + c;
          if (ch < 0 || ch >= C) {
            continue;
          }

          // printf("ch:%d\n",ch);
          int c_pos = m_pos + ch * one_img_size;
          auto c_out = framework::EigenTensor<float, 1>::From(
              out_e.data() + c_pos, framework::make_ddim({one_img_size}));

          auto c_mid = framework::EigenTensor<float, 1>::From(
              mid_e.data() + c_pos, framework::make_ddim({one_img_size}));

          auto c_out_g = framework::EigenTensor<float, 1>::From(
              out_g_e.data() + c_pos, framework::make_ddim({one_img_size}));

          /*
          if (ch == i ){
            i_x_g += i_mid.pow(-beta) * i_out_g
                +  ratio * i_out_g * i_out * i_x / i_mid;
            detail("i_mid", i_mid);
            detail("i_out_g", i_out_g);
            detail("i_out", i_out);
            detail("i_x", i_x);
            detail("i_x_g", i_x_g);
            continue;
          }

        framework::EigenTensor<float, 1>::ConstType c_x =
            framework::EigenTensor<float, 1>::From(
                x_e.data() + c_pos, framework::make_ddim({one_img_size}));
        */

          i_x_g += ratio * c_out_g * c_out * i_x / c_mid;
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
