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
    std::cout << one_img_size << one_sample_size << std::endl;

    Tensor* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    int n = ctx.Attr<int>("n");
    float alpha = ctx.Attr<float>("alpha");
    float beta = ctx.Attr<float>("beta");
    float k = ctx.Attr<float>("k");

    PADDLE_ENFORCE(n >= 0, "n should >= 0");
    PADDLE_ENFORCE(alpha >= 0.0, "alpha should >= 0");
    PADDLE_ENFORCE(beta >= 0.0, "beta should >= 0");
    PADDLE_ENFORCE(k >= 0.0, "k should >= 0");

    // f(x) = x * ( k + alpha * SUM((x)^2) )^(-beta)
    // x represents inputs
    // f(x) represents outputs
    // mid save the intermediate result for backward
    auto x_v = framework::EigenVector<T>::Flatten(*x);
    Eigen::Tensor<float, 1> mid(x_v.size());
    mid.setConstant(k);
    const T* data = x_v.data();

    const int start = -(n - 1) / 2;
    const int end = start + n;
    for (int m = 0; m < N; m++) {
      const T* sample = data + m * one_sample_size;
      for (int i = 0; i < C; i++) {
        auto mid_data = mid.data() + m * one_sample_size + i * one_img_size;
        auto s = framework::EigenTensor<float, 1>::From(
            mid_data, framework::make_ddim({one_img_size}));
        for (int c = start; c <= end; c++) {
          int cur_channel = i + c;
          if (cur_channel >= 0 && cur_channel < N) {
            auto input = framework::EigenTensor<T, 1>::From(
                (sample + cur_channel * one_img_size),
                framework::make_ddim({one_img_size}));
            s += input.square() * alpha;
          }
        }
      }
    }

    auto out_e = framework::EigenVector<T>::Flatten(*out);
    out_e.device(ctx.GetEigenDevice<Place>()) = x_v * mid.pow(-beta);
    /*
    Eigen::Tensor<float, 4> mid(N, C, H, W);
    mid.setConstant(k);

    auto x_e = framework::EigenTensor<T, 4>::From(*x);

    const int start = -(n - 1) / 2;
    const int end = start + n;
    std::cout << start << end << std::endl;
    for (int m = 0; m < N; m++) {
      for (int i = 0; i < C; i++) {
        Eigen::array<int, 4> offsets{{m, i, 0, 0}};
        Eigen::array<int, 4> extents{{0, 0, H, W}};
        Eigen::Tensor<float, 1> s = mid.slice(offsets, extents);
        std::cout << &mid << &offsets << &extents << &s << std::endl;
        //std::cout << s << std::endl;
        for (int c = start; c <= end; c++) {
          int cur_channel = i + c;
          if (cur_channel >= 0 && cur_channel < N) {
            Eigen::array<int, 4> offsets{{m, cur_channel, 0, 0}};
            Eigen::array<int, 4> extents{{0, 0, H, W}};
            auto input = x_e.slice(offsets, extents);
            s += input.square() * alpha;
          }
        }
      }
    }

    auto out_e = framework::EigenVector<T>::Flatten(*out);
    out_e.device(ctx.GetEigenDevice<Place>()) =
        x_e.reshape(Eigen::DSizes<T, 1>(x_e.size())) *
        mid.reshape(Eigen::DSizes<float, 1>(mid.size())).pow(-beta);

    Eigen::Tensor<int, 2> a(4, 3);
    a.setValues({{0, 100, 200}, {300, 400, 500},
            {600, 700, 800}, {900, 1000, 1100}});
    Eigen::array<int, 2> offsets{{1, 0}};
    Eigen::array<int, 2> extents{{2, 2}};
    Eigen::Tensor<int, 1> slice = a.slice(offsets, extents);
    std::cout << "a" << std::endl << a << std::endl;
    */
  }
};

template <typename Place, typename T>
class LRNGradKernel : public framework::OpKernel {
 public:
  using Tensor = framework::Tensor;
  void Compute(const framework::ExecutionContext& ctx) const override {
    // const Tensor* x = ctx.Input<Tensor>("X");
  }
};

}  // namespace operators
}  // namespace paddle
