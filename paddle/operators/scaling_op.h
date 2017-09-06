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

#pragma once

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename Place, typename T>
class ScalingKernel : public framework::OpKernel {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto* out = ctx.Output<Tensor>("Out");
    auto* in = ctx.Input<Tensor>("X");
    auto* weight = ctx.Input<Tensor>("weight");
    out->mutable_data<T>(in->place());

    auto rows = in->dims()[0];
    auto cols = in->dims()[1];

    auto col_vec_dims = framework::make_ddim({rows, 1});
    auto bd_cast_dims = framework::make_ddim({1, cols});

    auto eigen_out = framework::EigenMatrix<T>::From(*out);
    auto eigen_in = framework::EigenMatrix<T>::From(*in);
    auto eigen_weight = framework::EigenMatrix<T>::From(*weight, col_vec_dims);

    auto& dev = ctx.GetEigenDevice<Place>();
    eigen_out.device(dev) = eigen_in * eigen_weight.broadcast(bd_cast_dims);
  }
};

template <typename Place, typename T>
class ScalingGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* d_out = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* d_in = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* d_weight = ctx.Output<Tensor>(framework::GradVarName("weight"));

    auto* in = ctx.Input<Tensor>("X");
    auto* weight = ctx.Input<Tensor>("weight");

    d_in->mutable_data<T>(ctx.GetPlace());
    d_weight->mutable_data<T>(ctx.GetPlace());

    auto rows = d_out->dims()[0];
    auto cols = d_out->dims()[1];
    auto col_vec_dims = framework::make_ddim({rows, 1});
    auto bd_cast_dims = framework::make_ddim({1, cols});

    auto eigen_in = framework::EigenMatrix<T>::From(*in);
    auto eigen_weight = framework::EigenMatrix<T>::From(*weight, col_vec_dims);

    auto eigen_d_out = framework::EigenMatrix<T>::From(*d_out);
    auto eigen_d_in = framework::EigenMatrix<T>::From(*d_in);
    auto eigen_d_weight = framework::EigenVector<T>::From(*d_weight);

    auto& dev = ctx.GetEigenDevice<Place>();
    // dX = dOut * weight.broadcast()
    eigen_d_in.device(dev) = eigen_d_out * eigen_weight.broadcast(bd_cast_dims);

    Eigen::array<int, 1> dims{{1}};
    // d_weight = dOut * X, reduce to one column
    eigen_d_weight.device(dev) = (eigen_d_out * eigen_in).sum(dims);
  }
};
}  // namespace operators
}  // namespace paddle
