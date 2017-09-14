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

template <typename Place, typename T>
class RankLossKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* out = ctx.Output<framework::Tensor>("Out");
    auto* p_t = ctx.Input<framework::Tensor>("P");
    auto* oi_t = ctx.Input<framework::Tensor>("Oi");
    auto* oj_t = ctx.Input<framework::Tensor>("Oj");
    out->mutable_data<T>(ctx.GetPlace());

    auto& dev = ctx.GetEigenDevice<Place>();
    auto out_eig = framework::EigenVector<T>::Flatten(*out);
    auto p_eig = framework::EigenVector<T>::Flatten(*p_t);
    auto oi_eig = framework::EigenVector<T>::Flatten(*oi_t);
    auto oj_eig = framework::EigenVector<T>::Flatten(*oj_t);

    framework::Tensor o_t;
    o_t.Resize(oi_t->dims());
    o_t.mutable_data<T>(ctx.GetPlace());
    auto o_eig = framework::EigenVector<T>::Flatten(o_t);
    o_eig.device(dev) = oi_eig - oj_eig;

    out_eig.device(dev) = (1. + (o_eig).exp()).log() - p_eig * o_eig;
  }
};

template <typename Place, typename T>
class RankLossGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* d_oi = ctx.Output<framework::Tensor>(framework::GradVarName("Oi"));
    auto* d_oj = ctx.Output<framework::Tensor>(framework::GradVarName("Oj"));
    auto* d_p = ctx.Output<framework::Tensor>(framework::GradVarName("P"));

    auto* d_out = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* p_t = ctx.Input<framework::Tensor>("P");
    auto* oi_t = ctx.Input<framework::Tensor>("Oi");
    auto* oj_t = ctx.Input<framework::Tensor>("Oj");

    d_oi->mutable_data<T>(ctx.GetPlace());
    d_oj->mutable_data<T>(ctx.GetPlace());
    d_p->mutable_data<T>(ctx.GetPlace());

    auto& dev = ctx.GetEigenDevice<Place>();
    auto d_out_eig = framework::EigenVector<T>::Flatten(*d_out);
    auto p_eig = framework::EigenVector<T>::Flatten(*p_t);
    auto oi_eig = framework::EigenVector<T>::Flatten(*oi_t);
    auto oj_eig = framework::EigenVector<T>::Flatten(*oj_t);

    auto d_oi_eig = framework::EigenVector<T>::Flatten(*d_oi);
    auto d_oj_eig = framework::EigenVector<T>::Flatten(*d_oj);

    framework::Tensor o_t;
    o_t.Resize(oi_t->dims());
    o_t.mutable_data<T>(ctx.GetPlace());
    auto o_eig = framework::EigenVector<T>::Flatten(o_t);
    o_eig.device(dev) = oi_eig - oj_eig;

    // dOi & dOj
    d_oi_eig.device(dev) =
        d_out_eig * (o_eig.exp() / (1. + o_eig.exp()) - p_eig);
    d_oj_eig.device(dev) = -d_oi_eig;
    // dP
    framework::EigenVector<T>::Flatten(*d_p).device(dev) = -o_eig;
  }
};
}  // namespace operators
}  // namespace paddle
