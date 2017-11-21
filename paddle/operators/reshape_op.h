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
class ReshapeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* out = ctx.Output<framework::Tensor>("Out");
    auto* in = ctx.Input<framework::Tensor>("X");
    auto out_dims = out->dims();
    out->mutable_data<T>(ctx.GetPlace());
    out->CopyFrom(*in, ctx.GetPlace(), ctx.device_context());
    out->Resize(out_dims);
  }
};

template <typename Place, typename T>
class ReshapeGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* d_out = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* d_x = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    d_x->mutable_data<T>(ctx.GetPlace());

    auto in_dims = d_x->dims();
    d_x->CopyFrom(*d_out, ctx.GetPlace(), ctx.device_context());
    d_x->Resize(in_dims);
  }
};
}  // namespace operators
}  // namespace paddle
