
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
class ReshapeKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* out = ctx.Output<Tensor>("Out");
    auto* in = ctx.Input<Tensor>("X");
    out->mutable_data<T>(ctx.GetPlace());

    auto shape = ctx.Attr<std::vector<int>>("shape");
    std::vector<int64_t> tmp;
    for (auto dim : shape) {
      tmp.push_back(dim);
    }
    auto out_dims = framework::make_ddim(tmp);
    out->CopyFrom<T>(*in, ctx.GetPlace());
    out->Resize(out_dims);
  }
};

template <typename Place, typename T>
class ReshapeGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* d_out = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    d_x->mutable_data<T>(ctx.GetPlace());

    auto in_dims = d_x->dims();
    d_x->CopyFrom<T>(*d_out, ctx.GetPlace());
    d_x->Resize(in_dims);
  }
};
}
}
