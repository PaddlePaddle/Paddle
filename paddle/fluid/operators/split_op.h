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

#include <chrono>  // NOLINT
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/strided_memcpy.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SplitOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto outs = ctx.MultiOutput<framework::Tensor>("Out");
    int axis = ctx.Attr<int>("axis");
    auto place = ctx.GetPlace();

    std::vector<const framework::Tensor*> shape_refer;
    for (size_t j = 0; j < outs.size(); ++j) {
      outs[j]->mutable_data<T>(ctx.GetPlace());
      shape_refer.emplace_back(outs[j]);
    }

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    // Sometimes direct copies will be faster, this maybe need deeply analysis.
    if (axis == 0 && outs.size() < 10) {
      StridedMemcpyWithAxis0<T>(dev_ctx, *in, shape_refer, &outs);
    } else {
      framework::Tensor* out_info = ctx.Output<framework::Tensor>("OutInfo");

      math::SplitFunctor<DeviceContext, T> functor;
      functor(dev_ctx, *in, shape_refer, axis, &outs, out_info);
    }
  }
};

class SplitGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto op = new framework::OpDesc();
    op->SetType("concat");
    op->SetInput("X", OutputGrad("Out"));
    op->SetOutput("Out", InputGrad("X"));
    op->SetOutput("XInfo", Output("OutInfo"));
    op->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(op);
  }
};

}  // namespace operators
}  // namespace paddle
