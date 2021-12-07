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
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {
static inline framework::DDim UnbindOutsDims(const framework::DDim in_dims,
                                             int axis) {
  std::vector<int> out_dims;
  axis = axis < 0 ? in_dims.size() + axis : axis;
  for (int i = 0; i < in_dims.size(); i++) {
    if (i != axis) out_dims.push_back(in_dims[i]);
  }
  return framework::make_ddim(out_dims);
}
template <typename DeviceContext, typename T>
class UnbindOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto outs = ctx.MultiOutput<framework::Tensor>("Out");
    int axis = ctx.Attr<int>("axis");

    auto in_dims = in->dims();

    auto place = ctx.GetPlace();

    axis = axis < 0 ? in_dims.size() + axis : axis;
    std::vector<const framework::Tensor*> shape_refer;
    for (size_t j = 0; j < outs.size(); ++j) {
      outs[j]->mutable_data<T>(ctx.GetPlace());
      shape_refer.emplace_back(outs[j]);
    }

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    math::SplitFunctor<DeviceContext, T> functor;
    functor(dev_ctx, *in, shape_refer, axis, &outs);
  }
};

template <typename T>
class UnbindGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("stack");
    op->SetInput("X", this->OutputGrad("Out"));
    op->SetOutput("Y", this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle
