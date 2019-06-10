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
#include <string>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/sequence_pooling.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class SequencePoolKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<Tensor>("Out");
    std::string pooltype = context.Attr<std::string>("pooltype");
    T pad_value = static_cast<T>(context.Attr<float>("pad_value"));

    auto dims = in->dims();
    auto lod = in->lod();
    // InferShape by lod
    PADDLE_ENFORCE_EQ(lod.size(), 1UL, "Only support one level sequence now.");
    PADDLE_ENFORCE_GE(
        dims[0],
        /*batch size = */ static_cast<int64_t>(lod[0].size() - 1),
        "The first dimension of Input(X) must be large than batch size.");
    dims[0] = lod[0].size() - 1;
    out->Resize({dims});
    out->mutable_data<T>(context.GetPlace());
    Tensor* index = nullptr;

    const bool is_test = context.Attr<bool>("is_test");

    // Do not create index buffer for inference (is_test) mode
    // TODO(jczaja): Skip index buffer creation for other devices eg. GPU
    if (pooltype == "MAX" &&
        (is_test == false ||
         platform::is_cpu_place(context.GetPlace()) == false)) {
      index = context.Output<Tensor>("MaxIndex");
      index->Resize({dims});
      index->mutable_data<int>(context.GetPlace());
    }
    math::SequencePoolFunctor<DeviceContext, T> pool;
    pool(context.template device_context<DeviceContext>(), pooltype, pad_value,
         *in, out, is_test, index);
  }
};

template <typename DeviceContext, typename T>
class SequencePoolGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out_g = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* in_g = context.Output<LoDTensor>(framework::GradVarName("X"));
    std::string pooltype = context.Attr<std::string>("pooltype");
    const Tensor* index = nullptr;
    if (pooltype == "MAX") {
      index = context.Input<Tensor>("MaxIndex");
    }
    in_g->mutable_data<T>(context.GetPlace());
    math::SequencePoolGradFunctor<DeviceContext, T> pool;
    pool(context.template device_context<DeviceContext>(), pooltype, *out_g,
         in_g, index);
  }
};

}  // namespace operators
}  // namespace paddle
