/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class GraphSampleOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    /*auto* src = ctx.Input<Tensor>("Src");
    auto* dst_count = ctx.Input<Tensor>("Dst_Count");
    auto* vertices = ctx.Input<Tensor>("X");
    auto* sample_sizes = ctx.Input<Tensor>("Sample_Sizes");

    auto* sample_index = ctx.Output<Tensor>("Sample_index");
    auto* out_src = ctx.Output<Tensor>("Out_Src");
    auto* out_dst = ctx.Output<Tensor>("Out_Dst");*/
  }
};

}  // namespace operators
}  // namespace paddle
