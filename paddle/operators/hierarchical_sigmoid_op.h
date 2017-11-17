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
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/matrix_bit_code.h"

namespace paddle {
namespace operators {
template <typename Place, typename T>

class HierarchicalSigmoidOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    auto* label = ctx.Input<framework::Tensor>("Label");
    auto* bias = ctx.Input<framework::Tensor>("Bias");
    size_t num_classes = static_cast<size_t>(ctx.Attr<int>("num_classes"));
    int64_t batch_size = ins[0]->dims()[0];
    int64_t size = ins.size();
    framework::Tensor pre_out;
    std::vector<int64_t> pre_out_dims({batch_size, size});
    pre_out.mutable_data<T>(framework::make_ddim(pre_out_dims), ctx.GetPlace());

    if (bias != NULL) {
      math::AddByBitCode<T>(num_classes, *label, pre_out, *bias);
    }
  }
};

template <typename Place, typename T>
class HierarchicalSigmoidGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

}  // namespace operators
}  // namespace paddle
