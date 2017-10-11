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

template <typename T>
class FetchKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Tensor* input = ctx.Input<framework::Tensor>("Input");
    framework::Variable* g_fetch_variable =
        framework::GetGlobalScope().FindVar("fetch_value");
    auto* tensors =
        g_fetch_variable->GetMutable<std::vector<framework::Tensor>>();
    int col = ctx.template Attr<int>("col");
    if (tensors->size() < static_cast<size_t>(col + 1)) {
      tensors->resize(col + 1);
    }
    PADDLE_ENFORCE_GT(tensors->size(), static_cast<size_t>(col));
    (*tensors)[col].Resize(input->dims());
    (*tensors)[col].mutable_data<T>(platform::CPUPlace());
    (*tensors)[col].CopyFrom<T>(*input, platform::CPUPlace());
    // TODO(qijun): need to handle LodTensor later
  }
};

}  // namespace operators
}  // namespace paddle
