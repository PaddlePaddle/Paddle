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
class FeedKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    framework::Tensor* out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    framework::Variable* g_feed_variable =
        framework::GetGlobalScope().FindVar("feed_value");
    const auto& tensors =
        g_feed_variable->Get<std::vector<framework::Tensor>>();
    int col = ctx.template Attr<int>("col");
    PADDLE_ENFORCE_GT(tensors.size(), static_cast<size_t>(col));
    // TODO(qijun):
    //   check tensors[col].dims() with attribute,
    //   except the first dimenson.
    out->CopyFrom<T>(tensors[col], ctx.GetPlace(), ctx.device_context());
  }
};

}  // namespace operators
}  // namespace paddle
