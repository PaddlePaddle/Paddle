// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

class FeedOp : public OpLite {
 public:
  explicit FeedOp(const std::string& type) : OpLite(type) {}

  bool CheckShape() const override {
    CHECK_OR_FALSE(param_.feed_list);
    CHECK_OR_FALSE(param_.out);
    return true;
  }

  bool InferShape() const override { return true; }

 protected:
  void AttachKernel(KernelBase* kernel) override { kernel->SetParam(param_); }

 protected:
  bool AttachImpl(const OpDesc& opdesc, lite::Scope* scope) override {
    auto feed_var_name = opdesc.Input("X").front();
    auto* feed_var = scope->FindVar(feed_var_name);
    CHECK(feed_var);
    auto& feed_tensor_list = feed_var->Get<std::vector<Tensor>>();
    param_.feed_list = &feed_tensor_list;

    auto out_name = opdesc.Output("Out").front();
    auto* out_var = scope->FindVar(out_name);
    CHECK(out_var);
    param_.out = out_var->GetMutable<Tensor>();

    // NOTE need boost here
    // TODO(Superjomn) drop the need of framework::op_desc
    param_.col = opdesc.GetAttr("col").get<int>();
    return true;
  }

  std::string DebugString() const override { return "feed"; }

 private:
  mutable FeedParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(feed, paddle::lite::operators::FeedOp);
