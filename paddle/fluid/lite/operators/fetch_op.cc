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

class FetchOp : public OpLite {
 public:
  explicit FetchOp(const std::string& type) : OpLite(type) {}

  bool CheckShape() const override {
    CHECK_OR_FALSE(param_.input);
    CHECK_OR_FALSE(param_.fetch_list);
    return true;
  }

  bool InferShape() const override { return true; }
  void AttachKernel(KernelBase* kernel) override { kernel->SetParam(param_); }

 protected:
  bool AttachImpl(const OpDesc& opdesc, lite::Scope* scope) override {
    auto _x = opdesc.Input("X").front();
    auto* x = scope->FindVar(_x);
    CHECK(x);
    param_.input = &x->Get<Tensor>();

    auto _out = opdesc.Output("Out").front();
    auto* out = scope->FindVar(_out);
    param_.fetch_list = out->GetMutable<std::vector<lite::Tensor>>();

    param_.col = opdesc.GetAttr("col").get<int>();
    return true;
  }

  std::string DebugString() const override { return "fetch"; }

 private:
  mutable FetchParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(fetch, paddle::lite::operators::FetchOp);
