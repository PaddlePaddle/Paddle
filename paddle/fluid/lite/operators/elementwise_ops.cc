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

class ElementwiseOp : public OpLite {
 public:
  explicit ElementwiseOp(const std::string& type) : OpLite(type) {}

  bool CheckShape() const override {
    CHECK_OR_FALSE(param_.X);
    CHECK_OR_FALSE(param_.Y);
    CHECK_OR_FALSE(param_.Out);
    return true;
  }

  bool InferShape() const override {
    CHECK_OR_FALSE(param_.X->dims() == param_.Y->dims());
    param_.Out->Resize(param_.X->dims());
    return true;
  }

  bool AttachImpl(const OpDesc& opdesc, lite::Scope* scope) override {
    CHECK_EQ(opdesc.Inputs().size(), 2UL);
    auto X_name = opdesc.Input("X").front();
    auto Y_name = opdesc.Input("Y").front();
    auto Out_name = opdesc.Output("Out").front();

    param_.X = GetVar<lite::Tensor>(scope, X_name);
    param_.Y = GetVar<lite::Tensor>(scope, Y_name);
    param_.Out = GetMutableVar<Tensor>(scope, Out_name);
    param_.axis = boost::get<int>(opdesc.GetAttr("axis"));
  }

  void AttachKernel(KernelBase* kernel) override { kernel->SetParam(param_); }

 private:
  mutable operators::ElementwiseParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(elementwise_sub, paddle::lite::operators::ElementwiseOp);
