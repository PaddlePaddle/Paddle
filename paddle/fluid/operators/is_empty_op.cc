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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

constexpr char kInput[] = "X";
constexpr char kOutput[] = "Out";

class IsEmptyOp : public framework::OperatorBase {
 public:
  IsEmptyOp(const std::string &type, const framework::VariableNameMap &inputs,
            const framework::VariableNameMap &outputs,
            const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    // get input
    auto *var = scope.FindVar(Input(kInput));
    PADDLE_ENFORCE_NOT_NULL(var);
    auto &tensor = var->Get<framework::LoDTensor>();
    // get output
    auto *out = scope.FindVar(Output(kOutput));
    PADDLE_ENFORCE_NOT_NULL(out);
    auto *out_tensor = out->GetMutable<framework::LoDTensor>();

    out_tensor->Resize({1});
    out_tensor->mutable_data<bool>(platform::CPUPlace())[0] =
        framework::product(tensor.dims()) == 0;
  }
};

class IsEmptyOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  IsEmptyOpProtoMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(kInput, "(Tensor) Tensor which is to be checked.");
    AddOutput(kOutput, "(Tensor) a boolean Tensor that indicate empty or not.");
    AddComment(R"DOC(
IsEmpty Operator which checks whether a tensor is empty.

It will just return product(tensor.ddims()) > 0;
              )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(is_empty, paddle::operators::IsEmptyOp,
                             paddle::operators::IsEmptyOpProtoMaker);
