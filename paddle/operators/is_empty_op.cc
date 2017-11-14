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

#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace operators {

class IsEmptyOp : public framework::OperatorBase {
 public:
  IsEmptyOp(const std::string &type, const framework::VariableNameMap &inputs,
            const framework::VariableNameMap &outputs,
            const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    // get input
    auto *var = scope.FindVar(Input("X"));
    PADDLE_ENFORCE_NOT_NULL(var);
    auto &tensor = var->Get<framework::LoDTensor>();
    // get output
    auto *out = scope.FindVar(Output("Y"));
    PADDLE_ENFORCE_NOT_NULL(out);
    auto *out_tensor = out->GetMutable<framework::LoDTensor>();

    out_tensor->Resize(framework::make_ddim(std::vector<int>({1})));
    out_tensor->mutable_data<bool>(platform::CPUPlace())[0] =
        framework::product(tensor.dims()) == 0;
  }
};

class IsEmptyOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  IsEmptyOpProtoMaker(framework::OpProto *proto,
                      framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "the tensor to check whether is empty");
    AddOutput("Y",
              "a boolean variable that indicate whether the tensor is empty");
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
