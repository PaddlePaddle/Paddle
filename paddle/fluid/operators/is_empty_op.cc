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

class IsEmptyOp : public framework::OperatorBase {
 public:
  IsEmptyOp(const std::string &type, const framework::VariableNameMap &inputs,
            const framework::VariableNameMap &outputs,
            const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto x_var_name = Input("X");
    auto *x_var = scope.FindVar(x_var_name);
    PADDLE_ENFORCE_NOT_NULL(
        x_var, "Input(X) of is_empty op, name %s, should not be null.",
        x_var_name);

    auto out_var_name = Output("Out");
    auto *out_var = scope.FindVar(out_var_name);
    PADDLE_ENFORCE_NOT_NULL(
        out_var, "Output(Out) of is_empty op, name %s, should not be null.",
        out_var_name);

    auto &x_tensor = x_var->Get<framework::LoDTensor>();
    auto *out_tensor = out_var->GetMutable<framework::LoDTensor>();
    bool *out_data = out_tensor->mutable_data<bool>(framework::make_ddim({1}),
                                                    platform::CPUPlace());
    out_data[0] = x_tensor.numel() == 0;
  }
};

class IsEmptyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(LoDTensor) Tensor which is to be checked.");
    AddOutput("Out",
              "(LoDTensor) a boolean Tensor that indicate empty or not.");
    AddComment(R"DOC(
IsEmpty Operator which checks whether a tensor is empty.

It will just return product(tensor.ddims()) > 0.
Note: this operator will always run on CPU and the output tensor will always has CPU memory.

              )DOC");
  }
};

class IsEmptyInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of IsEmptyOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of IsEmptyOp should not be null.");
    ctx->SetOutputDim("Out", {1});
  }
};

class IsEmptyInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc &op_desc,
                  framework::BlockDesc *block) const override {
    auto x_name = op_desc.Input("X")[0];
    auto out_name = op_desc.Output("Out")[0];
    auto &out = block->FindRecursiveOrCreateVar(out_name);
    out.SetType(framework::proto::VarType::LOD_TENSOR);
    out.SetDataType(framework::proto::VarType::BOOL);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(is_empty, paddle::operators::IsEmptyOp,
                  paddle::operators::IsEmptyOpMaker,
                  paddle::operators::IsEmptyInferShape,
                  paddle::operators::IsEmptyInferVarType,
                  paddle::framework::EmptyGradOpMaker);
