/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/run_program_op.h"

#include <string>

namespace paddle {
namespace operators {

class RunProgramOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInputs("X"), true,
                      platform::errors::NotFound(
                          "Input(X) of RunProgramOp should not be null."));
    // PADDLE_ENFORCE_EQ(
    //     ctx->HasInputs("Params"), true,
    //     platform::errors::NotFound("Input(Params) of RunProgramOp should not
    //     be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutputs("Out"), true,
                      platform::errors::NotFound(
                          "Output(Out) of RunProgramOp should not be null."));

    // TODO(chenweihang): feed targets shape check
    // get var dims, compare to var desc shape
  }
};

class RunProgramOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(vector<LoDTensor>), The feed targets of executed program.")
        .AsDuplicable();
    AddInput("Params",
             "(vector<LoDTensor>), The parameters of executed program.")
        .AsDuplicable();
    AddOutput("Out",
              "，(vector<LoDTensor>), The fetch targets of executed program.")
        .AsDuplicable();
    AddOutput("OutScope", "(StepScopeVar), execution scope");
    AddAttr<BlockDesc*>("fwd_block", "The froward progarm desc.");
    AddAttr<BlockDesc*>("bwd_block", "The froward progarm desc.");
    AddAttr<std::vector<std::string>>("input_var_names", "input var names")
        .SetDefault({});
    AddAttr<std::vector<std::string>>("param_names", "param var names")
        .SetDefault({});
    AddAttr<std::vector<std::string>>("output_var_names", "output var names")
        .SetDefault({});
    // AddAttr<std::string>(
    //     "feed_var_name", "The feed var name")
    //     .SetDefault("feed");
    // AddAttr<std::string>(
    //     "fetch_var_name", "The fetch var name")
    //     .SetDefault("fetch");
    AddComment(R"DOC(Run static model in dygraph model.)DOC");
  }
};

class RunProgramGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInputs("X"), true,
                      platform::errors::NotFound(
                          "Input(X) of RunProgramGradOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInputs("Params"), true,
        platform::errors::NotFound(
            "Input(Params) of RunProgramGradOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInputs(framework::GradVarName("Out")), true,
        platform::errors::NotFound(
            "Input(Out@GRAD) of RunProgramGradOp should not be null."));

    // TODO(chenweihang): set output dims
  }
};

template <typename T>
class RunProgramGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("run_program_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Params", this->Input("Params"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetInput("OutScope", this->Output("OutScope"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Params"),
                    this->InputGrad("Params"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(run_program, ops::RunProgramOp, ops::RunProgramOpMaker,
                  ops::RunProgramGradOpMaker<paddle::framework::OpDesc>,
                  ops::RunProgramGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(run_program_grad, ops::RunProgramGradOp);

REGISTER_OP_CPU_KERNEL(
    run_program,
    ops::RunProgramOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::RunProgramOpKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::RunProgramOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::RunProgramOpKernel<paddle::platform::CPUDeviceContext, double>)

REGISTER_OP_CPU_KERNEL(
    run_program_grad,
    ops::RunProgramGradOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::RunProgramGradOpKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::RunProgramGradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::RunProgramGradOpKernel<paddle::platform::CPUDeviceContext, double>)
