// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/stack_op.h"
#include <memory>
#include <vector>

namespace paddle {
namespace operators {

class StackOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_GT(ctx->Inputs("X").size(), 0,
                      "Number of Inputs(X) must be larger than 0");
    PADDLE_ENFORCE(ctx->HasOutput("Y"), "Output(Y) must exist.");

    auto input_dims = ctx->GetInputsDim("X");
    for (size_t i = 1; i < input_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(input_dims[i], input_dims[0],
                        "Dims of all Inputs(X) must be the same");
    }

    // Only lod of X[0] would be shared with Y
    ctx->ShareLoD("X", /*->*/ "Y");

    int axis = ctx->Attrs().Get<int>("axis");
    int rank = input_dims[0].size();
    PADDLE_ENFORCE(
        axis >= -(rank + 1) && axis < rank + 1,
        "Attr(axis) must be inside [-(rank+1), rank+1), where rank = %d", rank);
    if (axis < 0) axis += (rank + 1);

    auto vec = framework::vectorize2int(input_dims[0]);
    vec.insert(vec.begin() + axis, input_dims.size());
    ctx->SetOutputDim("Y", framework::make_ddim(vec));
  }
};

class StackOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of stack op.").AsDuplicable();
    AddOutput("Y", "The output of stack op.");
    AddAttr<int>("axis",
                 "The axis along which all of the Inputs(X) should be stacked.")
        .SetDefault(0);
    AddComment(R"DOC(
Stack Operator.

Stack all of the Inputs(X) into one tensor along Attr(axis). The dims of all Inputs(X) must be the same.
)DOC");
  }
};

class StackOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")),
                   "Input(Y@Grad) must exist.");

    int axis = ctx->Attrs().Get<int>("axis");
    auto dy_dim = ctx->GetInputDim(framework::GradVarName("Y"));
    int rank = dy_dim.size();
    PADDLE_ENFORCE(axis >= -rank && axis < rank,
                   "Attr(axis) must be inside [-rank, rank), where rank = %d",
                   rank);
    if (axis < 0) axis += rank;

    PADDLE_ENFORCE_EQ(ctx->Outputs(framework::GradVarName("X")).size(),
                      static_cast<size_t>(dy_dim[axis]),
                      "Number of Outputs(X@Grad) is wrong");
    auto vec = framework::vectorize2int(dy_dim);
    vec.erase(vec.begin() + axis);
    ctx->SetOutputsDim(
        framework::GradVarName("X"),
        std::vector<framework::DDim>(dy_dim[axis], framework::make_ddim(vec)));
  }
};

class StackGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("stack_grad");
    op->SetInput(framework::GradVarName("Y"), OutputGrad("Y"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X", false));
    op->SetAttrMap(Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
namespace ops = paddle::operators;
REGISTER_OPERATOR(stack, ops::StackOp, ops::StackOpMaker,
                  ops::StackGradOpDescMaker);
REGISTER_OPERATOR(stack_grad, ops::StackOpGrad);

REGISTER_OP_CPU_KERNEL(stack, ops::StackKernel<plat::CPUDeviceContext, float>,
                       ops::StackKernel<plat::CPUDeviceContext, double>,
                       ops::StackKernel<plat::CPUDeviceContext, int>,
                       ops::StackKernel<plat::CPUDeviceContext, int64_t>);

REGISTER_OP_CPU_KERNEL(stack_grad,
                       ops::StackGradKernel<plat::CPUDeviceContext, float>,
                       ops::StackGradKernel<plat::CPUDeviceContext, double>,
                       ops::StackGradKernel<plat::CPUDeviceContext, int>,
                       ops::StackGradKernel<plat::CPUDeviceContext, int64_t>);
