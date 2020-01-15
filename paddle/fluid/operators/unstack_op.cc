/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/unstack_op.h"
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

class UnStackOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true, "Input(X) must exist.");

    int axis = ctx->Attrs().Get<int>("axis");
    int num = ctx->Attrs().Get<int>("num");
    auto x_dim = ctx->GetInputDim("X");
    int rank = x_dim.size();
    PADDLE_ENFORCE_GE(
        axis, -rank, "Attr(axis) must be inside [-rank, rank), where rank = %d",
        rank);
    PADDLE_ENFORCE_LT(
        axis, rank, "Attr(axis) must be inside [-rank, rank), where rank = %d",
        rank);
    if (axis < 0) axis += rank;

    PADDLE_ENFORCE_EQ(ctx->Outputs("Y").size(), static_cast<size_t>(num),
                      "Number of Outputs(Y) is wrong");
    if (x_dim[axis] > 0) {
      PADDLE_ENFORCE_EQ(num, x_dim[axis], "Number of Outputs(Y) is wrong");
    }
    auto vec = framework::vectorize<int>(x_dim);
    vec.erase(vec.begin() + axis);
    ctx->SetOutputsDim("Y", std::vector<framework::DDim>(  // NOLINT
                                x_dim[axis], framework::make_ddim(vec)));
  }
};

class UnStackOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of unstack op.");
    AddOutput("Y", "The output of unstack op.").AsDuplicable();
    AddAttr<int>("axis", "The axis along which Input(X) should be unstacked.")
        .SetDefault(0);
    AddAttr<int>("num", "The number of outputs(Y).").GreaterThan(0);
    AddComment(R"DOC(
      UnStack Operator.

      UnStack Input(X) into several tensors along Attr(axis).
    )DOC");
  }
};

template <typename T>
class UnStackGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    std::unique_ptr<T> op(new T());
    op->SetType("unstack_grad");
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
    return op;
  }
};

class UnStackGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_GT(ctx->Inputs(framework::GradVarName("Y")).size(), 0,
                      "Number of Inputs(Y@Grad) must be larger than 0");
    PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("X")), true,
                      "Output(X@Grad) must exist.");

    auto input_dims = ctx->GetInputsDim(framework::GradVarName("Y"));
    for (size_t i = 1; i < input_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(input_dims[i], input_dims[0],
                        "Dims of all Inputs(Y@Grad) must be the same");
    }

    int axis = ctx->Attrs().Get<int>("axis");
    int rank = input_dims[0].size();
    PADDLE_ENFORCE_GE(
        axis, -(rank + 1),
        "Attr(axis) must be inside [-(rank+1), rank+1), where rank = %d", rank);
    PADDLE_ENFORCE_LT(
        axis, rank + 1,
        "Attr(axis) must be inside [-(rank+1), rank+1), where rank = %d", rank);
    if (axis < 0) axis += (rank + 1);

    auto vec = framework::vectorize<int>(input_dims[0]);
    vec.insert(vec.begin() + axis, input_dims.size());
    ctx->SetOutputDim(framework::GradVarName("X"), framework::make_ddim(vec));
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
namespace ops = paddle::operators;

REGISTER_OPERATOR(unstack, ops::UnStackOp, ops::UnStackOpMaker,
                  ops::UnStackGradOpMaker<paddle::framework::OpDesc>,
                  ops::UnStackGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(unstack_grad, ops::UnStackGradOp);

REGISTER_OP_CPU_KERNEL(unstack,
                       ops::UnStackKernel<plat::CPUDeviceContext, float>,
                       ops::UnStackKernel<plat::CPUDeviceContext, double>,
                       ops::UnStackKernel<plat::CPUDeviceContext, int>,
                       ops::UnStackKernel<plat::CPUDeviceContext, int64_t>);

REGISTER_OP_CPU_KERNEL(unstack_grad,
                       ops::UnStackGradKernel<plat::CPUDeviceContext, float>,
                       ops::UnStackGradKernel<plat::CPUDeviceContext, double>,
                       ops::UnStackGradKernel<plat::CPUDeviceContext, int>,
                       ops::UnStackGradKernel<plat::CPUDeviceContext, int64_t>);
