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

#include "paddle/fluid/operators/relu2_op.h"
#include <memory>
#include <string>
#include <unordered_map>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class Relu2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->ShareDim("X", "Out");
    ctx->ShareLoD("X", "Out");
    if (!ctx->IsRuntime()) {
      ctx->SetOutputDim("Mask", {-1});
    }
  }
};

class Relu2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of relu2 op");
    AddOutput("Mask", "A mask to indicate whether X >= 0. Used in backward")
        .AsIntermediate();
    AddOutput("Out", "Output of relu2 op");
    AddComment(R"DOC(Relu op with Output(mask))DOC");
  }
};

class Relu2GradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("relu2_grad");
    op->SetInput("Mask", Output("Mask"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

class Relu2GradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Mask"), "Mask must exist");
    auto y_grad = framework::GradVarName("Out");
    auto x_grad = framework::GradVarName("X");
    ctx->ShareDim(y_grad, x_grad);
    ctx->ShareLoD(y_grad, x_grad);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"))->type(),
        ctx.GetPlace());
  }
};

class Relu2Inplace : public framework::InplaceOpInference {
 public:
  std::unordered_map<std::string, std::string> operator()(
      const framework::OpDesc& op_desc) const override {
    return {{"X", "Out"}};
  }
};

class Relu2GradInplace : public framework::InplaceOpInference {
 public:
  std::unordered_map<std::string, std::string> operator()(
      const framework::OpDesc& op_desc) const override {
    return {{framework::GradVarName("Out"), framework::GradVarName("X")}};
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(relu2, ops::Relu2Op, ops::Relu2OpMaker,
                  ops::Relu2GradOpDescMaker, ops::Relu2Inplace);
REGISTER_OPERATOR(relu2_grad, ops::Relu2GradOp, ops::Relu2GradInplace);

REGISTER_OP_CPU_KERNEL(relu2, ops::Relu2Kernel<plat::CPUDeviceContext, float>,
                       ops::Relu2Kernel<plat::CPUDeviceContext, double>,
                       ops::Relu2Kernel<plat::CPUDeviceContext, plat::float16>);

REGISTER_OP_CPU_KERNEL(
    relu2_grad, ops::Relu2GradKernel<plat::CPUDeviceContext, float>,
    ops::Relu2GradKernel<plat::CPUDeviceContext, double>,
    ops::Relu2GradKernel<plat::CPUDeviceContext, plat::float16>);
