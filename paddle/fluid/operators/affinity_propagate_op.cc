/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */


#include <string>
#include <unordered_map>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class AffinityPropagateOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of AffinityPropagateOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("GateWeight"),
        "Input(GateWeight) of AffinityPropagateOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of AffinityPropagateOp should not be null.");

    auto dim_x = ctx->GetInputDim("X");
    PADDLE_ENFORCE(dim_x.size() == 4 || dim_x.size() == 5,
                   "Input(X) dimension should be 4 or 5");

    auto dim_gate_w = ctx->GetInputDim("GateWeight");
    PADDLE_ENFORCE_EQ(
        dim_x.size(), dim_gate_w.size(),
        "Input(X) and Input(GateWeight) dimension should be same.");

    int kernel_size = ctx->Attrs().Get<int>("kernel_size");
    PADDLE_ENFORCE_EQ(kernel_size % 2, 1, "kernel_size should be odd number.");
    PADDLE_ENFORCE_GT(kernel_size, 1, "kernel_size should be greater than 1.");

    int channel_num = dim_x.size() == 4
                          ? kernel_size * kernel_size - 1
                          : kernel_size * kernel_size * kernel_size - 1;
    for (int i = 0; i < dim_gate_w.size(); i++) {
      if (i == 1) {
        // Guidance channel number should be kernel_size * kernel_size - 1
        PADDLE_ENFORCE_EQ(dim_gate_w[i], channel_num,
                          "Input(GateWeight) channel number "
                          "be kernel_size * kernel_size - 1.");
      } else {
        PADDLE_ENFORCE_EQ(
            dim_x[i], dim_gate_w[i],
            "Input(X) and Input(GateWeight) should in same shape.");
      }
    }

    ctx->SetOutputDim("Out", dim_x);
    ctx->ShareLoD("X", "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.GetPlace());
  }
};

class AffinityPropagateOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of affinity propagate operator, "
             "This is a 4-D tensor with shape of [N, C, H, W] or a 5-D "
             "tensor with shape of [N, C, D, H, W].");
    AddInput("GateWeight",
             "This gate weight tesnor, which should be the normalized "
             "guidance, it should be in the same shape with Input(X) "
             "except channel number should be attr:`kernel_size` * "
             "attr:`kernel_size` - 1.");
    AddOutput("Out",
              "The output tensor of affinity propagate operator, "
              "It should be in the same shape with Input(X).");

    AddAttr<int>("kernel_size",
                 "the size of convolution kernel, "
                 "currently only support 3.")
        .SetDefault(3);
    AddComment(R"DOC(
          TODO(dengkaipeng): add doc.
         )DOC");
  }
};

class AffinityPropagateOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("GateWeight"),
                   "Input(GateWeight) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto dim_x = ctx->GetInputDim("X");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), dim_x);
    }
    auto dim_gate_weight = ctx->GetInputDim("GateWeight");
    if (ctx->HasOutput(framework::GradVarName("GateWeight"))) {
      ctx->SetOutputDim(framework::GradVarName("GateWeight"), dim_gate_weight);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.Input<Tensor>(framework::GradVarName("Out"))->type(),
        ctx.GetPlace());
  }
};

class AffinityPropagateGradDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("affinity_propagate_grad");
    op->SetInput("X", Input("X"));
    op->SetInput("GateWeight", Input("GateWeight"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));

    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetOutput(framework::GradVarName("GateWeight"),
                  InputGrad("GateWeight"));

    op->SetAttrMap(Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(affinity_propagate, ops::AffinityPropagateOp,
                  ops::AffinityPropagateOpMaker,
                  ops::AffinityPropagateGradDescMaker);
REGISTER_OPERATOR(affinity_propagate_grad, ops::AffinityPropagateOpGrad);
