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


#include "paddle/fluid/operators/affinity_propagate_op.h"

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
    PADDLE_ENFORCE(ctx->HasInput("Guidance"),
                   "Input(Guidance) of AffinityPropagateOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of AffinityPropagateOp should not be null.");
    
    auto dim_x = ctx->GetInputDim("X");
    PADDLE_ENFORCE(dim_x.size() == 4 || dim_x.size() == 5,
                   "Input(X) dimension should be 4 or 5");

    auto dim_guidance = ctx->GetInputDim("Guidance");
    PADDLE_ENFORCE_EQ(dim_x.size(), dim_guidance.size(),
                      "Input(X) and Input(Guidance) dimension should be same.");
    int kernel_size = ctx->Attrs().Get<int>("kernel_size");
    int channel_num = kernel_size * kernel_size - 1;
    for (int i = 0; i < dim_guidance.size(); i++) {
      if (i == 1) {
        // Guidance channel number should be kernel_size * kernel_size - 1
        PADDLE_ENFORCE_EQ(dim_guidance[i], channel_num, "Input(Guidance) channel number "
                          "be kernel_size * kernel_size - 1.");
      } else {
        PADDLE_ENFORCE_EQ(dim_x[i], dim_guidance[i],
                          "Input(X) and Input(Guidance) should in same shape.");
      }
    }

    if (ctx->HasInput("Mask")) {
      auto dim_mask = ctx->GetInputDim("Mask");
      PADDLE_ENFORCE_EQ(dim_x.size(), dim_guidance.size(),
                        "Input(X) and Input(Mask) dimension should be same.");
      for (int i = 0; i < dim_x.size(); i++) {
        PADDLE_ENFORCE_EQ(dim_x[i], dim_mask[i],
                          "Input(X) and Input(Mask) should in same shape.");
      }
    }

    int prop_iters = ctx->Attrs().Get<int>("prop_iters");
    PADDLE_ENFORCE_GT(prop_iters, 0, "Attr(prop_iters) should be greater than 0");
    std::string norm_type = ctx->Attrs().Get<std::string>("norm_type");
    PADDLE_ENFORCE(norm_type == "sum" || norm_type == "abs_sum",
                   "norm_type can only be \"sum\" or \"abs_sum\".");

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
    AddInput("Guidance",
             "This guidance weight tesnor, it should be in the same "
             "shape with Input(X)");
    AddInput("Mask",
             "The mask tensor, it should be in the same shape with "
             "Input(X)")
        .AsDispensable();
    AddOutput("Out",
              "The output tensor of affinity propagate operator, "
              "It should be in the same shape with Input(X).");

    AddAttr<int>("prop_iters", 
                 "the times to perform convolution spatial propagation.")
        .SetDefault(1);
    AddAttr<int>("kernel_size", "the size of convolution kernel, "
                 "currently only support 3.").SetDefault(3);
    AddAttr<std::string>("norm_type",
                         "the method to normalize affinity, currently "
                         "only support \"sum\" and \"abs_sum\".")
        .SetDefault("sum");
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
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto dim_x = ctx->GetInputDim("X");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), dim_x);
    }
    if (ctx->HasOutput(framework::GradVarName("Guidance"))) {
      ctx->SetOutputDim(framework::GradVarName("Guidance"), dim_x);
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
    op->SetInput("Guidance", Input("Guidance"));
    op->SetInput("Mask", Input("Mask"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));

    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetOutput(framework::GradVarName("Guidance"), InputGrad("Guidance"));

    op->SetAttrMap(Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(affinity_propagate, ops::AffinityPropagateOp, ops::AffinityPropagateOpMaker,
                  ops::AffinityPropagateGradDescMaker);
REGISTER_OPERATOR(affinity_propagate_grad, ops::AffinityPropagateOpGrad);

REGISTER_OP_CPU_KERNEL(affinity_propagate,
                       ops::AffinityPropagateOpKernel<float>,
                       ops::AffinityPropagateOpKernel<double>);
REGISTER_OP_CPU_KERNEL(affinity_propagate_grad, 
                       ops::AffinityPropagateGradOpKernel<float>,
                       ops::AffinityPropagateGradOpKernel<double>);
