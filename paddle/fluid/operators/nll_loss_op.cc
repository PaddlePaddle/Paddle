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

#include "paddle/fluid/operators/nll_loss_op.h"
#include <memory>
#include <string>

namespace paddle {
namespace operators {

class NLLLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "NLLLoss");
    OP_INOUT_CHECK(ctx->HasInput("Label"), "Input", "Label", "NLLLoss");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "NLLLoss");
    OP_INOUT_CHECK(ctx->HasOutput("Total_weight"), "Output", "Total_weight",
                   "NLLLoss");

    auto x_dims = ctx->GetInputDim("X");
    auto label_dims = ctx->GetInputDim("Label");
    auto reduction = ctx->Attrs().Get<std::string>("reduction");

    PADDLE_ENFORCE_EQ(x_dims.size() == 2 || x_dims.size() == 4, true,
                      platform::errors::InvalidArgument(
                          "The tensor rank of Input(X) must be 2 or 4."));
    bool contain_unknown_dim = framework::contain_unknown_dim(x_dims) ||
                               framework::contain_unknown_dim(label_dims);
    bool check = ctx->IsRuntime() || !contain_unknown_dim;
    if (check) {
      PADDLE_ENFORCE_EQ(
          x_dims[0], label_dims[0],
          platform::errors::InvalidArgument(
              "ShapeError: Expected input batch_size to match label batch_size,"
              "But received: the Input(x) batch_size is [%s], the Input(label) "
              " batch_size is [%s].",
              x_dims[0], label_dims[0]));
      if (ctx->HasInput("Weight")) {
        auto w_dims = ctx->GetInputDim("Weight");
        PADDLE_ENFORCE_EQ(w_dims.size(), 1,
                          platform::errors::InvalidArgument(
                              "Input(Weight) should be a 1D tensor."));
        PADDLE_ENFORCE_EQ(x_dims[1], w_dims[0],
                          platform::errors::InvalidArgument(
                              "Input(Weight) Tensor's size should match"
                              "to the class numer."));
      }
    }
    if (x_dims.size() == 2) {
      if (reduction == "none") {
        ctx->SetOutputDim("Out", {x_dims[0]});
      } else {
        ctx->SetOutputDim("Out", {1});
      }
    } else if (x_dims.size() == 4) {
      PADDLE_ENFORCE_EQ(label_dims.size(), 3,
                        platform::errors::InvalidArgument(
                            "The tensor rank of Input(Label) must be 3."));
      auto input0 = x_dims[0];
      auto input2 = x_dims[2];
      auto input3 = x_dims[3];
      auto label0 = label_dims[0];
      auto label1 = label_dims[1];
      auto label2 = label_dims[2];
      PADDLE_ENFORCE_EQ(
          input0 == label0 && input2 == label1 && input3 == label2, true,
          platform::errors::InvalidArgument("Input(X) tensor shape should "
                                            "match to Input(Label) tensor "
                                            "shape."));
      if (reduction == "none") {
        ctx->SetOutputDim("Out", {x_dims[0], x_dims[2], x_dims[3]});
      } else {
        ctx->SetOutputDim("Out", {1});
      }
    }
    ctx->SetOutputDim("Total_weight", {1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class NLLLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>) A tensor whose last dimension "
             "size is equal to the number of classes. It  is expected to "
             "contain log-probabilities of each class. "
             "The X tensor's shape has to be either [batch_size, C] or"
             "[batch_size, C, dim1, ..., dimK] in with K >= 1 in the case "
             " K-dimensional loss.");
    AddInput("Label",
             "(Tensor, default Tensor<int64_t>) A tensor which represents the "
             "the ground truth. It contains the class index in the range "
             "[0, C-1] where C = number of classes. The Lable tensor's "
             "shape has to be (batch_size), or "
             "(batch_size, dim1, ..., dimK) "
             "with K >= 1 in the case K-dimensional loss.");
    AddInput("Weight",
             "(Tensor, optional) A tensor should be a 1D tensor assigning "
             "weight to each of the classes. It's shape must be [C], where "
             "C is the class number.")
        .AsDispensable();
    AddOutput("Out",
              "(Tensor, default Tensor<float>) A tensor that represents the "
              "NLL loss.");
    AddOutput("Total_weight",
              "(Tensor, default Tensor<float>) A tensor saves the total"
              "weight value in the forward process.");
    AddAttr<int64_t>("ignore_index",
                     "(int64_t, default -100), Specifies a target value that is"
                     "ignored and does not contribute to the input gradient.")
        .SetDefault(-100);
    AddAttr<std::string>(
        "reduction",
        "(string, default mean), Specifies the reduction to apply"
        "to the output. The options include \"none\", \"mean\","
        "\"sum\".")
        .SetDefault("mean");
    AddComment(R"DOC(
NLL(Negative Log Likelihood) Loss Operator.

This operator computes the NLL loss according to the inputs.
The loss can be described as:

$Out[i] = -X[Label[i]]*Weight[Label[i]]$

It can also be used for higher dimension inputs, such as 2D images, by 
providing an input of shape (batch_size, C, d1, d2, ..., dK), with 
K >= 1, where K is the number of dimensions, and a Label of 
appropriate shape. In the case of images, it computes NLL loss 
per-pixel.

)DOC");
  }
};

class NLLLossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "NLLLoss");
    OP_INOUT_CHECK(ctx->HasInput("Label"), "Input", "Label", "NLLLoss");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "NLLLoss");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   framework::GradVarName("X"), "NLLLoss");

    auto reduction = ctx->Attrs().Get<std::string>("reduction");
    auto x_dims = ctx->GetInputDim("X");
    auto label_dims = ctx->GetInputDim("Label");
    auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    bool contain_unknown_dim = framework::contain_unknown_dim(x_dims) ||
                               framework::contain_unknown_dim(dout_dims);
    bool check = ctx->IsRuntime() || !contain_unknown_dim;

    if (check) {
      auto batch_size = x_dims[0];
      if (x_dims.size() == 2) {
        PADDLE_ENFORCE_EQ(dout_dims.size(), 1,
                          platform::errors::InvalidArgument(
                              "The dimensions of Input(Out@Grad) must be 1"));
        if (reduction == "none") {
          PADDLE_ENFORCE_EQ(
              dout_dims[0], batch_size,
              platform::errors::InvalidArgument(
                  "The unreduced size ofInput(Out@Grad) must be the "
                  "same as batch_size."));
        } else {
          PADDLE_ENFORCE_EQ(
              dout_dims[0], 1,
              platform::errors::InvalidArgument(
                  "The reduced size of Input(Out@Grad) must be 1"));
        }
      } else if (x_dims.size() == 4) {
        if (reduction == "none") {
          PADDLE_ENFORCE_EQ(
              dout_dims.size(), 3,
              platform::errors::InvalidArgument(
                  "The dimensions of Input(Out@Grad) must be 3,But got [%s].",
                  dout_dims.size()));
          PADDLE_ENFORCE_EQ(
              dout_dims[0] == label_dims[0] && dout_dims[1] == label_dims[1] &&
                  dout_dims[2] == label_dims[2],
              true, platform::errors::InvalidArgument(
                        "The dimensions of Input(Out@Grad) must be match "
                        "to Input(Label) dimensions."));
        } else {
          PADDLE_ENFORCE_EQ(
              dout_dims[0], 1,
              platform::errors::InvalidArgument(
                  "The reduced size of Input(Out@Grad) must be 1"));
        }
      }
    }

    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

template <typename T>
class NLLLossGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("nll_loss_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Label", this->Input("Label"));
    op->SetInput("Total_weight", this->Output("Total_weight"));

    if (this->HasInput("Weight")) {
      op->SetInput("Weight", this->Input("Weight"));
    }
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));

    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(nll_loss, ops::NLLLossOp, ops::NLLLossOpMaker,
                  ops::NLLLossGradMaker<paddle::framework::OpDesc>,
                  ops::NLLLossGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(nll_loss_grad, ops::NLLLossGradOp);
REGISTER_OP_CPU_KERNEL(
    nll_loss, ops::NLLLossOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::NLLLossOpKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    nll_loss_grad,
    ops::NLLLossGradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::NLLLossGradOpKernel<paddle::platform::CPUDeviceContext, double>);
