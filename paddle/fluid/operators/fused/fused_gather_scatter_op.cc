/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/fused_gather_scatter_op.h"

namespace paddle {
namespace operators {

class FusedGatherScatterOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "FusedGatherScatter");
    OP_INOUT_CHECK(ctx->HasInput("Gather_index"), "Input", "Gather_index",
                   "FusedGatherScatter");
    OP_INOUT_CHECK(ctx->HasInput("Scatter_index"), "Input", "Scatter_index",
                   "FusedGatherScatter");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out",
                   "FusedGatherScatter");

    auto gather_index_dims = ctx->GetInputDim("Gather_index");
    if (gather_index_dims.size() == 2) {
      PADDLE_ENFORCE_EQ(gather_index_dims[1], 1,
                        platform::errors::InvalidArgument(
                            "The last dim of Gather_index should be 1 when it "
                            "is 2D, but we get %d",
                            gather_index_dims[1]));
    } else {
      PADDLE_ENFORCE_EQ(
          gather_index_dims.size(), 1,
          platform::errors::InvalidArgument(
              "The Gather_index should be 1D, when it is not 2D, but we get %d",
              gather_index_dims.size()));
    }

    auto scatter_index_dims = ctx->GetInputDim("Scatter_index");
    if (scatter_index_dims.size() == 2) {
      PADDLE_ENFORCE_EQ(scatter_index_dims[1], 1,
                        platform::errors::InvalidArgument(
                            "The last dim of Scatter_index should be 1 when it "
                            "is 2D, but we get %d",
                            scatter_index_dims[1]));
    } else {
      PADDLE_ENFORCE_EQ(
          scatter_index_dims.size(), 1,
          platform::errors::InvalidArgument("The Scatter_index should be 1D, "
                                            "when it is not 2D, but we get %d",
                                            scatter_index_dims.size()));
    }

    // TODO(daisiming): If the shape of scatter_index and gather_index should be
    // same?
    auto dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", dims);

    if (ctx->Attrs().Get<std::string>("pool_type") == "MEAN") {
      OP_INOUT_CHECK(ctx->HasOutput("Scatter_count"), "Output", "Scatter_count",
                     "FusedGatherScatter");
      ctx->SetOutputDim("Scatter_count", {dims[0]});
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

class FusedGatherScatterGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    ctx->SetOutputDim(framework::GradVarName("X"), in_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

class FusedGatherScatterOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor with data type float32, "
             "float64 or float16");
    AddInput("Gather_index", "The gather index tensor.");
    AddInput("Scatter_index", "The scatter index tensor.");
    AddOutput("Out", "Output tensor of fused_gather_scatter op.");
    AddOutput("Scatter_count",
              "Count tensor of Scatter index, mainly for MEAN pool_type.")
        .AsIntermediate();
    AddAttr<std::string>(
        "pool_type",
        "(string, default 'SUM')"
        "We use Gather_index to gather correspoinding place of X. "
        "Then we need to use different pool type to scatter the result.")
        .SetDefault("SUM")
        .InEnum({"SUM", "MEAN", "MIN", "MAX"});
    // TODO(daisiming): Add a simple example here.
    AddComment(R"DOC(
Fused Gather Scatter Operator.

$Out = Scatter(Gather(X, Gather_index), Scatter_index, pool_type)$

This operator helps perform fused computation of gather operator and scatter operator, so as to 
decrease intermediate GPU memory occupation of using gather op and scatter op successively.

Example:

pass
)DOC");
  }
};

template <typename T>
class FusedGatherScatterGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("fused_gather_scatter_grad");
    op->SetInput("Gather_index", this->Input("Gather_index"));
    op->SetInput("Scatter_index", this->Input("Scatter_index"));

    if (BOOST_GET_CONST(std::string, this->GetAttr("pool_type")) == "MEAN") {
      op->SetInput("Scatter_count", this->Output("Scatter_count"));
    }

    if (BOOST_GET_CONST(std::string, this->GetAttr("pool_type")) == "MIN" ||
        BOOST_GET_CONST(std::string, this->GetAttr("pool_type")) == "MAX") {
      op->SetInput("X", this->Input("X"));
      op->SetInput("Out", this->Output("Out"));
    }

    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(
    fused_gather_scatter, ops::FusedGatherScatterOP,
    ops::FusedGatherScatterOpMaker,
    ops::FusedGatherScatterGradOpMaker<paddle::framework::OpDesc>,
    ops::FusedGatherScatterGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_gather_scatter_grad, ops::FusedGatherScatterGradOp);
REGISTER_OP_CPU_KERNEL(fused_gather_scatter,
                       ops::FusedGatherScatterOpKernel<CPU, float, int>,
                       ops::FusedGatherScatterOpKernel<CPU, float, int64_t>,
                       ops::FusedGatherScatterOpKernel<CPU, double, int>,
                       ops::FusedGatherScatterOpKernel<CPU, double, int64_t>,
                       ops::FusedGatherScatterOpKernel<CPU, int, int>,
                       ops::FusedGatherScatterOpKernel<CPU, int, int64_t>,
                       ops::FusedGatherScatterOpKernel<CPU, int64_t, int>,
                       ops::FusedGatherScatterOpKernel<CPU, int64_t, int64_t>);

REGISTER_OP_CPU_KERNEL(
    fused_gather_scatter_grad,
    ops::FusedGatherScatterGradOpKernel<CPU, float, int>,
    ops::FusedGatherScatterGradOpKernel<CPU, float, int64_t>,
    ops::FusedGatherScatterGradOpKernel<CPU, double, int>,
    ops::FusedGatherScatterGradOpKernel<CPU, double, int64_t>,
    ops::FusedGatherScatterGradOpKernel<CPU, int, int>,
    ops::FusedGatherScatterGradOpKernel<CPU, int, int64_t>,
    ops::FusedGatherScatterGradOpKernel<CPU, int64_t, int>,
    ops::FusedGatherScatterGradOpKernel<CPU, int64_t, int64_t>);
