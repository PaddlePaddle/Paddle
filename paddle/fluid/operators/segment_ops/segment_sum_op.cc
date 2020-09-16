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

#include "paddle/fluid/operators/segment_ops/segment_sum_op.h"
#include <memory>
#include <string>

namespace paddle {
namespace operators {

class SegmentSumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SegmentSum");
    OP_INOUT_CHECK(ctx->HasInput("SegmentIds"), "Input", "SegmentIds",
                   "SegmentSum");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "SegmentSum");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));

    // if (ctx->Attrs().Get<std::string>("pooltype") == "MAX") {
    //  OP_INOUT_CHECK(ctx->HasOutput("MaxIndex"), "Output", "MaxIndex",
    //                 "SegmentSum");
    //  ctx->SetOutputDim("MaxIndex", ctx->GetInputDim("X"));
    //}
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class SegmentSumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The variable-length input of SegmentSumOp");
    AddInput("SegmentIds",
             "(Tensor) The variable-length input of SegmentSumOp");
    // AddOutput("MaxIndex",
    //          "(Tensor<int>) This tensor is used for the sequence max-pooling
    //          "
    //          "to record the max indexes.")
    //    .AsIntermediate();
    AddOutput("Out", "(Tensor) The output of SegmentSumOp.");
    AddAttr<std::string>(
        "pooltype",
        "(string, default 'SUM') the pooling pooltype of SegmentSumOp.")
        .SetDefault("SUM")
        .InEnum({"SUM", "MEAN", "MIN", "MAX"});
    AddComment(R"DOC(
Segment Sum Operator.

This operator sums the elements of input `X` which with the same index in `SegmentIds`.
It computes a tensor such that $Out_i = \sum_{j} X_{j}$ where sum is over j such that `SegmentIds[j] == i`.
    )DOC");
  }
};

class SegmentSumGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "SegmentSumGrad");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SegmentSumGrad");
    auto og_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(og_dims.size(), x_dims.size(),
                      platform::errors::InvalidArgument(
                          "The rank of output grad must equal to Input(X). But "
                          "received: input rank %u, input shape [%s].",
                          og_dims.size(), og_dims));
    for (int64_t i = 1; i < og_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          og_dims[i], x_dims[i],
          platform::errors::InvalidArgument(
              "The dimension mismatch between Input(OUT@GRAD) and "
              "Input(X). Received Input(OUT@GRAD): input rank %u, "
              "input shape [%s]; received Input(X): input rank %u, "
              "input shape [%s].",
              og_dims.size(), og_dims, x_dims.size(), x_dims));
    }

    ctx->ShareDim("X", /*->*/ framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class SegmentSumGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op_desc_ptr) const override {
    op_desc_ptr->SetType("segment_sum_grad");
    op_desc_ptr->SetInput("X", this->Input("X"));
    op_desc_ptr->SetInput("SegmentIds", this->Input("SegmentIds"));
    op_desc_ptr->SetInput("Out", this->Output("Out"));
    // if (BOOST_GET_CONST(std::string, this->GetAttr("pooltype")) == "MAX") {
    //  op_desc_ptr->SetInput("MaxIndex", this->Output("MaxIndex"));
    //}
    op_desc_ptr->SetInput(framework::GradVarName("Out"),
                          this->OutputGrad("Out"));
    op_desc_ptr->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op_desc_ptr->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(segment_sum, ops::SegmentSumOp, ops::SegmentSumOpMaker,
                  ops::SegmentSumGradOpMaker<paddle::framework::OpDesc>,
                  ops::SegmentSumGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(segment_sum_grad, ops::SegmentSumGradOp);

REGISTER_OP_CPU_KERNEL(
    segment_sum,
    ops::SegmentSumKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SegmentSumKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    segment_sum_grad,
    ops::SegmentSumGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SegmentSumGradKernel<paddle::platform::CPUDeviceContext, double>);
