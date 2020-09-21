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

#include "paddle/fluid/operators/segment_pool_op.h"
#include <memory>
#include <string>

namespace paddle {
namespace operators {

class SegmentPoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SegmentPool");
    OP_INOUT_CHECK(ctx->HasInput("SegmentIds"), "Input", "SegmentIds",
                   "SegmentPool");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "SegmentPool");
    auto dims = ctx->GetInputDim("X");
    dims[0] = -1;
    ctx->SetOutputDim("Out", dims);

    if (ctx->Attrs().Get<std::string>("pooltype") == "MEAN") {
      OP_INOUT_CHECK(ctx->HasOutput("SummedIds"), "Output", "SummedIds",
                     "SegmentPool");
      ctx->SetOutputDim("SummedIds", {-1, 1});
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

class SegmentPoolOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input data of SegmentPoolOp");
    AddInput("SegmentIds",
             "(Tensor) 1-D tensor which have the same size with the fist "
             "dimension of input X.");
    AddOutput("Out", "(Tensor) The output of SegmentPoolOp.");
    AddOutput("SummedIds",
              "(Tensor) This tensor is used to counts of segment ids for the "
              "backward of the mean pool.")
        .AsIntermediate();
    AddAttr<std::string>(
        "pooltype",
        "(string, default 'SUM') the pooling type of SegmentPoolOp.")
        .SetDefault("SUM")
        .InEnum({"SUM", "MEAN", "MIN", "MAX"});
    AddComment(R"DOC(
Segment Pool Operator.

This operator will pool the elements of input `X` which with the same index
in `SegmentIds`.

For SUM operation, it computes a tensor such that $Out_i = \sum_{j} X_{j}$
where sum is over j such that `SegmentIds[j] == i`.

For MEAN operation, it computes a tensor such that
$Out_i = \frac{1}{n_i}  \sum_{j} X_{j}$ where sum is over j such that
`SegmentIds[j] == i` and $n_i$ is the number of all index `SegmentIds[j] == i`.

For MIN operation, it computes a tensor such that $Out_i = \min_{j} X_{j}$
where min is over j such that `SegmentIds[j] == i`.

For MAX operation, it computes a tensor such that $Out_i = \max_{j} X_{j}$
where max is over j such that `SegmentIds[j] == i`.
    )DOC");
  }
};

class SegmentPoolGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "SegmentPoolGrad");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SegmentPoolGrad");
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
class SegmentPoolGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op_desc_ptr) const override {
    op_desc_ptr->SetType("segment_pool_grad");
    op_desc_ptr->SetInput("X", this->Input("X"));
    op_desc_ptr->SetInput("SegmentIds", this->Input("SegmentIds"));
    op_desc_ptr->SetInput("Out", this->Output("Out"));
    if (BOOST_GET_CONST(std::string, this->GetAttr("pooltype")) == "MEAN") {
      op_desc_ptr->SetInput("SummedIds", this->Output("SummedIds"));
    }
    op_desc_ptr->SetInput(framework::GradVarName("Out"),
                          this->OutputGrad("Out"));
    op_desc_ptr->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op_desc_ptr->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(segment_pool, ops::SegmentPoolOp, ops::SegmentPoolOpMaker,
                  ops::SegmentPoolGradOpMaker<paddle::framework::OpDesc>,
                  ops::SegmentPoolGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(segment_pool_grad, ops::SegmentPoolGradOp);

REGISTER_OP_CPU_KERNEL(
    segment_pool,
    ops::SegmentPoolKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SegmentPoolKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    segment_pool_grad,
    ops::SegmentPoolGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SegmentPoolGradKernel<paddle::platform::CPUDeviceContext, double>);
