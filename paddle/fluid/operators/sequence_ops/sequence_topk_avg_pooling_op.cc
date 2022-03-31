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

#include "paddle/fluid/operators/sequence_ops/sequence_topk_avg_pooling_op.h"
#include <memory>
#include <string>

namespace paddle {
namespace operators {

class SequenceTopkAvgPoolingOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SequenceTopkAvgPooling");
    OP_INOUT_CHECK(ctx->HasInput("ROW"), "Input", "ROW",
                   "SequenceTopkAvgPooling");
    OP_INOUT_CHECK(ctx->HasInput("COLUMN"), "Input", "COLUMN",
                   "SequenceTopkAvgPooling");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out",
                   "SequenceTopkAvgPooling");
    OP_INOUT_CHECK(ctx->HasOutput("pos"), "Output", "pos",
                   "SequenceTopkAvgPooling");

    auto attr = ctx->Attrs();
    auto channel_num = attr.Get<int>("channel_num");
    PADDLE_ENFORCE_GT(
        channel_num, 0,
        platform::errors::InvalidArgument(
            "Expected channel_num > 0, but received %d.", channel_num));

    auto topks = attr.Get<std::vector<int>>("topks");
    auto num_k = topks.size();
    PADDLE_ENFORCE_GT(
        num_k, 0, platform::errors::InvalidArgument(
                      "Expected topks.size() > 0, but received %zu.", num_k));

    auto row_dim = ctx->GetInputDim("ROW");
    auto row_shape_0 = row_dim[0];

    std::vector<int> vec_out_shape;
    vec_out_shape.push_back(row_shape_0);
    vec_out_shape.push_back(channel_num * num_k);

    ctx->SetOutputDim("Out", phi::make_ddim(vec_out_shape));
    ctx->ShareLoD("ROW", "Out");
  }
};

class SequenceTopkAvgPoolingOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor) The variable-length input of SequenceTopkPoolingOp");
    AddInput("ROW", "(LoDTensor) the row info");
    AddInput("COLUMN", "(LoDTensor) the column info");
    AddOutput(
        "Out",
        "(Tensor) The output of SequenceTopkPoolingOp does not contain LoD "
        "information.");
    AddOutput("pos", "(Tensor<int>) store the topk index ").AsIntermediate();
    AddAttr<std::vector<int>>("topks", "topks");
    AddAttr<int>("channel_num", "channel number");
    AddComment(R"DOC(
    sequecen topk average pooling op
    )DOC");
  }
};

class SequenceTopkAvgPoolingGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "SequenceTopkAvgPoolingGrad");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X",
                   "SequenceTopkAvgPoolingGrad");

    ctx->ShareDim("X", /*->*/ framework::GradVarName("X"));
    ctx->ShareLoD("X", /*->*/ framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

template <typename T>
class SequenceTopkAvgPoolGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op_desc_ptr) const override {
    op_desc_ptr->SetType("sequence_topk_avg_pooling_grad");
    op_desc_ptr->SetInput("X", this->Input("X"));
    op_desc_ptr->SetInput("ROW", this->Input("ROW"));
    op_desc_ptr->SetInput("COLUMN", this->Input("COLUMN"));
    op_desc_ptr->SetInput("pos", this->Output("pos"));
    op_desc_ptr->SetInput(framework::GradVarName("Out"),
                          this->OutputGrad("Out"));
    op_desc_ptr->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op_desc_ptr->SetAttrMap(this->Attrs());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    sequence_topk_avg_pooling, ops::SequenceTopkAvgPoolingOp,
    ops::SequenceTopkAvgPoolingOpMaker,
    ops::SequenceTopkAvgPoolGradOpMaker<paddle::framework::OpDesc>,
    ops::SequenceTopkAvgPoolGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(sequence_topk_avg_pooling_grad,
                  ops::SequenceTopkAvgPoolingGradOp);
REGISTER_OP_CPU_KERNEL(sequence_topk_avg_pooling,
                       ops::SequenceTopkAvgPoolingKernel<
                           paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(sequence_topk_avg_pooling_grad,
                       ops::SequenceTopkAvgPoolingGradKernel<
                           paddle::platform::CPUDeviceContext, float>);
