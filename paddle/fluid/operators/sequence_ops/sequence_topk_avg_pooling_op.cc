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
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      "Input(X) of SequencePoolOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("ROW"), true,
                      "Input(ROW) of SequencePoolOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("COLUMN"), true,
                      "Input(COLUMN) of SequencePoolOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      "Output(Out) of SequencePoolOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("pos"), true,
                      "pos(out) should not be null");

    auto attr = ctx->Attrs();
    auto channel_num = attr.Get<int>("channel_num");
    auto topks = attr.Get<std::vector<int>>("topks");

    auto row_dim = ctx->GetInputDim("ROW");

    auto num_k = topks.size();
    auto row_shape_0 = row_dim[0];

    std::vector<int> vec_out_shape;
    vec_out_shape.push_back(row_shape_0);
    vec_out_shape.push_back(channel_num * num_k);

    ctx->SetOutputDim("Out", framework::make_ddim(vec_out_shape));
    ctx->ShareLoD("X", "Out");
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
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      "Gradient of Out should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      "The input X should not be null.");

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
  std::unique_ptr<T> Apply() const override {
    auto* op_desc_ptr = new T();
    op_desc_ptr->SetType("sequence_topk_avg_pooling_grad");
    op_desc_ptr->SetInput("X", this->Input("X"));
    op_desc_ptr->SetInput("ROW", this->Input("ROW"));
    op_desc_ptr->SetInput("COLUMN", this->Input("COLUMN"));
    op_desc_ptr->SetInput("pos", this->Output("pos"));
    op_desc_ptr->SetInput(framework::GradVarName("Out"),
                          this->OutputGrad("Out"));
    op_desc_ptr->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op_desc_ptr->SetAttrMap(this->Attrs());
    return std::unique_ptr<T>(op_desc_ptr);
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
