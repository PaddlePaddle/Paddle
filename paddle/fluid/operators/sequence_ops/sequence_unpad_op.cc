/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/sequence_ops/sequence_unpad_op.h"
#include <memory>
#include <string>

namespace paddle {
namespace operators {

class SequenceUnpadOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      "Input(X) of SequenceUnpadOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("Length"), true,
                      "Input(Length) of SequenceUnpadOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      "Output(Out) of SequenceUnpadOp should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_GE(x_dims.size(), 2,
                      "The rank of Input(X) can't be less than 2.");

    auto len_dims = ctx->GetInputDim("Length");
    PADDLE_ENFORCE_EQ(len_dims.size(), 1,
                      "The shape of Input(Length) should be [batch_size].");
    PADDLE_ENFORCE_EQ(
        len_dims[0], x_dims[0],
        "Input(X) and Input(Length) should have the same first dimension.");

    int64_t out_dim_0 = -1;
    if (ctx->IsRuntime()) {
      out_dim_0 = x_dims[0] * x_dims[1];
    }

    std::vector<int64_t> out_dims_vec{out_dim_0};
    if (x_dims.size() == 2) {
      out_dims_vec.push_back(1);
    } else {
      for (int i = 2; i < x_dims.size(); ++i) {
        out_dims_vec.push_back(x_dims[i]);
      }
    }
    ctx->SetOutputDim("Out", framework::make_ddim(out_dims_vec));
    if (!ctx->IsRuntime()) {
      ctx->SetLoDLevel("Out", 1);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class SequenceUnpadOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor, default LoDTensor<float>) Input tensor which "
             "contains the padded sequences with equal length.");
    AddInput("Length",
             "(LoDTensor) The input tensor which specifies the actual ength of "
             "sequences after unpadding.");
    AddOutput(
        "Out",
        "(LoDTensor) The output tensor which contains unpadded sequences.");
    AddComment(R"DOC(
      Sequence Unpad Operator

      This operator removes the padding data in the input sequences and convert 
      them into sequences with actual length as output, identitied by lod 
      information.

      Example:

      Given input tensor Input(X):
          X.data = [[ 1.0,  2.0,  3.0,  4.0,  5.0],
                    [ 6.0,  7.0,  8.0,  9.0, 10.0],
                    [11.0, 12.0, 13.0, 14.0, 15.0]], 
`     
      in which there are 3 sequences padded to length 5, and the actual length 
      specified by Input(Length):

          Length.data = [2, 3, 4],

      after unpadding, Output(Out) will be:

          Out.data = [[1.0, 2.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 14.0]]
          Out.lod = [[0, 2, 5, 9]]      

    )DOC");
  }
};

class SequenceUnpadGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      "Input(X) of SequenceUnpadGradOp should not be null.");
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Out")), true,
        "Input(Out@GRAD) of SequenceUnpadGradOp should not be null.");

    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
      ctx->ShareLoD("X", /*->*/ framework::GradVarName("X"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

template <typename T>
class SequenceUnpadGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    std::unique_ptr<T> op(new T());
    op->SetType("sequence_unpad_grad");
    op->SetAttrMap(this->Attrs());
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    return op;
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(
    SequenceUnpadGradOpNoNeedBufferVarsInference, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sequence_unpad, ops::SequenceUnpadOp,
                  ops::SequenceUnpadOpMaker,
                  ops::SequenceUnpadGradOpMaker<paddle::framework::OpDesc>,
                  ops::SequenceUnpadGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(sequence_unpad_grad, ops::SequenceUnpadGradOp,
                  ops::SequenceUnpadGradOpNoNeedBufferVarsInference);
REGISTER_OP_CPU_KERNEL(
    sequence_unpad,
    ops::SequenceUnpadOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SequenceUnpadOpKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SequenceUnpadOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SequenceUnpadOpKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    sequence_unpad_grad,
    ops::SequenceUnpadGradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SequenceUnpadGradOpKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SequenceUnpadGradOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SequenceUnpadGradOpKernel<paddle::platform::CPUDeviceContext,
                                   int64_t>);
