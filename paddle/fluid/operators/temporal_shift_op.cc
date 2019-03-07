/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/operators/temporal_shift_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class TemporalShiftOp: public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of TemporalShiftOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of TemporalShiftOp should not be null.");

    auto dim_x = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(dim_x.size(), 4, 
                   "Input(X) rank should be 4 in shape of [N*T, C, H, W].");

    int seg_num = ctx->Attrs().Get<int>("seg_num");
    PADDLE_ENFORCE_GT(seg_num, 0,
                   "Attr(seg_num) should be greater then 0.");

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(dim_x[0] % seg_num, 0,
                     "Input(X) dims[0] should be divided exactly by Attr(seg_num).");
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

class TemporalShiftOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of temporal shift operator. "
             "This is a 4-D tensor with shape of [N*T,  C, H, W]. "
             "While N is the batch size, T is the temporal segment "
             "number, C is the channel number, H is the height of "
             "features and W is the width of features.");
    AddOutput("Out",
              "The output tensor of temporal shift operator. "
              "This is a 4-D tensor in the same shape with Input(X).");

    AddAttr<int>("seg_num", 
              "The temporal segment number, this should be a positive "
              "interger.");

    AddComment(R"DOC(
          This operator calculates the temporal shift features for Input(X).

          For details of spectral normalization, please refer to paper: 
          `Temporal Shift Module <arxiv.org/abs/1802.0595://arxiv.org/abs/1811.08383>`_ .

         )DOC");
  }
};

class TemporalShiftOpGrad: public framework::OperatorWithKernel {
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
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(temporal_shift, ops::TemporalShiftOp, ops::TemporalShiftOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(temporal_shift_grad, ops::TemporalShiftOpGrad);
REGISTER_OP_CPU_KERNEL(temporal_shift, ops::TemporalShiftKernel<float>,
                       ops::TemporalShiftKernel<double>);
REGISTER_OP_CPU_KERNEL(temporal_shift_grad, ops::TemporalShiftGradKernel<float>,
                       ops::TemporalShiftGradKernel<double>);
