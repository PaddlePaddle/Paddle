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

#include "paddle/fluid/operators/sequence_ops/sequence_expand_as_op.h"
#include <memory>
#include <string>

namespace paddle {
namespace operators {

using framework::LoDTensor;

class SequenceExpandAsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SequenceExpandAsOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"),
                   "Input(Y) of SequenceExpandAsOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SequenceExpandAsOp should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    auto out_dims = x_dims;

    PADDLE_ENFORCE_GE(x_dims.size(), 2,
                      "Dimension number of Input(X) should be at least 2.");

    if (ctx->IsRuntime()) {
      framework::Variable* x_var =
          boost::get<framework::Variable*>(ctx->GetInputVarPtrs("X")[0]);
      framework::Variable* y_var =
          boost::get<framework::Variable*>(ctx->GetInputVarPtrs("Y")[0]);

      auto& x_dim = x_var->Get<LoDTensor>().dims();
      auto& y_lod = y_var->Get<LoDTensor>().lod();

      PADDLE_ENFORCE_EQ(y_lod.size(), 1,
                        "Level number of Input(Y)'s lod should be 1.");

      PADDLE_ENFORCE_EQ(static_cast<size_t>(x_dim[0]), y_lod[0].size() - 1,
                        "The first dimension of Input(X) should be equal "
                        "to the size of Input(Y)'s 0 level lod.");

      int64_t out_first_dim = 0;
      if (y_lod[0].size() <= 1) {
        out_first_dim = x_dims[0];
      } else {
        for (size_t i = 1; i < y_lod[0].size(); ++i) {
          out_first_dim += (y_lod[0][i] - y_lod[0][i - 1]);
        }
      }
      out_dims[0] = out_first_dim;
    } else {
      out_dims[0] = -1;
    }

    ctx->SetOutputDim("Out", out_dims);
    ctx->ShareLoD("Y", /*->*/ "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<framework::Tensor>("X")->type(),
                                   ctx.GetPlace());
  }
};

class SequenceExpandAsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor, default LoDTensor<float>) A 2-D LoDTensor whose lod "
             "level is at most 1.");
    AddInput("Y",
             "(LoDTensor, default LoDTensor<float>) Referred LoDTensor whose "
             "lod (specified level) is referred by Input(X).");
    AddOutput("Out",
              "(LodTensor, default LoDTensor<float>) Output LoDTensor which is "
              "generated from Input(X) by referring lod of Input(Y).");
    AddComment(R"DOC(
Sequence Expand As Operator.

This operator expands `X` according to the zeroth level lod of `Y`. Current
implementation requires the level number of Input(Y)'s lod should be 1, and
the first dimension of Input(X) should be equal to the size of Input(Y)'s zeroth
level lod, and lod of Input(X) is not considered.

Following are cases to better explain how this works:

Case 1:

Given a 1-level LoDTensor input(X)
    X.data = [[a], [b], [c], [d]]
    X.dims = [4, 1]
and input(Y)
    Y.lod = [[0, 3, 6, 7, 8]]
ref_level: 0
then we get 1-level LoDTensor
    Out.lod =  [[0,            3,              6,  7,  8]]
    Out.data = [[a], [a], [a], [b], [b], [b], [c], [d]]
    Out.dims = [8, 1]

Case 2:

Given a common Tensor input(X)
    X.data = [[a, b], [c, d], [e, f]]
    X.dims = [3, 2]
and input(Y)
    Y.lod = [[0, 2, 3, 6]]
ref_level: 0
then we get a common LoDTensor
    Out.lod =  [[0,             2,     3,                    6]]
    Out.data = [[a, b], [a, b] [c, d], [e, f], [e, f], [e, f]]
    Out.dims = [6, 2]

)DOC");
  }
};

class SequenceExpandAsOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
      ctx->ShareLoD("X", x_grad_name);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"))->type(),
        ctx.GetPlace());
  }
};

class SequenceExpandAsOpGradOpDescMaker
    : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("sequence_expand_as_grad");
    op->SetInput("X", Input("X"));
    op->SetInput("Y", Input("Y"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(
    SequenceExpandAsOpNoNeedBufferVarsInference, "Y");
DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(
    SequenceExpandAsGradOpNoNeedBufferVarsInference, "X", "Y");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sequence_expand_as, ops::SequenceExpandAsOp,
                  ops::SequenceExpandAsOpMaker,
                  ops::SequenceExpandAsOpGradOpDescMaker,
                  ops::SequenceExpandAsOpNoNeedBufferVarsInference);
REGISTER_OPERATOR(sequence_expand_as_grad, ops::SequenceExpandAsOpGrad,
                  ops::SequenceExpandAsGradOpNoNeedBufferVarsInference);
REGISTER_OP_CPU_KERNEL(
    sequence_expand_as,
    ops::SequenceExpandAsKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SequenceExpandAsKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SequenceExpandAsKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SequenceExpandAsKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    sequence_expand_as_grad,
    ops::SequenceExpandAsGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SequenceExpandAsGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SequenceExpandAsGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SequenceExpandAsGradKernel<paddle::platform::CPUDeviceContext,
                                    int64_t>);
