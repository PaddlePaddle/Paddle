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

class SequenceExpandAsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs("X"), "Input", "X", "SequenceExpandAs");
    OP_INOUT_CHECK(ctx->HasInputs("Y"), "Input", "Y", "SequenceExpandAs");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "SequenceExpandAs");

    auto x_dims = ctx->GetInputDim("X");
    auto out_dims = x_dims;

    PADDLE_ENFORCE_GE(x_dims.size(),
                      2,
                      platform::errors::InvalidArgument(
                          "Dimension number of Input(X) should be at least 2. "
                          "But received X's dimensions = %d, X's shape = [%s].",
                          x_dims.size(),
                          x_dims));

    if (ctx->IsRuntime()) {
      framework::Variable* x_var =
          PADDLE_GET(framework::Variable*, ctx->GetInputVarPtrs("X")[0]);
      framework::Variable* y_var =
          PADDLE_GET(framework::Variable*, ctx->GetInputVarPtrs("Y")[0]);

      auto& x_dim = x_var->Get<phi::DenseTensor>().dims();
      auto& y_lod = y_var->Get<phi::DenseTensor>().lod();

      PADDLE_ENFORCE_EQ(y_lod.size(),
                        1,
                        platform::errors::InvalidArgument(
                            "Level number of Input(Y)'s lod should be 1. But "
                            "received Y's lod level = %d.",
                            y_lod.size()));

      PADDLE_ENFORCE_EQ(static_cast<size_t>(x_dim[0]),
                        y_lod[0].size() - 1,
                        platform::errors::InvalidArgument(
                            "The first dimension of Input(X) should be one "
                            "less than the size of Input(Y)'s 0 level lod. But "
                            "received X's shape[0] = %d, Y's lod[0].size = %d.",
                            x_dim[0],
                            y_lod[0].size()));

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
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class SequenceExpandAsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(phi::DenseTensor, default phi::DenseTensor<float>) A 2-D "
             "phi::DenseTensor whose lod "
             "level is at most 1.");
    AddInput("Y",
             "(phi::DenseTensor, default phi::DenseTensor<float>) Referred "
             "phi::DenseTensor whose "
             "lod (specified level) is referred by Input(X).");
    AddOutput("Out",
              "(phi::DenseTensor, default phi::DenseTensor<float>) Output "
              "phi::DenseTensor which is "
              "generated from Input(X) by referring lod of Input(Y).");
    AddComment(R"DOC(
Sequence Expand As Operator.

This operator expands `X` according to the zeroth level lod of `Y`. Current
implementation requires the level number of Input(Y)'s lod should be 1, and
the first dimension of Input(X) should be equal to the size of Input(Y)'s zeroth
level lod, and lod of Input(X) is not considered.

Following are cases to better explain how this works:

Case 1:

Given a 1-level phi::DenseTensor input(X)
    X.data = [[a], [b], [c], [d]]
    X.dims = [4, 1]
and input(Y)
    Y.lod = [[0, 3, 6, 7, 8]]
ref_level: 0
then we get 1-level phi::DenseTensor
    Out.lod =  [[0,            3,              6,  7,  8]]
    Out.data = [[a], [a], [a], [b], [b], [b], [c], [d]]
    Out.dims = [8, 1]

Case 2:

Given a common phi::DenseTensor input(X)
    X.data = [[a, b], [c, d], [e, f]]
    X.dims = [3, 2]
and input(Y)
    Y.lod = [[0, 2, 3, 6]]
ref_level: 0
then we get a common phi::DenseTensor
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
    OP_INOUT_CHECK(ctx->HasInputs("X"), "Input", "X", "SequenceExpandAsGrad");
    OP_INOUT_CHECK(ctx->HasInputs(framework::GradVarName("Out")),
                   "Input",
                   "Out@GRAD",
                   "SequenceExpandAsGrad");

    auto x_dims = ctx->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
      ctx->ShareLoD("X", x_grad_name);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

template <typename T>
class SequenceExpandAsOpGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("sequence_expand_as_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(SequenceExpandAsOpNoNeedBufferVarsInferer,
                                    "Y");
DECLARE_NO_NEED_BUFFER_VARS_INFERER(
    SequenceExpandAsGradOpNoNeedBufferVarsInferer, "X", "Y");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    sequence_expand_as,
    ops::SequenceExpandAsOp,
    ops::SequenceExpandAsOpMaker,
    ops::SequenceExpandAsOpGradOpMaker<paddle::framework::OpDesc>,
    ops::SequenceExpandAsOpGradOpMaker<paddle::imperative::OpBase>,
    ops::SequenceExpandAsOpNoNeedBufferVarsInferer);
REGISTER_OPERATOR(sequence_expand_as_grad,
                  ops::SequenceExpandAsOpGrad,
                  ops::SequenceExpandAsGradOpNoNeedBufferVarsInferer);
REGISTER_OP_CPU_KERNEL(sequence_expand_as,
                       ops::SequenceExpandAsKernel<phi::CPUContext, float>,
                       ops::SequenceExpandAsKernel<phi::CPUContext, double>,
                       ops::SequenceExpandAsKernel<phi::CPUContext, int>,
                       ops::SequenceExpandAsKernel<phi::CPUContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    sequence_expand_as_grad,
    ops::SequenceExpandAsGradKernel<phi::CPUContext, float>,
    ops::SequenceExpandAsGradKernel<phi::CPUContext, double>,
    ops::SequenceExpandAsGradKernel<phi::CPUContext, int>,
    ops::SequenceExpandAsGradKernel<phi::CPUContext, int64_t>);
