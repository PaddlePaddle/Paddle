/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/sequence_ops/sequence_expand_op.h"
#include <memory>

namespace paddle {
namespace operators {

using framework::LoDTensor;

class SequenceExpandOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SequenceExpand");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "SequenceExpand");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "SequenceExpand");

    auto x_dims = ctx->GetInputDim("X");
    auto out_dims = x_dims;
    int ref_level = ctx->Attrs().Get<int>("ref_level");

    PADDLE_ENFORCE_GE(
        x_dims.size(), 2,
        platform::errors::InvalidArgument(
            "Dimension number of Input(X) should be at least 2. But "
            "received: input rank %u, input shape [%s].",
            x_dims.size(), x_dims));

    if (ctx->IsRuntime()) {
      framework::Variable* x_var =
          BOOST_GET(framework::Variable*, ctx->GetInputVarPtrs("X")[0]);
      framework::Variable* y_var =
          BOOST_GET(framework::Variable*, ctx->GetInputVarPtrs("Y")[0]);

      auto& x_lod = x_var->Get<LoDTensor>().lod();
      auto& y_lod = y_var->Get<LoDTensor>().lod();

      PADDLE_ENFORCE_LE(x_lod.size(), 1UL,
                        platform::errors::InvalidArgument(
                            "Level of Input(X)'s lod should not be "
                            "greater than 1. But received: lod level %u.",
                            x_lod.size()));
      PADDLE_ENFORCE_GT(
          y_lod.size(), 0UL,
          platform::errors::InvalidArgument(
              "Level of Input(Y)'s lod should be greater than 0. But "
              "received: lod level %u.",
              y_lod.size()));
      PADDLE_ENFORCE_EQ(
          ref_level == -1 ||
              (ref_level >= 0 && ref_level < static_cast<int>(y_lod.size())),
          true, platform::errors::InvalidArgument(
                    "Invlid `ref_level`, which should be either equal to -1 "
                    "or in [0, %d), but received `ref_level` = %u.",
                    y_lod.size(), ref_level));

      if (ref_level == -1) ref_level = y_lod.size() - 1;

      if (x_lod.size() > 0) {
        PADDLE_ENFORCE_EQ(
            x_lod[0].size(), y_lod[ref_level].size(),
            platform::errors::InvalidArgument(
                "Level number of Input(X)'s lod could be 0. Otherwise "
                "size of Input(X)'s first level lod should be equal to "
                "size of Input(Y)'s referred level lod. But received: "
                "Input(X).lod[0].size() = %u, Input(Y).lod[%d].size() = "
                "%u",
                x_lod[0].size(), ref_level, y_lod[ref_level].size()));
      } else {
        PADDLE_ENFORCE_EQ(
            x_dims[0], static_cast<int64_t>(y_lod[ref_level].size()) - 1,
            platform::errors::InvalidArgument(
                "When Input(X)'s lod is null, the dims[0] of "
                "Input(X) should match the "
                "size of Input(Y)'s referred level lod. But received "
                "Input(X): input rank %u, input shape [%s]; received "
                "Input(Y).lod[%d].size() - 1 = %d.",
                x_dims.size(), x_dims, ref_level,
                static_cast<int64_t>(y_lod[ref_level].size()) - 1));
      }

      int64_t out_first_dim = 0;
      if (y_lod[ref_level].size() <= 1) {
        out_first_dim = x_dims[0];
      } else {
        for (size_t i = 1; i < y_lod[ref_level].size(); ++i) {
          int x_seq_len = 1;
          if (x_lod.size() == 1) {
            x_seq_len = x_lod[0][i] - x_lod[0][i - 1];
          }
          out_first_dim +=
              (y_lod[ref_level][i] - y_lod[ref_level][i - 1]) * x_seq_len;
        }
      }
      out_dims[0] = out_first_dim;
    } else {
      out_dims[0] = -1;
    }
    ctx->SetOutputDim("Out", out_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class SequenceExpandOpMaker : public framework::OpProtoAndCheckerMaker {
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
    AddAttr<int>("ref_level", "Specify lod level of Input(Y).").SetDefault(-1);
    AddComment(R"DOC(
Sequence Expand Operator.

This operator expands `X` according to specified level lod of `Y`. Current
implementation constaints that lod level of `X` should be at most 1. Attribute
`ref_level` is used to specify which level lod of `Y` is referred to expand `X`.
If set `ref_level` to -1, then last level lod of `Y` would be referred.
Please note, rank of `X` should be at least 2, when the rank exceeds 2, `X`
would be viewed as a 2-D tensor.

Following are cases to better explain how this works:

Case 1:

Given a 1-level LoDTensor input(X)
    X.lod =  [[0,   2,        4]]
    X.data = [[a], [b], [c], [d]]
    X.dims = [4, 1]
and input(Y)
    Y.lod = [[0,    2,    4],
             [0, 3, 6, 7, 8]]
ref_level: 0
then we get 1-level LoDTensor
    Out.lod =  [[0,   2,        4,        6,        8]]
    Out.data = [[a], [b], [a], [b], [c], [d], [c], [d]]
    Out.dims = [8, 1]

Case 2:

Given 1-level LoDTensor input(X)
    X.lod =  [[0,   1,        4]]
    X.data = [[a], [b], [c], [d]]
    X.dims = [4, 1]
and input(Y)
    Y.lod = [[0,    2,    4],
             [0, 3, 6, 6, 8]]
ref_level: 0
then we get 1-level LoDTensor
    Out.lod =  [[0,   1,   2,        5,             8]]
    Out.data = [[a], [a], [b], [c], [d], [b], [c], [d]]
    Out.dims = [8, 1]

Case 3:

Given a common Tensor input(X)
    X.data = [[a], [b], [c]]
    X.dims = [3, 1]
and input(Y)
    Y.lod = [[0, 2, 3, 6]]
ref_level: -1
then we get a common Tensor
    Out.data = [[a], [a], [b], [c], [c], [c]]
    Out.dims = [6, 1]

Case 4:

Given a common Tensor input(X)
    X.data = [[a, b], [c, d], [e, f]]
    X.dims = [3, 2]
and input(Y)
    Y.lod = [[0, 2, 3, 6]]
ref_level: 0
then we get a common LoDTensor
    Out.data = [[a, b], [a, b] [c, d], [e, f], [e, f], [e, f]]
    Out.dims = [6, 2]

)DOC");
  }
};

class SequenceExpandOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SequenceExpandOpGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "SequenceExpandOpGrad");

    auto x_dims = ctx->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
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
class SequenceExpandOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("sequence_expand_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(SequenceExpandOpNoNeedBufferVarsInference,
                                    "Y");
DECLARE_NO_NEED_BUFFER_VARS_INFERER(
    SequenceExpandGradOpNoNeedBufferVarsInference, "X", "Y");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sequence_expand, ops::SequenceExpandOp,
                  ops::SequenceExpandOpMaker,
                  ops::SequenceExpandOpGradMaker<paddle::framework::OpDesc>,
                  ops::SequenceExpandOpGradMaker<paddle::imperative::OpBase>,
                  ops::SequenceExpandOpNoNeedBufferVarsInference);
REGISTER_OPERATOR(sequence_expand_grad, ops::SequenceExpandOpGrad,
                  ops::SequenceExpandGradOpNoNeedBufferVarsInference);
REGISTER_OP_CPU_KERNEL(
    sequence_expand,
    ops::SequenceExpandKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SequenceExpandKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SequenceExpandKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SequenceExpandKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    sequence_expand_grad,
    ops::SequenceExpandGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SequenceExpandGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SequenceExpandGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SequenceExpandGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
