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

#include "paddle/fluid/operators/sequence_ops/sequence_pool_op.h"

#include <memory>
#include <string>

namespace paddle {
namespace operators {

class SequencePoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SequencePool");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "SequencePool");

    if (!ctx->IsRuntime()) {
      // Check the lod_level for compile-time.
      auto in_lod_level = ctx->GetLoDLevel("X");
      PADDLE_ENFORCE_GT(
          in_lod_level,
          0,
          platform::errors::InvalidArgument("The LoD level of Input(X) should "
                                            "be larger than 0, but received: "
                                            "lod level %u.",
                                            in_lod_level));
      ctx->SetLoDLevel("Out", in_lod_level - 1);
    }

    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    if (ctx->Attrs().Get<std::string>("pooltype") == "MAX") {
      OP_INOUT_CHECK(
          ctx->HasOutput("MaxIndex"), "Output", "MaxIndex", "SequencePool");
      ctx->SetOutputDim("MaxIndex", ctx->GetInputDim("X"));
    }
  }
};

class SequencePoolOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(phi::DenseTensor) The variable-length input of SequencePoolOp");
    AddOutput(
        "Out",
        "(phi::DenseTensor) The output of SequencePoolOp does not contain LoD "
        "information.");
    AddOutput("MaxIndex",
              "(phi::DenseTensor<int>) This tensor is used for the sequence "
              "max-pooling "
              "to record the max indexes.")
        .AsIntermediate();
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false)
        .AsExtra();
    AddAttr<std::string>(
        "pooltype",
        "(string, default 'AVERAGE') the pooling pooltype of SequencePoolOp.")
        .SetDefault("AVERAGE")
        .InEnum({"AVERAGE", "SUM", "SQRT", "LAST", "FIRST", "MAX"});
    AddAttr<float>("pad_value",
                   "(float, default 0.0) The value to pad for empty sequence.")
        .SetDefault(0.0);
    AddComment(R"DOC(
Sequence Pool Operator.

The SequencePoolOp pools features of all time-steps of each instance.
It supports six pooling types:
1. AVERAGE: $$Out[i] = \frac{\sum_i X_i}{N}$$
2. SUM:     $$Out[i] = \sum_jX_{ij}$$
3. SQRT:    $$Out[i] = \frac{\sum_jX_{ij}}{\sqrt{len(X_i)}}$$
4. LAST:    Out[i] = last instance in i-th sequence X[i]
5. FIRST:   Out[i] = first instance in i-th sequence X[i]
6. MAX:     $$Out[i] = max(X_i)$$

and for the empty sequence Out[i] = attr(pad_value).

The following example explains how this works:
For a mini-batch of 3 variable-length sentences,
containing 2, 3, and 2 time-steps:

Assume X is a [7,M,N] phi::DenseTensor, and X->lod()[0] = [0, 2, 5, 7], 7=2+3+2.
Besides, for the sake of simplicity, we assume M=1 and N=1,
and the value of X = [[1, 3], [2, 4, 6], [5, 1]].

Thus, Out is a [3,1,1] phi::DenseTensor without LoD information.
And for different pooltype, the value of Out is as follows:

- AVERAGE: [2, 4, 3], where 2=(1+3)/2, 4=(2+4+6)/3, 3=(5+1)/2
- SUM: [4, 12, 6], where 4=1+3, 12=2+4+6, 6=5+1
- SQRT: [2.82, 6.93, 4.24], where 2.82=(1+3)/sqrt(2),
           6.93=(2+4+6)/sqrt(3), 4.24=(5+1)/sqrt(2)
- MAX: [3, 6, 5], where 3=max(1,3), 6=max(2,4,6), 5=max(5,1)
- LAST: [3, 6, 1], where 3=last(1,3), 6=last(2,4,6), 1=last(5,1)
- FIRST: [1, 2, 5], where 1=first(1,3), 2=first(2,4,6), 5=first(5,1)

    )DOC");
  }
};

class SequencePoolGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   framework::GradVarName("Out"),
                   "SequencePoolGrad");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SequencePoolGrad");

    auto og_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(og_dims.size(),
                      x_dims.size(),
                      platform::errors::InvalidArgument(
                          "The rank of output grad must equal to Input(X). But "
                          "received: input rank %u, input shape [%s].",
                          og_dims.size(),
                          og_dims));
    for (int64_t i = 1; i < og_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          og_dims[i],
          x_dims[i],
          platform::errors::InvalidArgument(
              "The dimension mismatch between Input(OUT@GRAD) and "
              "Input(X). Received Input(OUT@GRAD): input rank %u, "
              "input shape [%s]; received Input(X): input rank %u, "
              "input shape [%s].",
              og_dims.size(),
              og_dims,
              x_dims.size(),
              x_dims));
    }

    ctx->ShareDim("X", /*->*/ framework::GradVarName("X"));
    ctx->ShareLoD("X", /*->*/ framework::GradVarName("X"));
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
class SequencePoolGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op_desc_ptr) const override {
    op_desc_ptr->SetType("sequence_pool_grad");
    op_desc_ptr->SetInput("X", this->Input("X"));
    if (PADDLE_GET_CONST(std::string, this->GetAttr("pooltype")) == "MAX") {
      op_desc_ptr->SetInput("MaxIndex", this->Output("MaxIndex"));
    }
    op_desc_ptr->SetInput(framework::GradVarName("Out"),
                          this->OutputGrad("Out"));
    op_desc_ptr->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op_desc_ptr->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(SequencePoolGradOpNoNeedBufferVarsInferer,
                                    "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sequence_pool,
                  ops::SequencePoolOp,
                  ops::SequencePoolOpMaker,
                  ops::SequencePoolGradOpMaker<paddle::framework::OpDesc>,
                  ops::SequencePoolGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(sequence_pool_grad,
                  ops::SequencePoolGradOp,
                  ops::SequencePoolGradOpNoNeedBufferVarsInferer);
REGISTER_OP_CPU_KERNEL(sequence_pool,
                       ops::SequencePoolKernel<phi::CPUContext, float>,
                       ops::SequencePoolKernel<phi::CPUContext, double>);

REGISTER_OP_CPU_KERNEL(sequence_pool_grad,
                       ops::SequencePoolGradKernel<phi::CPUContext, float>,
                       ops::SequencePoolGradKernel<phi::CPUContext, double>);
