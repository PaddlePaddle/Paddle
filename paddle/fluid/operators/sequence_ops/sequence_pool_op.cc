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
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(
                              ctx, framework::GradVarName("Out")),
                          ctx.GetPlace());
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
REGISTER_OPERATOR(sequence_pool_grad,
                  ops::SequencePoolGradOp,
                  ops::SequencePoolGradOpNoNeedBufferVarsInferer);

REGISTER_OP_CPU_KERNEL(sequence_pool_grad,
                       ops::SequencePoolGradKernel<phi::CPUContext, float>,
                       ops::SequencePoolGradKernel<phi::CPUContext, double>);
