// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/shuffle_batch_op.h"

#include <memory>

#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"
#include "paddle/fluid/framework/var_type_inference.h"

namespace paddle {
namespace operators {
class ShuffleBatchOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"),
        true,
        platform::errors::NotFound("Input(X) should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Seed"),
        true,
        platform::errors::NotFound("Input(Seed) should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"),
        true,
        platform::errors::NotFound("Output(Out) should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("ShuffleIdx"),
        true,
        platform::errors::NotFound("Output(ShuffleIdx) should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("SeedOut"),
        true,
        platform::errors::NotFound("Output(SeedOut) should not be null."));

    ctx->ShareDim("X", "Out");
    ctx->ShareLoD("X", "Out");
    ctx->ShareDim("Seed", "SeedOut");
    ctx->ShareLoD("Seed", "SeedOut");
    ctx->SetOutputDim("ShuffleIdx", phi::make_ddim({-1}));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "Seed") {
      return expected_kernel_type;
    }
    return framework::OperatorWithKernel::GetKernelTypeForVar(
        var_name, tensor, expected_kernel_type);
  }
};

class ShuffleBatchOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(LoDTensor) The input tensor of shuffle_batch op.");
    AddInput("Seed", "(LoDTensor) The input seed tensor.");
    AddAttr<int>(
        "startup_seed",
        "If input tensor 'Seed' is not initialized, the 'startup_seed' "
        "will be used to replace it. The seed after shuffle batch will "
        "be saved in 'SeedOut'. ")
        .SetDefault(0);
    AddOutput("Out", "(LoDTensor) The output tensor of shuffle_batch op.");
    AddOutput("ShuffleIdx", "(Tensor) Record forword shuffle order");
    AddOutput("SeedOut", "(LoDTensor) Saved new generated seed.");
    AddComment(R"DOC(
Shuffle Batch Operator.

This operator is used to shuffle input $X$'s elements.

There is 2 input. The product of input dims (except last dim) numbers of elements will be shuffled. $Seed$ is tensor of seed.

There are 3 outputs. $Out$ is shuffled tensor of input. $ShuffleIdx$ is the tensor used to record shuffle order. $SeedOut$ is same tensor of $Seed$.
)DOC");
  }
};

class ShuffleBatchOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("ShuffleIdx"),
        true,
        platform::errors::NotFound("Input(ShuffleIdx) should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Out")),
        true,
        platform::errors::NotFound("Grad Input(Out) should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput(framework::GradVarName("X")),
        true,
        platform::errors::NotFound("Grad Output(X) should not be null"));

    ctx->ShareDim(framework::GradVarName("Out"), framework::GradVarName("X"));
    ctx->ShareLoD(framework::GradVarName("Out"), framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

template <typename T>
class ShuffleBatchGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("shuffle_batch_grad");
    op->SetInput("ShuffleIdx", this->Output("ShuffleIdx"));
    op->SetAttrMap(this->Attrs());
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(shuffle_batch,
                  ops::ShuffleBatchOp,
                  ops::ShuffleBatchOpMaker,
                  ops::ShuffleBatchGradOpMaker<paddle::framework::OpDesc>,
                  ops::ShuffleBatchGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(shuffle_batch_grad, ops::ShuffleBatchOpGrad);

REGISTER_OP_CPU_KERNEL(shuffle_batch,
                       ops::ShuffleBatchKernel<float>,
                       ops::ShuffleBatchKernel<double>,
                       ops::ShuffleBatchKernel<int32_t>,
                       ops::ShuffleBatchKernel<int64_t>);

REGISTER_OP_CPU_KERNEL(shuffle_batch_grad,
                       ops::ShuffleBatchGradKernel<float>,
                       ops::ShuffleBatchGradKernel<double>,
                       ops::ShuffleBatchGradKernel<int32_t>,
                       ops::ShuffleBatchGradKernel<int64_t>);
