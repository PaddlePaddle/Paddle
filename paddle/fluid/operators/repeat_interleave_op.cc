// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/repeat_interleave_op.h"
#include <memory>

namespace paddle {
namespace operators {

using framework::Tensor;

class RepeatInterleaveOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::InvalidArgument(
            "Input(X) of RepeatInterleaveOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"), true,
        platform::errors::InvalidArgument(
            "Output(Out) of RepeatInterleaveOp should not be null."));

    auto input_dim = ctx->GetInputDim("X");
    auto dim = ctx->Attrs().Get<int>("dim");
    auto output_dim = phi::vectorize(input_dim);
    PADDLE_ENFORCE_EQ(
        dim < input_dim.size() && dim >= (0 - input_dim.size()), true,
        platform::errors::OutOfRange(
            "Attr(dim) is out of range, It's expected "
            "to be in range of [-%d, %d]. But received Attr(dim) = %d.",
            input_dim.size(), input_dim.size() - 1, dim));

    auto repeats = ctx->Attrs().Get<int>("Repeats");
    if (ctx->HasInput("RepeatsTensor")) {
      auto repeats_dim = ctx->GetInputDim("RepeatsTensor");

      PADDLE_ENFORCE_EQ(
          repeats_dim.size() == 1 ||
              (repeats_dim.size() == 2 && repeats_dim[1] == 1),
          true, platform::errors::InvalidArgument(
                    "The 'shape' of Input(RepeatsTensor) must be 1-D tensor. "
                    "But received: the 'shape' of Input(Index) is [%s], "
                    "the dimension of Input(Index) is [%d].",
                    repeats_dim, repeats_dim.size()));

      PADDLE_ENFORCE_EQ(repeats_dim[0] != 0, true,
                        platform::errors::InvalidArgument(
                            "The length of Input(RepeatsTensor) can't be 0."));

      if (dim < 0) {
        dim += input_dim.size();
      }
      output_dim[dim] = -1;
    } else if (repeats > 0) {
      output_dim[dim] = input_dim[dim] * repeats;
    }
    VLOG(3) << "infershap out " << output_dim[dim];
    ctx->SetOutputDim("Out", phi::make_ddim(output_dim));
    auto type = ctx->GetInputsVarType("X")[0];
    if (type == framework::proto::VarType::LOD_TENSOR) {
      ctx->ShareLoD("X", /*->*/ "Out");
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class RepeatInterleaveGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      platform::errors::InvalidArgument(
                          "Input(Out@GRAD) should be not null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("X")), true,
                      platform::errors::InvalidArgument(
                          "Output(X@GRAD) should be not null."));

    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

class RepeatInterleaveOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) the input tensor.");
    AddInput("RepeatsTensor",
             "the 1-D tensor containing the repeats alongsize the axis.")
        .AsDispensable();
    AddOutput("Out", "the output tensor.");
    AddAttr<int>("Repeats", "the number of repetitions for each element.")
        .SetDefault(0);
    AddAttr<int>("dim", "the dimension in which we repeat.").SetDefault(0);
    AddComment(R"DOC(
Returns a new tensor which repeats the input tensor
along dimension dim using the entries in repeats which
is a Tensor or int.

The returned tensor has the same number of dimensions
as the original tensor (input), except along the given axis.
    )DOC");
  }
};

template <typename T>
class RepeatInterleaveGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("repeat_interleave_grad");

    op->SetInput("X", this->Input("X"));
    op->SetInput("RepeatsTensor", this->Input("RepeatsTensor"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(RepeatInterleaveGradNoNeedBufferVarsInferer,
                                    "X");
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(repeat_interleave, ops::RepeatInterleaveOp,
                  ops::RepeatInterleaveOpMaker,
                  ops::RepeatInterleaveGradMaker<paddle::framework::OpDesc>,
                  ops::RepeatInterleaveGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(repeat_interleave_grad, ops::RepeatInterleaveGradOp,
                  ops::RepeatInterleaveGradNoNeedBufferVarsInferer);
REGISTER_OP_CPU_KERNEL(
    repeat_interleave,
    ops::RepeatInterleaveKernel<paddle::platform::CPUDeviceContext, float>,
    ops::RepeatInterleaveKernel<paddle::platform::CPUDeviceContext, double>,
    ops::RepeatInterleaveKernel<paddle::platform::CPUDeviceContext, int>,
    ops::RepeatInterleaveKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    repeat_interleave_grad,
    ops::RepeatInterleaveGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::RepeatInterleaveGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::RepeatInterleaveGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::RepeatInterleaveGradKernel<paddle::platform::CPUDeviceContext,
                                    int64_t>);
