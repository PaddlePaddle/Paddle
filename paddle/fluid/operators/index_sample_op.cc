/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/index_sample_op.h"
#include <vector>
#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
class IndexSampleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input(Tensor), dtype support int32/int64/float/double");
    AddInput("Index", "Index(Tensor), dtype support int32/int64");
    AddOutput("Out", "Return the element of input at index");

    AddComment(R"DOC(
    IndexSample OP returns the element of the specified location of X, 
    and the location is specified by Index. 

    X tensor and Index tensor's shape must be 2-D, 
    dimension at 0 which usually is batch size must be equal.

    The returned tensor has the same shape and dimensions as the Index tensor.
    )DOC");
  }
};

class IndexSampleOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Inputs(Input) of FindByIndex should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Index"), true,
                      platform::errors::InvalidArgument(
                          "Inputs(Index) of FindByIndex should not be null."));

    auto input_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(
        input_dims.size(), 2,
        platform::errors::InvalidArgument(
            "Inputs(X) shape of IndexSample op should be 2-D, but "
            "got X's shape = [%s], please check X shape.",
            input_dims));

    auto index_dims = ctx->GetInputDim("Index");
    PADDLE_ENFORCE_EQ(
        input_dims.size(), 2,
        platform::errors::InvalidArgument(
            "Inputs(Index) shape of IndexSample op should be 2-D, but "
            "got Index's shape [%s] , please check index shape.",
            input_dims));
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(input_dims[0], index_dims[0],
                        platform::errors::InvalidArgument(
                            "Inputs(X)'s value of dimension 0 must same with "
                            "Inputs(Index)'s value of dimension 0, but "
                            "got %d of Inputs(X), and got %d of Inputs(Index), "
                            "please check Inputs shape.",
                            input_dims[0], index_dims[0]));
    }
    ctx->SetOutputDim("Out", index_dims);
    auto type = ctx->GetInputsVarType("Index")[0];
    if (type == framework::proto::VarType::LOD_TENSOR) {
      ctx->ShareLoD("Index", /*->*/ "Out");
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class IndexSampleGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Index"), true,
        platform::errors::InvalidArgument("Input(Index) should be not null."));
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
    auto data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

template <typename T>
class IndexSampleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("index_sample_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Index", this->Input("Index"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(IndexSampleGradNoNeedBufferVarInferer, "X");
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(index_sample, ops::IndexSampleOp, ops::IndexSampleOpMaker,
                  ops::IndexSampleGradMaker<paddle::framework::OpDesc>,
                  ops::IndexSampleGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(index_sample_grad, ops::IndexSampleGradOp,
                  ops::IndexSampleGradNoNeedBufferVarInferer);
REGISTER_OP_CPU_KERNEL(
    index_sample,
    ops::IndexSampleKernel<paddle::platform::CPUDeviceContext, float>,
    ops::IndexSampleKernel<paddle::platform::CPUDeviceContext, double>,
    ops::IndexSampleKernel<paddle::platform::CPUDeviceContext, int>,
    ops::IndexSampleKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    index_sample_grad,
    ops::IndexSampleGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::IndexSampleGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::IndexSampleGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::IndexSampleGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
