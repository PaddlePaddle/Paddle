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

#include "paddle/fluid/operators/filter_by_instag_op.h"

#include <memory>

#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"
#include "paddle/fluid/framework/var_type_inference.h"

namespace paddle {
namespace operators {
class FilterByInstagOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Ins"),
        true,
        platform::errors::InvalidArgument("Input(Ins) should be not null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Ins_tag"),
                      true,
                      platform::errors::InvalidArgument(
                          "Input(Ins_tag) should be not null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Filter_tag"),
                      true,
                      platform::errors::InvalidArgument(
                          "Input(Filter_tag) should be not null."));

    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"),
        true,
        platform::errors::InvalidArgument("Output(Out) should be not null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("LossWeight"),
                      true,
                      platform::errors::InvalidArgument(
                          "Output(LossWeight) shoudl not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("IndexMap"),
                      true,
                      platform::errors::InvalidArgument(
                          "Output(IndexMap) should be not null."));

    auto x1_dims = ctx->GetInputDim("Ins");  // batch_size * vec

    ctx->SetOutputDim("Out", phi::make_ddim({-1, x1_dims[1]}));
    ctx->SetOutputDim("LossWeight", phi::make_ddim({-1, 1}));
    ctx->SetOutputDim("IndexMap", phi::make_ddim({-1, 2}));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Ins");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class FilterByInstagOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Ins", "(LoDTensor) embeded tensor");
    AddInput("Ins_tag", "(LoDTensor) ins tag list");
    AddInput("Filter_tag", "(1D Tensor) filter tag list");
    AddAttr<bool>("is_lod", "is Ins with LoD info or not, default True");
    AddAttr<int64_t>("out_val_if_empty",
                     "if the output after filter is empty, the output value")
        .SetDefault(0);
    AddOutput("Out", "(LoDTensor) embeded tensor filtered by instag");
    AddOutput("LossWeight", "(Tensor) loss weight.");
    AddOutput("IndexMap", "(LoDTensor) mapping from Out rows to X1 rows");
    AddComment(R"DOC(
Filter By Instag Op

This operator is used to filter embeded ins.

There are 3 inputs. First is embeded ins, Second is tags for ins,
Third is tags to filter.

There are 3 outputs. First is filtered embeded ins, Second is Loss Weight,
Third is the IndexMap from Out line number to X1 line number.
)DOC");
  }
};

class FilterByInstagOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("IndexMap"),
                      true,
                      platform::errors::InvalidArgument(
                          "Input(IndexMap) should be not null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")),
                      true,
                      platform::errors::InvalidArgument(
                          "Grad Input(Out) should be not null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Ins"),
        true,
        platform::errors::InvalidArgument("Input(Ins) should be not null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput("LossWeight"),
                      true,
                      platform::errors::InvalidArgument(
                          "Input(LossWeight) should be not null"));
    PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("Ins")),
                      true,
                      platform::errors::InvalidArgument(
                          "Grad Output(Ins) should be not null"));

    auto grad_out_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    auto x1_dims = ctx->GetInputDim("Ins");
    ctx->SetOutputDim(framework::GradVarName("Ins"),
                      phi::make_ddim({x1_dims[0], grad_out_dims[1]}));
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
class FilterByInstagGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("filter_by_instag_grad");
    op->SetInput("IndexMap", this->Output("IndexMap"));
    op->SetInput("Ins", this->Input("Ins"));
    op->SetAttrMap(this->Attrs());
    op->SetInput("LossWeight", this->Output("LossWeight"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("Ins"), this->InputGrad("Ins"));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(filter_by_instag,
                  ops::FilterByInstagOp,
                  ops::FilterByInstagOpMaker,
                  ops::FilterByInstagGradOpMaker<paddle::framework::OpDesc>,
                  ops::FilterByInstagGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(filter_by_instag_grad, ops::FilterByInstagOpGrad);

REGISTER_OP_CPU_KERNEL(filter_by_instag,
                       ops::FilterByInstagKernel<float>,
                       ops::FilterByInstagKernel<double>,
                       ops::FilterByInstagKernel<int32_t>,
                       ops::FilterByInstagKernel<int64_t>);

REGISTER_OP_CPU_KERNEL(filter_by_instag_grad,
                       ops::FilterByInstagGradKernel<float>,
                       ops::FilterByInstagGradKernel<double>,
                       ops::FilterByInstagGradKernel<int32_t>,
                       ops::FilterByInstagGradKernel<int64_t>);
