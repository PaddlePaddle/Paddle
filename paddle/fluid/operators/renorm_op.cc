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

#include "paddle/fluid/operators/renorm_op.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

class RenormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  using DDim = paddle::framework::DDim;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "abs");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "abs");

    auto in_dims = ctx->GetInputDim("X");

    ctx->SetOutputDim("Out", in_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class RenormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of renorm op.");
    AddOutput("Out", "(Tensor), The output tensor of renorm op.");
    AddAttr<float>("p", "(float, norm's power");
    AddAttr<int>("axis",
                 "int,the dimension to slice over to get the sub-tensors");
    AddAttr<float>("max_norm", "(float, the norm upper-bound");
    AddAttr<bool>("use_cudnn",
                  "(bool, default false) Only used in cudnn kernel, need "
                  "install cudnn")
        .SetDefault(false)
        .AsExtra();
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false)
        .AsExtra();
    AddComment(R"DOC(
Renorm Operator.

This operator is used to scale tensor sliced by axis if its p-norm execeeds maxnorm

)DOC");
  }
};

class RenormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@Grad", "AbsGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   "X@Grad", "AbsGrad");

    auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    ctx->SetOutputDim(framework::GradVarName("X"), dout_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(dtype, ctx.GetPlace());
  }
};

template <typename T>
class RenormGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("renorm_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetInput("X", this->Input("X"));
    retv->SetAttrMap(this->Attrs());
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(renorm, ops::RenormOp, ops::RenormOpMaker,
                  ops::RenormGradMaker<paddle::framework::OpDesc>,
                  ops::RenormGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(renorm_grad, ops::RenormGradOp);

REGISTER_OP_CPU_KERNEL(renorm, ops::CPURenormKernel<float>,
                       ops::CPURenormKernel<double>);

REGISTER_OP_CPU_KERNEL(renorm_grad, ops::CPURenormGradKernel<float>,
                       ops::CPURenormGradKernel<double>);
