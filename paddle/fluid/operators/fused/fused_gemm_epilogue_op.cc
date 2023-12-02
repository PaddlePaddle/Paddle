/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/fusion.h"
#include "paddle/phi/kernels/funcs/fused_gemm_epilogue.h"

namespace paddle {
namespace operators {

class FusedGemmEpilogueOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return phi::KernelKey(data_type, ctx.GetPlace());
  }
};

class FusedGemmEpilogueOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor X of Out = Act((X * Y) + Bias).");
    AddInput("Y", "The input tensor Y of Out = Act((X * Y) + Bias).");
    AddInput("Bias", "The input tensor bias of Out = Act((X * Y) + Bias).");

    AddOutput("Out", "The output tensor Out of Out = Act((X * Y) + Bias).");
    AddOutput("ReserveSpace",
              R"DOC(Reserve GPU space to place
        auxiliary data pointer. It is used to pass auxiliary data pointer
        for fused_gemm_epilogue op. If not given (empty string), the
        auxiliary mode would not be enable.)DOC")
        .AsDispensable()
        .AsExtra();

    AddAttr<bool>(
        "trans_x",
        R"DOC((bool, default false), Whether to transpose input tensor X
    or not. The input tensor X coulbe be more than two dimension. When
    set trans_x=true, it would fully reverse X. For instant: X with shpae
    [d0, d1, d2, d3] -> [d3, d2, d1, d0].)DOC")
        .SetDefault(false);
    AddAttr<bool>(
        "trans_y",
        R"DOC((bool, default false), Whether to transpose input tensor Y
    or not. The input tensor Y should be two dimension. When
    set trans_y=true, it would transpose Y. For instant: Y with shpae
    [d0, d1] -> [d1, d0].)DOC")
        .SetDefault(false);

    AddAttr<std::string>(
        "activation",
        R"DOC((string, default none), The activation function. It could be
    one of {none, relu, gelu}. When none is given, Act would be null
    operations)DOC")
        .SetDefault("none");

    AddComment(R"DOC(
FusedGemmEpilogue Operator
This operator is used to perform Activeation(Elementwise_add(Matmul(X, Y), bias)).
It is equal to paddle.nn.Linear + Activation (None, ReLU or GeLU).

Note:
X could be more than two dimension and would be flatten to 2D for computing.
X with shape [d0, d1, d2, d3] -> X_2D with shape [d0*d1*d2, d3]
)DOC");
  }
};

class FusedGemmEpilogueGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return phi::KernelKey(data_type, ctx.GetPlace());
  }
};

template <typename T>
class FusedGemmEpilogueOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    const auto& act_type = this->template Attr<std::string>("activation");

    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    if (act_type != "none") {
      op->SetInput("ReserveSpace", this->Output("ReserveSpace"));
    }
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));

    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(fused_gemm_epilogue,
                            FusedGemmEpilogueInferShapeFunctor,
                            PD_INFER_META(phi::FusedGemmEpilogueInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(fused_gemm_epilogue_grad,
                            FusedGemmEpilogueGradInferShapeFunctor,
                            PD_INFER_META(phi::FusedGemmEpilogueGradInferMeta));
REGISTER_OPERATOR(fused_gemm_epilogue,
                  ops::FusedGemmEpilogueOp,
                  ops::FusedGemmEpilogueOpMaker,
                  ops::FusedGemmEpilogueOpGradMaker<paddle::framework::OpDesc>,
                  ops::FusedGemmEpilogueOpGradMaker<paddle::imperative::OpBase>,
                  FusedGemmEpilogueInferShapeFunctor);
REGISTER_OPERATOR(fused_gemm_epilogue_grad,
                  ops::FusedGemmEpilogueGradOp,
                  FusedGemmEpilogueGradInferShapeFunctor);
