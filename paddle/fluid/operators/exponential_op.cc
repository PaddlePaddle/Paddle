/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/exponential_op.h"

namespace paddle {
namespace operators {

class ExponentialOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ExponentialOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "ExponentialOp");
    auto dim = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", dim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class ExponentialOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(
This operator fills the input tensor with random values sampled from a
exponential distribution.
)DOC");
    AddInput("X", "The input tensor.");
    AddOutput("Out", "The output tensor of exponential OP.");
    AddAttr<float>(
        "lambda", "lambd parameter of exponential distribution. [default 1.0].")
        .SetDefault(1.0f);
  }
};

class ExponentialOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string> &GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

template <typename T>
class ExponentialKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *out = ctx.Output<framework::Tensor>("Out");
    T *out_data = out->mutable_data<T>(ctx.GetPlace());

    T lambda = static_cast<T>(ctx.Attr<float>("lambda"));
    int64_t size = out->numel();

    auto gen = framework::DefaultCPUGenerator();
    auto engine = gen->GetCPUEngine();

    std::uniform_real_distribution<T> uniform(0.0, 1.0);
    phi::funcs::exponential_transform<T> trans(lambda);
    for (int64_t i = 0; i < size; ++i) {
      out_data[i] = trans(uniform(*engine));
    }
  }
};

class ExponentialGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out_Grad", "ExponentialGradOp");

    auto dout_dim = ctx->GetInputDim(framework::GradVarName("Out"));
    ctx->SetOutputDim(framework::GradVarName("X"), dout_dim);
  }
};

template <typename T>
class ExponentialGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("exponential_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

DECLARE_INPLACE_OP_INFERER(ExponentialInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(ExponentialGradInferer,
                           {paddle::framework::GradVarName("Out"),
                            paddle::framework::GradVarName("X")});

REGISTER_OPERATOR(exponential, ops::ExponentialOp, ops::ExponentialOpMaker,
                  ops::ExponentialOpInferVarType,
                  ops::ExponentialGradOpMaker<paddle::framework::OpDesc>,
                  ops::ExponentialGradOpMaker<paddle::imperative::OpBase>,
                  ExponentialInferer);
REGISTER_OPERATOR(exponential_grad, ops::ExponentialGradOp,
                  ExponentialGradInferer);

REGISTER_OP_CPU_KERNEL(exponential,
                       ops::ExponentialKernel<plat::CPUDeviceContext, float>,
                       ops::ExponentialKernel<plat::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    exponential_grad, ops::ExponentialGradKernel<plat::CPUDeviceContext, float>,
    ops::ExponentialGradKernel<plat::CPUDeviceContext, double>);
