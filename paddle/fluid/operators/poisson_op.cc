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

#include <string>

#include "paddle/fluid/operators/poisson_op.h"

namespace paddle {
namespace operators {

class PoissonOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "PoissonOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "PoissonOp");

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

class PoissonOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input tensor of poisson op");
    AddOutput("Out",
              "The output tensor of poisson op, it has the same shape and "
              "dtype with input. Each element corresponds to input tensor");
    AddComment(R"DOC(
This operator generate random value that obey poisson distribution.
)DOC");
  }
};

class PoissonOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string> &GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

template <typename T>
class PoissonKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *x = ctx.Input<framework::Tensor>("X");
    auto *out = ctx.Output<framework::Tensor>("Out");

    const T *x_data = x->data<T>();
    T *out_data = out->mutable_data<T>(ctx.GetPlace());

    int64_t size = x->numel();

    auto gen = framework::DefaultCPUGenerator();
    auto engine = gen->GetCPUEngine();

    for (int64_t i = 0; i < size; ++i) {
      std::poisson_distribution<> dist(x_data[i]);
      out_data[i] = static_cast<T>(dist(*engine));
    }
  }
};

class PoissonGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out_Grad", "PoissonGradOp");

    auto dout_dim = ctx->GetInputDim(framework::GradVarName("Out"));
    ctx->SetOutputDim(framework::GradVarName("X"), dout_dim);
  }
};

template <typename T>
class PoissonGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("poisson_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(poisson, ops::PoissonOp, ops::PoissonOpMaker,
                  ops::PoissonOpInferVarType,
                  ops::PoissonGradOpMaker<paddle::framework::OpDesc>,
                  ops::PoissonGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(poisson_grad, ops::PoissonGradOp);

REGISTER_OP_CPU_KERNEL(poisson,
                       ops::PoissonKernel<plat::CPUDeviceContext, float>,
                       ops::PoissonKernel<plat::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(poisson_grad,
                       ops::PoissonGradKernel<plat::CPUDeviceContext, float>,
                       ops::PoissonGradKernel<plat::CPUDeviceContext, double>);
