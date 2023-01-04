/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class KronOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type =
        OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "X", "Y");
    return phi::KernelKey(data_type, ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const phi::KernelKey& expected_kernel_type) const override {
    if (framework::IsComplexType(expected_kernel_type.dtype())) {
      // only promote inputs’s types when contains complex input
      return phi::KernelKey(tensor.place(), tensor.layout(), tensor.dtype());
    } else {
      return phi::KernelKey(
          tensor.place(), tensor.layout(), expected_kernel_type.dtype());
    }
  }
};

class KronOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), the first operand of kron op");
    AddInput("Y", "(Tensor), the second operand of kron op");
    AddOutput("Out", "(Tensor), the output of kron op.");
    AddComment(R"DOC(
          Kron Operator.

          This operator computes the Kronecker product of two tensors, a
          composite tensor made of blocks of the second tensor scaled by the
          first.

          This operator assumes that the rank of the two tensors, $X$ and $Y$
          are the same, if necessary prepending the smallest with ones. If the
          shape of $X$ is [$r_0$, $r_1$, ..., $r_N$] and the shape of $Y$ is
          [$s_0$, $s_1$, ..., $s_N$], then the shape of the output tensor is
          [$r_{0}s_{0}$, $r_{1}s_{1}$, ..., $r_{N}s_{N}$]. The elements are
          products of elements from $X$ and $Y$.

          The equation is:
          $$
          output[k_{0}, k_{1}, ..., k_{N}] = X[i_{0}, i_{1}, ..., i_{N}] *
          Y[j_{0}, j_{1}, ..., j_{N}]
          $$

          where
          $$
          k_{t} = i_{t} * s_{t} + j_{t}, t = 0, 1, ..., N
          $$
        )DOC");
  }
};

class KronGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "kron_grad");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "kron_grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   framework::GradVarName("Out"),
                   "kron_grad");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, ctx->GetInputDim("X"));
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, ctx->GetInputDim("Y"));
    }
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto out_grad_name = framework::GradVarName("Out");
    return phi::KernelKey(
        OperatorWithKernel::IndicateVarDataType(ctx, out_grad_name),
        ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const phi::KernelKey& expected_kernel_type) const override {
    if (framework::IsComplexType(expected_kernel_type.dtype())) {
      // only promote inputs’s types when contains complex input
      return phi::KernelKey(tensor.place(), tensor.layout(), tensor.dtype());
    } else {
      return phi::KernelKey(
          tensor.place(), tensor.layout(), expected_kernel_type.dtype());
    }
  }
};

template <typename T>
class KronGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("kron_grad");

    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput("Y", this->Input("Y"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));

    grad_op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(kron,
                            KronInferShapeFunctor,
                            PD_INFER_META(phi::KronInferMeta));
REGISTER_OPERATOR(kron,
                  ops::KronOp,
                  ops::KronOpMaker,
                  ops::KronGradOpMaker<paddle::framework::OpDesc>,
                  ops::KronGradOpMaker<paddle::imperative::OpBase>,
                  KronInferShapeFunctor);
REGISTER_OPERATOR(kron_grad, ops::KronGradOp);
