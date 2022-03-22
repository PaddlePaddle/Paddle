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

#include <memory>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"

#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class MultiplexOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class MultiplexOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Ids",
             "Tensor<int32>, index variable which is a 2-D tensor with shape "
             "[M, 1] where M is the batch size.");
    AddInput("X",
             "A list of variables to gather from. All variables have the same "
             "shape and the rank is at least 2.")
        .AsDuplicable();
    AddOutput("Out", "The output tensor of multiplex operator.");
    AddComment(R"DOC(
Referring to the given index variable, this layer selects rows from the
input variables to construct a multiplex variable. Assuming that there are
:math:`m` input variables and :math:`I_i` represents the i-th input
variable and :math:`i` is in [0, :math:`m`). All input variables are
tensors with same shape [:math:`d_0`, :math:`d_1`, ..., :math:`d_R`].
Please note that rank of the input tensor should be at least 2. Each input
variable will be treated as a 2-D matrix with shape [:math:`M`, :math:`N`]
where :math:`M` for :math:`d_0` and :math:`N` for :math:`d_1` * :math:`d_2`
* ... * :math:`d_R`. Let :math:`I_i[j]` be the j-th row of the i-th input
variable. The given index variable should be a 2-D tensor with shape
[:math:`M`, 1]. Let `ID[i]` be the i-th index value of the index variable.
Then the output variable will be a tensor with shape [:math:`d_0`,
:math:`d_1`, ..., :math:`d_R`]. If we treat the output tensor as a 2-D
matrix with shape [:math:`M`, :math:`N`] and let :math:`O[i]` be the i-th
row of the matrix, then `O[i]` is equal to :math:`I_{ID[i]}[i]`.

* Ids: the index tensor.

* X[0 : N - 1]: the candidate tensors for output (N >= 2).

* For each index i from 0 to batchSize - 1, the output is the i-th row of the
the (Ids[i])-th tensor.

For i-th row of the output tensor:

$$
y[i] = x_{k}[i]
$$

where $y$ is the output tensor, $x_{k}$ is the k-th input tensor,
and $k = Ids[i]$.

)DOC");
  }
};

class MultiplexGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto dxs = ctx->Outputs(framework::GradVarName("X"));
    PADDLE_ENFORCE_NE(dxs.empty(), true,
                      platform::errors::InvalidArgument(
                          "Output(X@Grad) should not be null."));
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "MultiplexGrad");
    auto dout_dim = ctx->GetInputDim(framework::GradVarName("Out"));
    ctx->SetOutputsDim(framework::GradVarName("X"),
                       std::vector<framework::DDim>(dxs.size(), dout_dim));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class MultiplexGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("multiplex_grad");
    op->SetInput("Ids", this->Input("Ids"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X", false));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(multiplex, MultiplexInferShapeFunctor,
                            PD_INFER_META(phi::MultiplexInferMeta));

REGISTER_OPERATOR(multiplex, ops::MultiplexOp, ops::MultiplexOpMaker,
                  ops::MultiplexGradMaker<paddle::framework::OpDesc>,
                  ops::MultiplexGradMaker<paddle::imperative::OpBase>,
                  MultiplexInferShapeFunctor);
REGISTER_OPERATOR(multiplex_grad, ops::MultiplexGradOp);
