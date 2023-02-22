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
#include <string>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/prim/api/composite_backward/composite_backward_api.h"
#include "paddle/fluid/prim/utils/static/composite_grad_desc_maker.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class GatherOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.device_context().GetPlace());
  }
  phi::KernelKey GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const phi::KernelKey& expected_kernel_type) const override {
    if (var_name == "Axis") {
      return phi::KernelKey(phi::Backend::ALL_BACKEND,
                            expected_kernel_type.layout(),
                            expected_kernel_type.dtype());
    }
    return phi::KernelKey(
        tensor.place(), tensor.layout(), expected_kernel_type.dtype());
  }
};

class GatherGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(
                              ctx, framework::GradVarName("Out")),
                          ctx.device_context().GetPlace());
  }
  phi::KernelKey GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const phi::KernelKey& expected_kernel_type) const override {
    if (var_name == "Axis") {
      return phi::KernelKey(phi::Backend::ALL_BACKEND,
                            expected_kernel_type.layout(),
                            expected_kernel_type.dtype());
    }
    return phi::KernelKey(
        tensor.place(), tensor.layout(), expected_kernel_type.dtype());
  }
};

class GatherOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The source input of gather op");
    AddInput("Index", "The index input of gather op");
    AddInput("Axis",
             "The Tensor which contains the axis that we do gather operation.")
        .AsDispensable();
    AddOutput("Out", "The output of gather op");
    AddAttr<int>(
        "axis",
        "The Tensor which contains the axis that we do gather operation.")
        .SetDefault(0);
    AddComment(R"DOC(
Gather Operator.

$Out = X[Index]$

Out is obtained by gathering entries of the outer-most dimension
of X indexed by Index and concatenate them together.

Example:

X = [[1, 2],
     [3, 4],
     [5, 6]]

Index = [[1, 2]]

Then:

Out = [[3, 4],
       [5, 6]]

)DOC");
  }
};

template <typename T>
class GatherGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("gather_grad");
    op->SetInput("Index", this->Input("Index"));
    op->SetInput("Axis", this->Input("Axis"));

    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

class GatherCompositeGradOpMaker : public prim::CompositeGradOpMakerBase {
 public:
  using prim::CompositeGradOpMakerBase::CompositeGradOpMakerBase;

 protected:
  void Apply() override {
    paddle::experimental::Tensor index = this->GetSingleForwardInput("Index");
    paddle::optional<paddle::experimental::Tensor> tensor_axis =
        this->GetOptionalSingleForwardInput("Axis");
    paddle::experimental::Tensor x = this->GetSingleForwardInput("X");
    paddle::experimental::Tensor dout = this->GetSingleOutputGrad("Out");
    paddle::experimental::Tensor dx = this->GetSingleInputGrad("X");
    auto* dx_ptr = this->GetOutputPtr(&dx);
    std::string dx_name = this->GetOutputName(*dx_ptr);
    int axis = static_cast<int>(this->Attr<int>("axis"));
    VLOG(3) << "Runing gather_grad composite func";
    if (tensor_axis.is_initialized()) {
      PADDLE_THROW(platform::errors::Unimplemented(
          "We don't support dynamic index from tensor for gather composite "
          "grad for now. "));
    } else {
      prim::gather_grad<prim::DescTensor>(x, index, dout, axis, false, dx_ptr);
    }
    this->RecoverOutputName(dx, dx_name);
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(GatherGradNoNeedBufferVarInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(gather,
                            GatherInferShapeFunctor,
                            PD_INFER_META(phi::GatherInferMeta));
REGISTER_OPERATOR(gather,
                  ops::GatherOp,
                  ops::GatherOpMaker,
                  ops::GatherGradOpMaker<paddle::framework::OpDesc>,
                  ops::GatherGradOpMaker<paddle::imperative::OpBase>,
                  ops::GatherCompositeGradOpMaker,
                  GatherInferShapeFunctor);
DECLARE_INFER_SHAPE_FUNCTOR(gather_grad,
                            GatherGradInferShapeFunctor,
                            PD_INFER_META(phi::GeneralUnaryGradInferMeta));
REGISTER_OPERATOR(gather_grad,
                  ops::GatherGradOp,
                  ops::GatherGradNoNeedBufferVarInferer,
                  GatherGradInferShapeFunctor);

REGISTER_OP_VERSION(gather).AddCheckpoint(
    R"ROC(upgrad gather, add a new input [Axis])ROC",
    paddle::framework::compatible::OpVersionDesc().NewInput(
        "Axis", "Specify the axis of gather operation."));
