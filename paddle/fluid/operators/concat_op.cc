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

#include "paddle/fluid/operators/concat_op.h"

#include <paddle/fluid/platform/complex.h>

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/prim/api/composite_backward/composite_backward_api.h"
#include "paddle/fluid/prim/utils/static/composite_grad_desc_maker.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"

namespace paddle {
namespace operators {

class ConcatOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto inputs = ctx.MultiInput<phi::DenseTensor>("X");
    auto input_data_type = framework::proto::VarType::Type(0);
    bool flag = 0;
    for (auto *input : inputs) {
      if (input->IsInitialized()) {
        input_data_type = framework::TransToProtoVarType(input->dtype());
        flag = 1;
        break;
      }
    }
    if (flag == 0) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "All Inputs of Concat OP are Empty!"));
    }
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    if (var_name == "AxisTensor") {
      return phi::KernelKey(phi::Backend::ALL_BACKEND,
                            expected_kernel_type.layout(),
                            expected_kernel_type.dtype());
    }
    return phi::KernelKey(
        tensor.place(), tensor.layout(), expected_kernel_type.dtype());
  }
};

class ConcatOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input tensors of concat operator.").AsDuplicable();
    AddOutput("Out", "Output tensor of concat operator.");
    AddAttr<int>("axis",
                 "The axis along which the input tensors will be concatenated."
                 "The axis could also be negative numbers. Negative axis is "
                 "interpreted as counting from the end of the rank."
                 "i.e., axis + rank(X) th dimension.")
        .SetDefault(0)
        .SupportTensor();
    AddInput("AxisTensor",
             "(Tensor) The axis along which the input tensors will be "
             "concatenated.  "
             "It has higher priority than Attr(axis). "
             "The shape of AxisTensor must be [1].")
        .AsDispensable();
    AddComment(R"DOC(
Concat Operator.

Concatenate the input tensors along dimension axis.
Examples:
  Input[0] = [[1,2],[3,4]]
  Input[1] = [[5,6]]
  axis = 0
  Output = [[1,2],
            [3,4],
            [5,6]]

)DOC");
  }
};

class ConcatOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto in_x = "X";
    auto out_x_g_n = framework::GradVarName(in_x);
    ctx->SetOutputsDim(out_x_g_n, ctx->GetInputsDim(in_x));

    ctx->ShareAllLoD(in_x, out_x_g_n);
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    if (var_name == "AxisTensor") {
      return phi::KernelKey(phi::Backend::ALL_BACKEND,
                            expected_kernel_type.layout(),
                            expected_kernel_type.dtype());
    }
    return phi::KernelKey(
        tensor.place(), tensor.layout(), expected_kernel_type.dtype());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ConcatOpGradNoNeedBufferVarInferer, "X");

template <typename T>
class ConcatGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("concat_grad");
    op->SetInput("X", this->Input("X"));
    if (this->HasInput("AxisTensor")) {
      op->SetInput("AxisTensor", this->Input("AxisTensor"));
    }
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X", false));
    op->SetAttrMap(this->Attrs());
  }
};

class ConcatCompositeGradOpMaker : public prim::CompositeGradOpMakerBase {
  using prim::CompositeGradOpMakerBase::CompositeGradOpMakerBase;

 public:
  void Apply() override {
    std::vector<paddle::Tensor> input = this->GetMultiForwardInput("X");
    paddle::optional<paddle::Tensor> tensor_axis =
        this->GetOptionalSingleForwardInput("AxisTensor");
    paddle::Tensor out_grad = this->GetSingleOutputGrad("Out");
    std::vector<paddle::Tensor> input_grad = this->GetMultiInputGrad("X");

    std::vector<paddle::Tensor *> input_grad_ptr;
    for (auto i = 0; i < static_cast<int>(input_grad.size()); ++i) {
      input_grad_ptr.push_back(&input_grad[i]);
    }
    int axis = static_cast<int>(this->Attr<int>("axis"));
    std::vector<paddle::Tensor *> dx_ptr = this->GetOutputPtr(input_grad_ptr);
    std::vector<std::string> dx_name = this->GetOutputName(input_grad);

    VLOG(6) << "Runing concat_grad composite func";
    if (tensor_axis.is_initialized()) {
      PADDLE_THROW(platform::errors::Unimplemented(
          "We don't support dynamic index from tensor for concat composite "
          "grad for now. "));
    } else {
      prim::concat_grad<prim::DescTensor>(input, out_grad, axis, dx_ptr);
    }
    this->RecoverOutputName(input_grad, dx_name);
  }
};

template <typename T>
class ConcatDoubleGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("concat");
    grad_op->SetInput("X", this->OutputGrad(framework::GradVarName("X")));
    grad_op->SetOutput("Out", this->InputGrad(framework::GradVarName("Out")));
    grad_op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(concat,
                            ConcatInferShapeFunctor,
                            PD_INFER_META(phi::ConcatInferMeta));

REGISTER_OPERATOR(concat,
                  ops::ConcatOp,
                  ops::ConcatOpMaker,
                  ops::ConcatGradOpMaker<paddle::framework::OpDesc>,
                  ops::ConcatGradOpMaker<paddle::imperative::OpBase>,
                  ops::ConcatCompositeGradOpMaker,
                  ConcatInferShapeFunctor);
REGISTER_OPERATOR(concat_grad,
                  ops::ConcatOpGrad,
                  ops::ConcatDoubleGradOpMaker<paddle::framework::OpDesc>,
                  ops::ConcatDoubleGradOpMaker<paddle::imperative::OpBase>,
                  ops::ConcatOpGradNoNeedBufferVarInferer);
