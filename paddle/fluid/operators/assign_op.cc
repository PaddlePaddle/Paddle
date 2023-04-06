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

#include "paddle/fluid/operators/assign_op.h"

#include <string>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/prim/api/composite_backward/composite_backward_api.h"
#include "paddle/fluid/prim/utils/static/composite_grad_desc_maker.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"
namespace paddle {
namespace framework {
class OpDesc;
class Variable;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

class AssignOp : public framework::OperatorWithKernel {
 public:
  AssignOp(const std::string &type,
           const framework::VariableNameMap &inputs,
           const framework::VariableNameMap &outputs,
           const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    return phi::KernelKey(phi::Backend::ALL_BACKEND,
                          tensor.layout(),
                          expected_kernel_type.dtype());
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    const framework::Variable *var = ctx.InputVar("X");
    if (var->IsType<framework::LoDTensorArray>()) {
      auto t_arr = var->Get<framework::LoDTensorArray>();
      // NOTE(liym27): Support an empty tensor array as Input.
      // And set the kernel type is float.
      if (t_arr.size() == 0) {
        return phi::KernelKey(framework::proto::VarType::FP32,
                              ctx.device_context().GetPlace());
      }
    }

    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.device_context().GetPlace());
  }
};

class AssignInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SyncTypeAndDataType("X", "Out");
  }
};

class AssignOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(phi::DenseTensor, SelectedRows or phi::DenseTensorArray) The input "
        "variable "
        "could be phi::DenseTensor, SelectedRows or phi::DenseTensorArray.")
        .AsDispensable();
    AddOutput("Out",
              "(phi::DenseTensor, SelectedRows or phi::DenseTensorArray) The "
              "type of output "
              "is the same as input X.");
    AddComment(R"DOC(Assign Operator

Out = X,  when type in [phi::DenseTensor/SelectedRows/phi::DenseTensorArray]
raise error if the type is not listed above.
)DOC");
  }
};

template <typename T>
class AssignGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("assign");
    op->SetInput("X", this->OutputGrad("Out"));
    op->SetOutput("Out", this->InputGrad("X"));
  }
};

class AssignCompositeGradOpMaker : public prim::CompositeGradOpMakerBase {
  using prim::CompositeGradOpMakerBase::CompositeGradOpMakerBase;

 public:
  void Apply() override {
    paddle::Tensor out_grad = this->GetSingleOutputGrad("Out");
    paddle::Tensor input_grad = this->GetSingleInputGrad("X");

    auto dx_ptr = this->GetOutputPtr(&input_grad);
    std::string dx_name = this->GetOutputName(input_grad);

    VLOG(6) << "Running assign_grad composite func";
    prim::assign_grad<prim::DescTensor>(out_grad, dx_ptr);
    this->RecoverOutputName(input_grad, dx_name);
  }
};

DECLARE_INPLACE_OP_INFERER(AssignOpInplaceInferer, {"X", "Out"});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

DECLARE_INFER_SHAPE_FUNCTOR(assign,
                            AssignInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));
REGISTER_OPERATOR(assign,
                  ops::AssignOp,
                  ops::AssignCompositeGradOpMaker,
                  ops::AssignGradMaker<paddle::framework::OpDesc>,
                  ops::AssignGradMaker<paddle::imperative::OpBase>,
                  ops::AssignOpProtoMaker,
                  ops::AssignOpInplaceInferer,
                  ops::AssignInferVarType,
                  AssignInferShapeFunctor);
