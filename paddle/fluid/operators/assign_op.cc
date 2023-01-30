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
<<<<<<< HEAD
  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    return phi::KernelKey(phi::Backend::ALL_BACKEND,
                          tensor.layout(),
                          expected_kernel_type.dtype());
  }

  phi::KernelKey GetExpectedKernelType(
=======
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name,
      const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   expected_kernel_type.place_,
                                   tensor.layout());
  }

  framework::OpKernelType GetExpectedKernelType(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      const framework::ExecutionContext &ctx) const override {
    const framework::Variable *var = ctx.InputVar("X");
    if (var->IsType<framework::LoDTensorArray>()) {
      auto t_arr = var->Get<framework::LoDTensorArray>();
      // NOTE(liym27): Support an empty tensor array as Input.
      // And set the kernel type is float.
      if (t_arr.size() == 0) {
<<<<<<< HEAD
        return phi::KernelKey(framework::proto::VarType::FP32,
                              ctx.device_context().GetPlace());
      }
    }

    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.device_context().GetPlace());
=======
        return framework::OpKernelType(framework::proto::VarType::FP32,
                                       ctx.device_context());
      }
    }

    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
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
=======
    AddInput("X",
             "(LoDTensor, SelectedRows or LoDTensorArray) The input variable "
             "could be LoDTensor, SelectedRows or LoDTensorArray.")
        .AsDispensable();
    AddOutput("Out",
              "(LoDTensor, SelectedRows or LoDTensorArray) The type of output "
              "is the same as input X.");
    AddComment(R"DOC(Assign Operator

Out = X,  when type in [LoDTensor/SelectedRows/LoDTensorArray]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
                  ops::AssignGradMaker<paddle::framework::OpDesc>,
                  ops::AssignGradMaker<paddle::imperative::OpBase>,
                  ops::AssignOpProtoMaker,
                  ops::AssignOpInplaceInferer,
                  ops::AssignInferVarType,
                  AssignInferShapeFunctor);
