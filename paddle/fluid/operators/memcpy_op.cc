/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace framework {
class OpDesc;
class InferShapeContext;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

class MemcpyOp : public framework::OperatorWithKernel {
 public:
  MemcpyOp(const std::string &type,
           const framework::VariableNameMap &inputs,
           const framework::VariableNameMap &outputs,
           const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto type = ctx->GetInputsVarType("X")[0];
    if (type == framework::proto::VarType::SELECTED_ROWS ||
        type == framework::proto::VarType::DENSE_TENSOR) {
      ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
      if (type == framework::proto::VarType::DENSE_TENSOR) {
        ctx->ShareLoD("X", /*->*/ "Out");
      }
    }
  }

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
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.GetPlace());
  }
};

class MemcpyInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SyncTypeAndDataType("X", "Out");
  }
};
class MemcpyOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(phi::DenseTensor) The input variable ");
    AddOutput("Out",
              "(phi::DenseTensor) The type of output "
              "is the same as input X.");
    AddAttr<int>("dst_place_type",
                 "Determine the dst place of tensor copy. "
                 "By Now it ONLY support CUDAPlace <-> CUDAPinnedPlace."
                 "Other place type is Unimplemented and will cause ERROR."
                 "0: dst is on CPUPlace. "
                 "1: dst is on CUDAPlace. "
                 "2: dst is on CUDAPinnedPlace. "
                 "3: dst is on XPUPlace. "
                 "5: dst is on CustomDevicePlace");
    AddComment(R"DOC(
    Memcpy Operator.
    By now, it ONLY supports the memcopy between CUDAPinnedPlace <-> CUDAPlace, and used as an internal op by Recompute-Offload.
    You would have to update it if you want other more capacities.

Out = X,  when type in [phi::DenseTensor]
raise error if the type is not listed above.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(memcpy,
                            MemcpyInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

REGISTER_OPERATOR(
    memcpy,
    ops::MemcpyOp,
    ops::MemcpyOpProtoMaker,
    ops::MemcpyInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    MemcpyInferShapeFunctor);
