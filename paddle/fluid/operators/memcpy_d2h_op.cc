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
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle::framework {
class OpDesc;
class InferShapeContext;
template <typename T>
class EmptyGradOpMaker;
}  // namespace paddle::framework
namespace paddle::imperative {
class OpBase;
}  // namespace paddle::imperative

namespace paddle::operators {

class MemcpyD2HOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

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
                          ctx.device_context().GetPlace());
  }
};

class MemcpyD2HInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SyncTypeAndDataType("X", "Out");
  }
};

class MemcpyD2HOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(phi::DenseTensor) The input variable ");
    AddOutput("Out",
              "(phi::DenseTensor) The type of output "
              "is the same as input X.");
    AddAttr<int>("dst_place_type",
                 "Determine the dst place of tensor copy. "
                 "By Now it ONLY support XPU/CUDAPlace <-> CUDAPinnedPlace/CPU"
                 "Other place type is Unimplemented and will cause ERROR."
                 "0: dst is on CPUPlace. "
                 "1: dst is on CUDAPinnedPlace. ");
    AddComment(R"DOC(
    MemcpyD2H Operator.
    By now, it ONLY supports the memcopy between CUDAPlace <-> CUDAPinnedPlace/CPU.
    You would have to update it if you want other more capacities.
Out = X,  when type in [phi::DenseTensor]
raise error if the type is not listed above.
)DOC");
  }
};

}  // namespace paddle::operators

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(memcpy_d2h,
                            MemcpyD2HInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

REGISTER_OPERATOR(
    memcpy_d2h,
    ops::MemcpyD2HOp,
    ops::MemcpyD2HOpProtoMaker,
    ops::MemcpyD2HInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    MemcpyD2HInferShapeFunctor);
