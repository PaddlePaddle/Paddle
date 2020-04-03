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

#include <memory>
#include <string>

namespace paddle {
namespace operators {

class AssignOp : public framework::OperatorWithKernel {
 public:
  AssignOp(const std::string &type, const framework::VariableNameMap &inputs,
           const framework::VariableNameMap &outputs,
           const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    if (ctx->HasInput("X")) {
      auto type = ctx->GetInputsVarType("X")[0];
      if (type == framework::proto::VarType::SELECTED_ROWS ||
          type == framework::proto::VarType::LOD_TENSOR) {
        ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
        if (type == framework::proto::VarType::LOD_TENSOR) {
          ctx->ShareLoD("X", /*->*/ "Out");
        }
      }
    }
  }

 protected:
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   expected_kernel_type.place_,
                                   tensor.layout());
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class AssignKernel {
 public:
  void operator()(const framework::ExecutionContext &ctx) const {
    auto *x = ctx.InputVar("X");
    if (x == nullptr) {
      return;
    }
    auto *out = ctx.OutputVar("Out");
    PADDLE_ENFORCE(
        out != nullptr,
        "The Output(Out) should not be null if the Input(X) is set.");
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(ctx.GetPlace());

    framework::VisitVarType(*x, AssignFunctor(out, dev_ctx));
  }
};

class AssignOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor, SelectedRows or LoDTensorArray) The input variable "
             "could be LoDTensor, SelectedRows or LoDTensorArray.")
        .AsDispensable();
    AddOutput("Out",
              "(LoDTensor, SelectedRows or LoDTensorArray) The type of output "
              "is the same as input X.");
    AddComment(R"DOC(Assign Operator

Out = X,  when type in [LoDTensor/SelectedRows/LoDTensorArray]
raise error if the type is not listed above.
)DOC");
  }
};

template <typename T>
class AssignGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    auto *op = new T();
    op->SetType("assign");
    op->SetInput("X", this->OutputGrad("Out"));
    op->SetOutput("Out", this->InputGrad("X"));
    return std::unique_ptr<T>(op);
  }
};

DECLARE_INPLACE_OP_INFERER(AssignOpInplaceInferer, {"X", "Out"});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OPERATOR(assign, ops::AssignOp,
                  ops::AssignGradMaker<paddle::framework::OpDesc>,
                  ops::AssignGradMaker<paddle::imperative::OpBase>,
                  ops::AssignOpProtoMaker, ops::AssignOpInplaceInferer);

REGISTER_OP_CPU_KERNEL_FUNCTOR(assign, float, ops::AssignKernel, double,
                               ops::AssignKernel, int, ops::AssignKernel,
                               int64_t, ops::AssignKernel, bool,
                               ops::AssignKernel, plat::float16,
                               ops::AssignKernel);

#ifdef PADDLE_WITH_CUDA
REGISTER_OP_CUDA_KERNEL_FUNCTOR(assign, float, ops::AssignKernel, double,
                                ops::AssignKernel, int, ops::AssignKernel,
                                int64_t, ops::AssignKernel, bool,
                                ops::AssignKernel, plat::float16,
                                ops::AssignKernel);
#endif
