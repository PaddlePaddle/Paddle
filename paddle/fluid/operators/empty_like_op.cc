/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/empty_like_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class EmptyLikeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(empty_like operator
Returns a tensor filled with uninitialized data. The shape of the tensor is
defined by the variable argument `input.shape`.

The type of the tensor is specify by `input.dtype`.
)DOC");
    AddInput("X", "The input of empty_like op.");
    AddOutput("Out", "(Tensor) The output tensor.");
    AddAttr<int>("dtype",
                 "Output tensor data type. defalut value is -1,"
                 "according to the input dtype.")
        .SetDefault(-1);
  }
};

class EmptyLikeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "empty_like");
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out", "empty_like");
    context->SetOutputDim("Out", context->GetInputDim("X"));
    context->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &context) const override {
    framework::OpKernelType kt =
        OperatorWithKernel::GetExpectedKernelType(context);
    const auto &data_type = context.Attr<int>("dtype");
    if (data_type >= 0) {
      kt.data_type_ = static_cast<framework::proto::VarType::Type>(data_type);
    }
    return kt;
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   expected_kernel_type.place_,
                                   tensor.layout());
  }
};

class EmptyLikeOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *context) const override {
    auto var_data_type = static_cast<framework::proto::VarType::Type>(
        BOOST_GET_CONST(int, context->GetAttr("dtype")));
    if (var_data_type < 0) {
      context->SetOutputDataType("Out", context->GetInputDataType("X"));
    } else {
      context->SetOutputDataType("Out", var_data_type);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(
    empty_like, ops::EmptyLikeOp, ops::EmptyLikeOpMaker,
    ops::EmptyLikeOpVarTypeInference,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(
    empty_like, ops::EmptyLikeKernel<plat::CPUDeviceContext, float>,
    ops::EmptyLikeKernel<plat::CPUDeviceContext, double>,
    ops::EmptyLikeKernel<plat::CPUDeviceContext, int64_t>,
    ops::EmptyLikeKernel<plat::CPUDeviceContext, int>,
    ops::EmptyLikeKernel<plat::CPUDeviceContext, plat::float16>);
